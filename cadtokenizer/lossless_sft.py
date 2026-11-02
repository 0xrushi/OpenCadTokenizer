"""Bounded lossless SEM SFT: frozen base vocabulary, small new rows and LoRA."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import statistics
import time

from .common import file_hash, fraction, load_checkpoint, read_config, require_container, save_checkpoint, seed_all, write_json
from .lossless_sem import LosslessSemCodec
from .representation import stringify, tokenize


def freeze_pilot(settings):
    root=Path(settings['output']); codec_root=Path(settings['codec_root'])
    cache=codec_root/'tokens.sqlite'
    audit=read_config(codec_root/'manifest.json'); database=audit['settings']['database']
    if file_hash(database)!=audit['database_sha256']:
        raise ValueError('Source data changed since codec audit')
    with sqlite3.connect(cache) as db:
        rows=db.execute('SELECT id,example_id,split,task,context_overflow FROM samples ORDER BY example_id').fetchall()
    selected={}
    for split,key in [('train','train_counts'),('val','validation_counts')]:
        selected[split]=[]
        for task,count in settings[key].items():
            candidates=[r for r in rows if r[2]==split and r[3]==task and not r[4]]
            candidates.sort(key=lambda r:(fraction(r[1],settings['seed']),r[1]))
            if len(candidates)<count: raise ValueError('Insufficient pilot cases')
            selected[split].extend({'cache_id':r[0],'example_id':r[1],'task':r[3]} for r in candidates[:count])
    components={'train':set(),'val':set()}; validation=[]
    with sqlite3.connect(database) as db:
        for split,cases in selected.items():
            for case in cases:
                instruction,source_id,target_id,stored=db.execute('SELECT instruction,source_id,target_id,split FROM examples WHERE id=?',(case['example_id'],)).fetchone()
                if stored!=split: raise ValueError('Example split mismatch')
                for endpoint in [target_id]+([source_id] if source_id else []):
                    component,stored=db.execute('SELECT component,split FROM geometry WHERE id=?',(endpoint,)).fetchone()
                    if stored!=split: raise ValueError('Endpoint split mismatch')
                    components[split].add(component)
                if split=='val':
                    target=db.execute('SELECT sequence FROM geometry WHERE id=?',(target_id,)).fetchone()[0]
                    source=db.execute('SELECT sequence FROM geometry WHERE id=?',(source_id,)).fetchone()[0] if source_id else None
                    validation.append({**case,'instruction':instruction,'source':source,'target':target})
    if components['train'] & components['val']: raise ValueError('Component overlap')
    model_root=Path(settings['model_path'])
    shards=sorted(set(read_config(model_root/'model.safetensors.index.json')['weight_map'].values()))
    signature={'settings':settings,'cache_sha256':file_hash(cache),'codec_sha256':file_hash(codec_root/'codec.json'),
        'source_database_sha256':audit['database_sha256'],
        'model_sha256':{name:file_hash(model_root/name) for name in ['config.json',*shards]},
        'code_sha256':{str(p):file_hash(p) for p in sorted(Path('cadtokenizer').glob('*.py'))},
        'image_id':os.environ.get('CADTOKENIZER_IMAGE_ID')}
    manifest={'signature':signature,'train':selected['train'],'validation':validation,
        'design':f"Preselected {len(selected['train'])} training and {len(validation)} component-disjoint validation examples. Only new token rows and LoRA train. Fixed {settings['max_updates']}-update endpoint; no promotion or test cases."}
    path=root/'manifest.json'
    if path.exists() and read_config(path)!=manifest: raise ValueError('Pilot provenance changed; use a fresh output')
    write_json(path,manifest)
    return manifest


def load_samples(settings,cases):
    result=[]
    with sqlite3.connect(Path(settings['codec_root'])/'tokens.sqlite') as db:
        for case in cases:
            text,prefix,overflow=db.execute('SELECT input_ids,prompt_length,context_overflow FROM samples WHERE id=?',(case['cache_id'],)).fetchone()
            ids=json.loads(text)
            if overflow or len(ids)>settings['context_length']: raise ValueError('Truncation forbidden')
            result.append({'input_ids':ids,'labels':[-100]*prefix+ids[prefix:],'prompt_length':prefix})
    return result


def build_model(settings,codec,resume=None):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import LoraConfig,PeftModel,get_peft_model
    from .lossless_model import install_rows,semantic_row_initialization
    tokenizer=AutoTokenizer.from_pretrained(settings['model_path'],local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(settings['model_path'],local_files_only=True,
        dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
    inputs,outputs,initialization=semantic_row_initialization(base,tokenizer,codec)
    install_rows(base,inputs,outputs)
    if resume:
        model=PeftModel.from_pretrained(base,str(resume/'adapter'),is_trainable=True)
    else:
        cfg=LoraConfig(r=settings['lora_rank'],lora_alpha=settings['lora_alpha'],target_modules=settings['lora_targets'],
            lora_dropout=0.,bias='none',modules_to_save=['cad_rows'],task_type='CAUSAL_LM')
        model=get_peft_model(base,cfg)
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    trainable={name:list(p.shape) for name,p in model.named_parameters() if p.requires_grad}
    if any('lora_' not in name and 'cad_rows.modules_to_save.' not in name for name in trainable):
        raise ValueError('Original vocabulary or backbone unexpectedly trainable')
    names=[name for name in trainable if 'cad_rows.modules_to_save.' in name]
    if len(names)!=2 or any(trainable[name][0]!=79 for name in names):
        raise ValueError('Both 79-row extensions must train')
    return model,initialization,trainable


def optimizer_for(model,settings):
    import torch
    groups=[];names=[]
    for row_group,lr in [(False,settings['learning_rate']),(True,settings['token_learning_rate'])]:
        chosen=[(name,p) for name,p in model.named_parameters() if p.requires_grad and ('cad_rows.modules_to_save.' in name)==row_group]
        groups.append({'params':[p for _,p in chosen],'lr':lr,'initial_lr':lr,'weight_decay':0. if row_group else settings['weight_decay']})
        names.append([name for name,_ in chosen])
    return torch.optim.AdamW(groups),names


def save_pilot(model,optimizer,settings,manifest,step,epoch,offset,param_names):
    import torch
    path=Path(settings['output'])/'checkpoints'/f'step-{step:06d}'
    temp=path.with_name(path.name+'.tmp')
    if path.exists(): raise FileExistsError('Immutable checkpoint exists')
    if temp.exists(): shutil.rmtree(temp)
    temp.mkdir(parents=True)
    model.save_pretrained(temp/'adapter',save_embedding_layers=False)
    save_checkpoint(temp/'state.pt',{'optimizer':optimizer.state_dict(),'step':step,'epoch':epoch,'offset':offset,
        'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
        'signature':manifest['signature'],'parameter_names':param_names})
    write_json(temp/'complete.json',{'step':step,'trainable_token_rows':79})
    temp.rename(path)


def teacher_evaluate(model,samples,collator):
    import torch
    total=loss_sum=0;model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for sample in samples:
            batch={k:v.cuda() for k,v in collator([sample]).items()}
            count=int(batch['labels'][:,1:].ne(-100).sum());value=float(model(**batch).loss)
            if not math.isfinite(value): raise FloatingPointError('Non-finite validation loss')
            total+=count;loss_sum+=value*count
    return {'loss':loss_sum/total,'target_tokens':total,'examples':len(samples)}


def generate_one(model,prompt,codec,settings):
    """Only the prompt enters generation, never target CAD or its pair count."""
    import torch
    from transformers import LogitsProcessor,LogitsProcessorList
    from .lossless_sem import LosslessCadConstraint
    constraint=LosslessCadConstraint(codec,max_pairs=settings['max_pairs']);prefix=len(prompt)
    class Mask(LogitsProcessor):
        def __call__(self,input_ids,scores):
            generated=input_ids[0,prefix:].tolist();allowed=constraint.allowed(generated)
            if settings['max_new_tokens']-len(generated)<36:
                allowed-={codec.native_ids[token-1] for token in (1,2,3)}
            if not allowed: raise ValueError('No legal CAD continuation within budget')
            mask=torch.full_like(scores,-torch.inf);mask[:,sorted(allowed)]=0
            return scores+mask
    ids=torch.tensor([prompt],device='cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        output=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,
            max_new_tokens=settings['max_new_tokens'],use_cache=True,eos_token_id=codec.boundaries['end'],
            pad_token_id=128001,logits_processor=LogitsProcessorList([Mask()]))
    continuation=output[0,prefix:].tolist()
    return codec.decode([codec.boundaries['start']]+continuation),continuation


def generation_evaluate(model,cases,samples,codec,settings,step):
    import numpy as np
    from scipy.spatial import cKDTree
    from .bench_runtime import KernelRunner
    from .bench_metrics import normalize_points,cadfusion_f1
    root=Path(settings['output'])/'generation'/f'step-{step:06d}';root.mkdir(parents=True,exist_ok=True)
    if (root/'results.json').exists(): return read_config(root/'results.json')
    rows=[];journal=root/'predictions.jsonl'
    existing={r['id']:r for r in [json.loads(line) for line in journal.read_text().splitlines()]} if journal.exists() else {}
    model.eval()
    with journal.open('a') as stream:
        for case,sample in zip(cases,samples):
            key=case['example_id']
            if key in existing: rows.append(existing[key]);continue
            row={'id':key,'task':case['task'],'sequence':None,'error':None}
            try:
                row['sequence'],row['generated_ids']=generate_one(model,sample['input_ids'][:sample['prompt_length']],codec,settings)
            except ValueError as exc: row['error']=str(exc)
            stream.write(json.dumps(row)+'\n');stream.flush();rows.append(row)
            print(json.dumps({'phase':'generation','step':step,'completed':len(rows),'total':len(cases)}),flush=True)
    cfg={**read_config('configs/benchmark.json'),'output':str(Path(settings['output'])/'geometry-cache'),
        'render':False,'point_count':settings['point_count'],'seed':settings['seed']}
    runner=KernelRunner(cfg)
    jobs=[(c['example_id'],'reference',c['target']) for c in cases]+[(r['id'],'prediction',r['sequence']) for r in rows]
    jobs=list(dict.fromkeys(jobs));geometry={}
    with ThreadPoolExecutor(max_workers=settings['kernel_workers']) as pool:
        for job,value in zip(jobs,pool.map(lambda j:runner.reconstruct(j[2],j[0],j[1]),jobs)): geometry[job]=value
    for case,row in zip(cases,rows):
        ref=geometry[(row['id'],'reference',case['target'])];pred=geometry[(row['id'],'prediction',row['sequence'])]
        row.update(reference_geometry=ref,prediction_geometry=pred,chamfer=None,relative_volume_error=None)
        row['exact_sem']=row['sequence']==stringify(tokenize(case['target']))
        row['copies_source']=case['source'] is not None and row['sequence']==stringify(tokenize(case['source']))
        if ref['status']==pred['status']=='valid':
            a,b=[np.asarray(normalize_points(np.load(g['points']))) for g in (ref,pred)]
            row['chamfer']=float(np.mean(cKDTree(b).query(a)[0]**2)+np.mean(cKDTree(a).query(b)[0]**2))
            row['relative_volume_error']=abs(pred['volume']-ref['volume'])/ref['volume']
    summaries={}
    for task in ('all','generation','editing'):
        chosen=[r for r in rows if task=='all' or r['task']==task];cd=[r['chamfer'] for r in chosen if r['chamfer'] is not None]
        summaries[task]={'cases':len(chosen),'syntax_decoded':sum(r['sequence'] is not None for r in chosen),
            'known_valid_references':sum(r['reference_geometry']['status']=='valid' for r in chosen),
            'valid_predictions':sum(r['prediction_geometry']['status']=='valid' for r in chosen),
            'valid_of_valid_references':len(cd),'mean_chamfer':statistics.mean(cd) if cd else None,
            'median_volume_error':statistics.median(r['relative_volume_error'] for r in chosen if r['relative_volume_error'] is not None) if cd else None,
            'exact_sem':sum(r['exact_sem'] for r in chosen),'copies_source':sum(r['copies_source'] for r in chosen)}
    result={'step':step,'summaries':summaries,'count_f1':cadfusion_f1([(c['target'],r['sequence']) for c,r in zip(cases,rows)]),
        'limits':'60 fixed validation prompts; one short pilot, no existing-model comparison, test cases or paper-level claim.'}
    write_json(root/'per_case.json',rows);write_json(root/'results.json',result)
    return result


def run_lossless_sft(config_path,mode='train'):
    require_container();settings=read_config(config_path)
    if mode=='inspect':
        import inspect
        from peft import LoraConfig
        print(inspect.signature(LoraConfig),flush=True);return
    if mode=='smoke':
        settings={**settings,'output':settings['output']+'-smoke','max_updates':2,'teacher_evaluation_updates':[0,2],
            'generation_evaluation_updates':[],'checkpoint_updates':[2],'logging_updates':1}
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from .sft import InstructionCollator
    from .tokenizer_learning_curve import epoch_order,preserve_rng
    root=Path(settings['output']);root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('a') as lock,sdpa_kernel(SDPBackend.MATH):
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
            torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 CUDA required')
            write_json(root/'state.json',{'status':'running','phase':'freeze'})
            manifest=freeze_pilot(settings)
            if (root/'results.json').exists(): return read_config(root/'results.json')
            (root/'source').mkdir(exist_ok=True)
            for p in Path('cadtokenizer').glob('*.py'): shutil.copy2(p,root/'source'/p.name)
            codec=LosslessSemCodec.from_dict(read_config(Path(settings['codec_root'])/'codec.json'))
            train=load_samples(settings,manifest['train']);validation=load_samples(settings,manifest['validation'])
            seed_all(settings['seed'])
            checkpoints=sorted(p for p in (root/'checkpoints').glob('step-*') if (p/'complete.json').exists())
            resume=checkpoints[-1] if checkpoints else None
            write_json(root/'state.json',{'status':'running','phase':'load_model'})
            model,initialization,trainable=build_model(settings,codec,resume)
            write_json(root/'token_initialization.json',initialization)
            write_json(root/'trainable_parameters.json',{'parameters':trainable,
                'count':sum(p.numel() for p in model.parameters() if p.requires_grad),'new_rows':79,'frozen_original_vocabulary':True})
            optimizer,param_names=optimizer_for(model,settings);step=epoch=offset=0
            if resume:
                state=load_checkpoint(resume/'state.pt')
                if state['signature']!=manifest['signature'] or state['parameter_names']!=param_names: raise ValueError('Checkpoint mismatch')
                optimizer.load_state_dict(state['optimizer']);step,epoch,offset=(state[k] for k in ('step','epoch','offset'))
                torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);del state
            collator=InstructionCollator(128001)
            def evaluate(current):
                if current in settings['teacher_evaluation_updates']:
                    path=root/'teacher'/f'step-{current:06d}.json'
                    if not path.exists():
                        write_json(root/'state.json',{'status':'running','phase':'teacher_validation','step':current})
                        with preserve_rng(): report=teacher_evaluate(model,validation,collator)
                        write_json(path,report)
                if current in settings['generation_evaluation_updates']:
                    write_json(root/'state.json',{'status':'running','phase':'free_running_validation','step':current})
                    with preserve_rng(): generation_evaluate(model,manifest['validation'],validation,codec,settings,current)
            evaluate(step);order=epoch_order(len(train),settings['seed'],epoch)
            started=time.monotonic();start_step=step;torch.cuda.reset_peak_memory_stats()
            while step<settings['max_updates']:
                chosen=order[offset:offset+settings['effective_batch_size']];samples=[train[i] for i in chosen]
                count=sum(sum(value!=-100 for value in s['labels'][1:]) for s in samples)
                factor=min(1.,(step+1)/settings['warmup_updates'])*.5*(1+math.cos(math.pi*step/settings['max_updates']))
                for group in optimizer.param_groups: group['lr']=group['initial_lr']*factor
                model.train();model.config.use_cache=False;optimizer.zero_grad(set_to_none=True);loss_sum=0.
                for sample in samples:
                    batch={k:v.cuda() for k,v in collator([sample]).items()};tokens=int(batch['labels'][:,1:].ne(-100).sum())
                    with torch.autocast('cuda',dtype=torch.bfloat16): output=model(**batch);loss=output.loss*tokens/count
                    if not bool(torch.isfinite(loss)): raise FloatingPointError('Non-finite training loss')
                    loss.backward();loss_sum+=float(loss.detach());del output,loss,batch
                if step==0:
                    checks={name:bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0)
                        for name,p in model.named_parameters() if p.requires_grad and 'cad_rows.modules_to_save.' in name}
                    write_json(root/'first_update_row_gradients.json',checks)
                    if not all(checks.values()): raise RuntimeError('New token rows did not receive gradients')
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                optimizer.step();step+=1;offset+=len(chosen)
                if offset==len(train): epoch+=1;offset=0;order=epoch_order(len(train),settings['seed'],epoch)
                if step%settings['logging_updates']==0:
                    wall=time.monotonic()-started
                    metrics={'step':step,'loss':loss_sum,'gradient_norm':float(norm),'epoch':epoch,'offset':offset,
                        'seconds_per_update':wall/(step-start_step),'wall_seconds':wall,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
                    with (root/'training.jsonl').open('a') as stream: stream.write(json.dumps(metrics)+'\n')
                    write_json(root/'state.json',{'status':'running','phase':'training',**metrics});print(json.dumps(metrics),flush=True)
                if step in settings['checkpoint_updates']: save_pilot(model,optimizer,settings,manifest,step,epoch,offset,param_names)
                evaluate(step)
            teacher={str(i):read_config(root/'teacher'/f'step-{i:06d}.json') for i in settings['teacher_evaluation_updates']}
            generations={str(i):read_config(root/'generation'/f'step-{i:06d}'/'results.json') for i in settings['generation_evaluation_updates']}
            result={'updates':step,'teacher':teacher,'generation':generations,'train_examples':len(train),
                'validation_examples':len(validation),'production_promoted':False}
            write_json(root/'results.json',result)
            (root/'findings.md').write_text('# Lossless SEM bounded SFT pilot\n\n'+json.dumps(result,indent=2)+'\n\nNo test cases or production promotion. This is a short downstream pilot, not a paper-level comparison.\n')
            write_json(root/'state.json',{'status':'complete','results':str(root/'results.json')})
        except Exception as exc:
            write_json(root/'state.json',{'status':'failed','error':repr(exc)});raise
