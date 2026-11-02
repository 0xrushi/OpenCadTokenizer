"""Matched ordinary/changed-number-weighted editing SFT with GPU profiling."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import random
import sqlite3
import statistics
import time

from .common import require_container,read_config,write_json,file_hash,seed_all,load_checkpoint
from .numeric_supervision import framed_changed_numbers,attach_weights


def prepare_data(s,root):
    from .numeric_supervision import changed_numbers
    from .lossless_sft import load_samples
    pilot=read_config(Path(s['parent']).parents[1]/'manifest.json');original=pilot['signature']
    for p,digest in [('data/prepared/cad.sqlite',original['source_database_sha256']),
                     (Path(s['codec_root'])/'tokens.sqlite',original['cache_sha256']),
                     (Path(s['codec_root'])/'codec.json',original['codec_sha256'])]:
        if file_hash(p)!=digest:raise ValueError('Frozen pilot data changed')
    for name,digest in original['model_sha256'].items():
        if file_hash(Path(s['model_path'])/name)!=digest:raise ValueError('Base model changed')
    for name in ('lossless_model.py','lossless_sem.py','lossless_sft.py'):
        p='cadtokenizer/'+name
        if file_hash(p)!=original['code_sha256'][p]:raise ValueError('Original model or inference implementation changed')
    if os.environ.get('CADTOKENIZER_IMAGE_ID')!=original['image_id']:raise ValueError('Image changed')
    cases=[c for c in pilot['train'] if c['task']=='editing']
    with sqlite3.connect('data/prepared/cad.sqlite') as db:
        for c in cases:
            a,b,split=db.execute('SELECT a.sequence,b.sequence,e.split FROM examples e JOIN geometry a ON a.id=e.source_id JOIN geometry b ON b.id=e.target_id WHERE e.id=?',(c['example_id'],)).fetchone()
            if split!='train':raise ValueError('Editing training split mismatch')
            c.update(source=a,target=b,flags=framed_changed_numbers(a,b))
    if len(cases)!=2560:raise ValueError('Expected original 2560 editing examples')
    raw=load_samples(s,cases)
    for c,x in zip(cases,raw):attach_weights(x,c['flags'],4.) # validate native framing lengths
    expected=read_config('artifacts/lossless-edit-supervision-v1/results.json')['numeric_supervision_by_instruction_first_word']['all']['changed_numeric_target_fields']
    if sum(sum(c['flags']) for c in cases)!=expected:raise ValueError('New numeric mask disagrees with the earlier field audit')
    batches=[];epoch=0
    while len(batches)<s['max_updates']:
        ids=list(range(len(cases)));random.Random(s['batch_order_seed']+epoch*1009).shuffle(ids)
        for i in range(0,len(ids),s['effective_batch_size']):
            if len(batches)==s['max_updates']:break
            batches.append([cases[j]['example_id'] for j in ids[i:i+s['effective_batch_size']]])
        epoch+=1
    validation=read_config('artifacts/lossless-editing-validation-v1/manifest.json')['cases']
    by_id={c['example_id']:c for c in cases}
    recall=[{**by_id[key],'group':'add_numeric' if i<16 else 'add_no_numeric' if i<32 else 'remove'}
            for i,key in enumerate(read_config(s['recall_plan'])['sample_ids'])]
    data={'train':cases,'batches':batches,'validation':validation,'teacher':[c for c in pilot['validation'] if c['task']=='editing'],
        'recall':recall,'changed_numeric_fields':expected}
    path=root/'data.json'
    if path.exists() and read_config(path)!=data:raise ValueError('Frozen numeric study cases changed')
    write_json(path,data)
    return data,raw


def backward_batch(model,samples,microbatch):
    import torch
    from .numeric_supervision import collate_weighted,weighted_numerator
    denominator=sum(sum(x['weights'][1:]) for x in samples)
    if denominator<=0:raise ValueError('No active weighted targets')
    total=0.
    for i in range(0,len(samples),microbatch):
        batch={k:v.cuda() for k,v in collate_weighted(samples[i:i+microbatch]).items()}
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False)
            loss=weighted_numerator(output.logits,batch['labels'],batch['weights'])/denominator
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Non-finite weighted editing loss')
        loss.backward();total+=float(loss.detach());del output,loss,batch
    return total


def run_profile(s,root,data,raw):
    import torch
    from .lossless_sft import build_model
    from .lossless_sem import LosslessSemCodec
    from .routed_release import GpuMonitor
    from .bench_data import code_hash
    codec=LosslessSemCodec.from_dict(read_config(Path(s['codec_root'])/'codec.json'))
    seed_all(s['seed']);model,_,_=build_model(s,codec,Path(s['parent']))
    weighted=[attach_weights(x,c['flags'],s['numeric_multiplier']) for c,x in zip(data['train'],raw)]
    groups=[sorted(weighted,key=lambda x:len(x['input_ids']),reverse=True)[:8],
            sorted(weighted,key=lambda x:len(x['input_ids'])-x['prompt_length'],reverse=True)[:8]]
    # Forward/backward only: no optimizer step, no checkpoint, no inherited state.
    projected=sum(p.numel()*4*2 for p in model.parameters() if p.requires_grad)
    records=[]
    for size in s['microbatch_candidates']:
        model.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        error=None;losses=[];model.train();model.config.use_cache=False
        with GpuMonitor() as monitor:
            started=time.monotonic()
            try:
                for group in groups:
                    model.zero_grad(set_to_none=True);losses.append(backward_batch(model,group,size))
                torch.cuda.synchronize()
            except torch.OutOfMemoryError as exc:error=str(exc)
            elapsed=time.monotonic()-started
        peak=torch.cuda.max_memory_allocated();metrics=monitor.report()
        total_peak=(metrics['peak_total_gpu_memory_mib'] or peak/2**20)*2**20+projected
        row={'microbatch':size,'seconds':elapsed,'examples_per_second':16/elapsed if error is None else None,
            'peak_cuda_allocated_bytes':peak,'projected_optimizer_state_bytes':projected,
            'projected_total_peak_bytes':total_peak,'memory_margin_passed':total_peak<torch.cuda.get_device_properties(0).total_memory*.9,
            'error':error,'losses':losses,**metrics}
        records.append(row);write_json(root/'profile-progress.json',records);print(json.dumps(row),flush=True)
        model.zero_grad(set_to_none=True)
        if error is not None:
            import gc
            gc.collect();torch.cuda.empty_cache()
    good=[r for r in records if r['error'] is None and r['memory_margin_passed']]
    if not good:raise RuntimeError('No safe training microbatch profile')
    selected=max(good,key=lambda r:r['examples_per_second'])
    result={'selected_microbatch':selected['microbatch'],'effective_batch_size':8,'records':records,
        'code_sha256':code_hash(),'image_id':os.environ.get('CADTOKENIZER_IMAGE_ID'),
        'data_sha256':file_hash(root/'data.json'),'settings':s,
        'note':'Two representative worst-length groups, forward/backward without parameter updates. Memory projection includes AdamW states; profiling is not quality evidence.'}
    write_json(root/'profile.json',result)


def freeze(s,root,data):
    from .bench_data import code_hash
    parent=Path(s['parent']);router=read_config('artifacts/lossless-routed-v2/router.json')
    adapter={p.name:file_hash(p) for p in sorted((parent/'adapter').iterdir()) if p.is_file()}
    if adapter!=router['adapter_sha256']['editing']:raise ValueError('Original editing adapter changed')
    profile=read_config(root/'profile.json')
    if profile['code_sha256']!=code_hash() or profile['data_sha256']!=file_hash(root/'data.json') or profile['settings']!=s:
        raise ValueError('GPU profile provenance differs from training protocol')
    signature={'settings':s,'code_sha256':code_hash(),'tests_sha256':file_hash('tests/test_numeric_supervision.py'),
        'analysis_sha256':file_hash('scripts/analyze_edit_correctness.py'),'data_sha256':file_hash(root/'data.json'),
        'profile_sha256':file_hash(root/'profile.json'),'parent_adapter':adapter,
        'router_sha256':file_hash('artifacts/lossless-routed-v2/router.json'),
        'generation_adapter_sha256':router['adapter_sha256']['generation'],
        'image_id':os.environ.get('CADTOKENIZER_IMAGE_ID')}
    manifest={'signature':signature,'microbatch':profile['selected_microbatch'],'train_ids':[c['example_id'] for c in data['train']],
        'validation_ids':[c['example_id'] for c in data['validation']],
        'decision_rule':'Treatment must improve exact Add target matches versus control without reducing valid outputs, increasing paired shared-valid mean Chamfer, or reducing unchanged-command preservation. Also report against the original parent. No automatic promotion.',
        'limits':'One fixed 4x weight, one matched schedule/seed, reused validation panel with repeated components. Numerical batching protocol is new and shared by parent/control/treatment; historical outputs are preserved.'}
    if (root/'manifest.json').exists() and read_config(root/'manifest.json')!=manifest:raise ValueError('Numeric study provenance changed')
    write_json(root/'manifest.json',manifest);return manifest


def run_numeric(action='control',mode='train'):
    require_container()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    if mode=='inspect':
        import unittest
        suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_numeric_supervision')
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise RuntimeError('Numeric mask/loss tests failed')
        return
    s=read_config('configs/lossless-numeric-ablation.json');root=Path(s['output']);root.mkdir(parents=True,exist_ok=True)
    from .lossless_sft import build_model,optimizer_for,teacher_evaluate,load_samples
    from .lossless_sem import LosslessSemCodec
    from .task_experts import save_arm,lr_factor
    from .tokenizer_learning_curve import preserve_rng
    from .sft import InstructionCollator
    with (root/'run.lock').open('a') as lock,sdpa_kernel(SDPBackend.MATH):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def status(phase,**extra):write_json(root/'state.json',{'status':'running','arm':action,'phase':phase,**extra})
        try:
            torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
            torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
            status('prepare');data,raw=prepare_data(s,root)
            if action=='profile':
                if (root/'profile.json').exists():return
                status('gpu_profile');run_profile(s,root,data,raw);return
            manifest=freeze(s,root,data)
            if action=='report':return finish_numeric(s,root,manifest)
            arm_root=root/(action+('-smoke' if mode=='smoke' else ''));arm_root.mkdir(exist_ok=True)
            if (arm_root/'results.json').exists():return read_config(arm_root/'results.json')
            codec=LosslessSemCodec.from_dict(read_config(Path(s['codec_root'])/'codec.json'))
            seed_all(s['seed']);signature={**manifest['signature'],'arm':action,'mode':mode}
            checkpoints=sorted(p for p in (arm_root/'checkpoints').glob('step-*') if (p/'complete.json').exists())
            status('load_model');model,_,_=build_model(s,codec,checkpoints[-1] if checkpoints else Path(s['parent']))
            if action=='parent':
                from .numeric_evaluation import evaluate_numeric
                report=evaluate_numeric(model,data,codec,s,root,arm_root)
                write_json(arm_root/'results.json',{'arm':action,'evaluation':report,'optimizer_updates':0});return
            multiplier=s['numeric_multiplier'] if action=='weighted' else 1.
            train={c['example_id']:attach_weights(x,c['flags'],multiplier) for c,x in zip(data['train'],raw)}
            optimizer,names=optimizer_for(model,s);cursor=updates=tokens=0
            if checkpoints:
                saved=load_checkpoint(checkpoints[-1]/'state.pt')
                if saved['signature']!=signature or saved['parameter_names']!=names:raise ValueError('Numeric arm checkpoint mismatch')
                optimizer.load_state_dict(saved['optimizer']);cursor,updates,tokens=(saved[k] for k in ('cursor','updates','tokens'))
                torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);del saved
            teacher=load_samples(s,data['teacher']);collator=InstructionCollator(128001)
            if not (arm_root/'teacher-before.json').exists():
                with preserve_rng():before=teacher_evaluate(model,teacher,collator)
                expected=read_config('artifacts/lossless-task-experts-v1/editing/results.json')['teacher_before']['loss']
                if abs(before['loss']-expected)>1e-6:raise ValueError('Numeric arm parent reload mismatch')
                write_json(arm_root/'teacher-before.json',before)
            if not checkpoints:save_arm(model,optimizer,arm_root,0,0,0,names,signature)
            endpoint=2 if mode=='smoke' else s['max_updates'];started=time.monotonic();start_updates=updates
            checked=(arm_root/'first_update_gradients.json').exists()
            for index in range(cursor,endpoint):
                chosen=[train[k] for k in data['batches'][index]]
                for group in optimizer.param_groups:group['lr']=group['initial_lr']*lr_factor(index,s['max_updates'],s['warmup_updates'])
                model.train();model.config.use_cache=False;optimizer.zero_grad(set_to_none=True)
                loss=backward_batch(model,chosen,manifest['microbatch'])
                norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
                if not checked:
                    checks={n:bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0)
                        for n,p in model.named_parameters() if p.requires_grad and 'cad_rows.modules_to_save.' in n}
                    if len(checks)!=2 or not all(checks.values()):raise RuntimeError('Missing numeric-supervision row gradients')
                    write_json(arm_root/'first_update_gradients.json',checks);checked=True
                optimizer.step();updates+=1;tokens+=sum(sum(v!=-100 for v in x['labels'][1:]) for x in chosen)
                if updates%10==0 or mode=='smoke':
                    record={'update':updates,'total_updates':endpoint,'loss':loss,'gradient_norm':norm,
                        'supervised_tokens':tokens,'seconds_per_update':(time.monotonic()-started)/(updates-start_updates),
                        'microbatch':manifest['microbatch'],'numeric_multiplier':multiplier}
                    status('training',**record)
                    with (arm_root/'training.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                    print(json.dumps({'arm':action,**record}),flush=True)
                if updates%s['checkpoint_every']==0:save_arm(model,optimizer,arm_root,index+1,updates,tokens,names,signature)
            save_arm(model,optimizer,arm_root,endpoint,updates,tokens,names,signature)
            with preserve_rng():after=teacher_evaluate(model,teacher,collator)
            result={'arm':action,'mode':mode,'optimizer_updates':updates,'example_presentations':updates*8,
                'supervised_tokens':tokens,'numeric_multiplier':multiplier,'teacher_before':read_config(arm_root/'teacher-before.json'),
                'teacher_after':after,'production_promoted':False}
            if mode!='smoke':
                from .numeric_evaluation import evaluate_numeric
                status('evaluation');result['evaluation']=evaluate_numeric(model,data,codec,s,root,arm_root)
            write_json(arm_root/'results.json',result);status('arm_complete')
        except Exception as exc:
            write_json(root/'state.json',{'status':'failed','arm':action,'error':repr(exc)});raise


def finish_numeric(s,root,manifest):
    from .numeric_evaluation import compare_arms
    arms={name:read_config(root/name/'results.json') for name in ('parent','control','weighted')}
    for key in ('optimizer_updates','example_presentations','supervised_tokens'):
        if arms['control'][key]!=arms['weighted'][key]:raise ValueError('Matched budget failed: '+key)
    router=read_config('artifacts/lossless-routed-v2/router.json')
    if file_hash('artifacts/lossless-routed-v2/router.json')!=manifest['signature']['router_sha256']:raise ValueError('Serving router changed')
    for name,digest in manifest['signature']['generation_adapter_sha256'].items():
        if file_hash(Path(router['routes']['generation'])/name)!=digest:raise ValueError('Generation specialist changed')
    comparison=compare_arms(root,arms)
    result={'arms':arms,'comparison':comparison,'decision_rule':manifest['decision_rule'],'budget_verified':True,
        'production_promoted':False,'limits':manifest['limits']}
    write_json(root/'results.json',result)
    (root/'findings.md').write_text('# Changed-numeric supervision comparison\n\n'+json.dumps(result,indent=2)+'\n')
    write_json(root/'state.json',{'status':'complete','results':str(root/'results.json')})
    return result
