"""Matched joint-versus-task-adapter experiment; explicitly routed experts."""
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import sqlite3
import statistics
import time

from .common import read_config,write_json,file_hash,require_container,load_checkpoint,save_checkpoint,seed_all


def make_batches(cases, steps, size, seed):
    pools={task:[c for c in cases if c['task']==task] for task in ('generation','editing')}
    if any(not p or len(p)%size for p in pools.values()):
        raise ValueError('Task pools must be nonempty and divisible by batch size')
    orders={};cursors={};epochs=Counter()
    def reset(task):
        order=list(pools[task])
        random.Random(seed+epochs[task]*1009+(0 if task=='generation' else 1)).shuffle(order)
        orders[task]=order;cursors[task]=0
    for task in pools:reset(task)
    result=[]
    for index in range(steps):
        task='generation' if index%6==0 else 'editing'
        if cursors[task]==len(pools[task]):epochs[task]+=1;reset(task)
        cases=orders[task][cursors[task]:cursors[task]+size];cursors[task]+=size
        if len(cases)!=size:raise ValueError('Partial task batch')
        result.append({'index':index,'task':task,'example_ids':[c['example_id'] for c in cases]})
    return result


def lr_factor(index, steps, warmup):
    return min(1.,(index+1)/warmup)*.5*(1+math.cos(math.pi*index/steps))


def freeze_experiment(s,root):
    from .bench_data import code_hash
    parent=Path(s['parent']);original=read_config(parent.parents[1]/'manifest.json')
    sig=original['signature'];codec_root=Path(s['codec_root'])
    for path,expected in [('data/prepared/cad.sqlite',sig['source_database_sha256']),
                          (codec_root/'tokens.sqlite',sig['cache_sha256']),(codec_root/'codec.json',sig['codec_sha256'])]:
        if file_hash(path)!=expected:raise ValueError('Frozen data or codec changed')
    for name,expected in sig['model_sha256'].items():
        if file_hash(Path(s['model_path'])/name)!=expected:raise ValueError('Base model changed')
    for name in ('lossless_model.py','lossless_sem.py','lossless_sft.py'):
        path='cadtokenizer/'+name
        if file_hash(path)!=sig['code_sha256'][path]:raise ValueError('Model/loss/decoder implementation changed')
    if os.environ.get('CADTOKENIZER_IMAGE_ID')!=sig['image_id']:raise ValueError('Image changed')
    for key in ('seed','codec_root','model_path','train_counts','validation_counts','effective_batch_size',
                'lora_rank','lora_alpha','lora_targets','context_length','max_new_tokens','max_pairs'):
        if s[key]!=sig['settings'][key]:raise ValueError('Unexpected recipe change: '+key)
    batches=make_batches(original['train'],s['max_updates'],s['effective_batch_size'],s['batch_order_seed'])
    signature={'settings':s,'parent_manifest_sha256':file_hash(parent.parents[1]/'manifest.json'),
        'adapter':{p.name:file_hash(p) for p in sorted((parent/'adapter').iterdir()) if p.is_file()},
        'recall_plan_sha256':file_hash(s['recall_plan']),'code_sha256':code_hash(),
        'routing_tests_sha256':file_hash('tests/test_task_experts.py'),
        'image_id':os.environ.get('CADTOKENIZER_IMAGE_ID')}
    manifest={'signature':signature,'train':original['train'],'validation':original['validation'],'batches':batches}
    path=root/'manifest.json'
    if path.exists() and read_config(path)!=manifest:raise ValueError('Experiment provenance changed; choose a fresh output')
    write_json(path,manifest)
    return manifest


def save_arm(model,optimizer,arm_root,cursor,updates,tokens,names,signature):
    path=arm_root/'checkpoints'/f'step-{cursor:06d}';tmp=path.with_name(path.name+'.tmp')
    if path.exists():return
    if tmp.exists():shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp/'adapter',save_embedding_layers=False)
    save_checkpoint(tmp/'state.pt',{'cursor':cursor,'updates':updates,'tokens':tokens,'optimizer':optimizer.state_dict(),
        'rng':__import__('torch').get_rng_state(),'cuda_rng':__import__('torch').cuda.get_rng_state_all(),
        'parameter_names':names,'signature':signature})
    write_json(tmp/'complete.json',{'cursor':cursor,'optimizer_updates':updates,'supervised_tokens':tokens})
    tmp.rename(path)


def run_experts(arm='joint',mode='train'):
    require_container()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    if mode=='inspect':
        import unittest
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tests.test_task_experts'))
        if not result.wasSuccessful():raise RuntimeError('Task-expert tests failed')
        return
    s=read_config('configs/lossless-task-experts.json')
    if mode=='smoke':s.update(output=s['output']+'-smoke',max_updates=2,checkpoint_every=1)
    root=Path(s['output']);root.mkdir(parents=True,exist_ok=True)
    if arm=='report':
        with sdpa_kernel(SDPBackend.MATH):return finish_experiment(s,root)
    from .lossless_sft import build_model,load_samples,optimizer_for,teacher_evaluate,generation_evaluate
    from .lossless_sem import LosslessSemCodec
    from .lossless_continue import recall_evaluate
    from .tokenizer_learning_curve import preserve_rng
    from .sft import InstructionCollator
    arm_root=root/arm;arm_root.mkdir(exist_ok=True)
    def status(phase,**extra):
        state={'status':'running','arm':arm,'phase':phase,**extra}
        write_json(arm_root/'state.json',state);write_json(root/'state.json',state)
    with (root/'run.lock').open('a') as lock,sdpa_kernel(SDPBackend.MATH):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if (arm_root/'results.json').exists():return read_config(arm_root/'results.json')
            status('freeze');manifest=freeze_experiment(s,root)
            signature={**manifest['signature'],'arm':arm}
            (root/'source').mkdir(exist_ok=True)
            for path in Path('cadtokenizer').glob('*.py'):shutil.copy2(path,root/'source'/path.name)
            codec=LosslessSemCodec.from_dict(read_config(Path(s['codec_root'])/'codec.json'))
            train_cases=manifest['train'];all_samples=load_samples(s,train_cases)
            samples={c['example_id']:v for c,v in zip(train_cases,all_samples)}
            validation_cases=[c for c in manifest['validation'] if arm=='joint' or c['task']==arm]
            validation=load_samples(s,validation_cases)
            if not validation_cases:raise ValueError('Empty validation panel')
            torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
            torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
            seed_all(s['seed'])
            checkpoints=sorted(p for p in (arm_root/'checkpoints').glob('step-*') if (p/'complete.json').exists())
            resume=checkpoints[-1] if checkpoints else Path(s['parent'])
            status('load_model');model,_,_=build_model(s,codec,resume)
            optimizer,names=optimizer_for(model,s);cursor=updates=tokens=0
            if checkpoints:
                saved=load_checkpoint(resume/'state.pt')
                if saved['signature']!=signature or saved['parameter_names']!=names:raise ValueError('Arm resume mismatch')
                optimizer.load_state_dict(saved['optimizer']);cursor,updates,tokens=(saved[k] for k in ('cursor','updates','tokens'))
                torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);del saved
            collator=InstructionCollator(128001)
            if not (arm_root/'teacher-before.json').exists():
                status('teacher_before')
                with preserve_rng():before=teacher_evaluate(model,validation,collator)
                if arm=='joint':
                    parent_teacher=read_config(Path(s['parent']).parents[1]/'teacher/step-000500.json')
                    if abs(before['loss']-parent_teacher['loss'])>1e-6:raise ValueError('Joint initialization differs from parent')
                write_json(arm_root/'teacher-before.json',before)
            if not checkpoints:save_arm(model,optimizer,arm_root,0,0,0,names,signature)
            started=time.monotonic();start_updates=updates;torch.cuda.reset_peak_memory_stats()
            gradient_checked=(arm_root/'first_update_row_gradients.json').exists()
            for batch in manifest['batches'][cursor:]:
                index=batch['index'];cursor=index+1
                if arm!='joint' and batch['task']!=arm:continue
                chosen=[samples[key] for key in batch['example_ids']]
                count=sum(sum(v!=-100 for v in x['labels'][1:]) for x in chosen)
                factor=lr_factor(index,s['max_updates'],s['warmup_updates'])
                for group in optimizer.param_groups:group['lr']=group['initial_lr']*factor
                model.train();model.config.use_cache=False;optimizer.zero_grad(set_to_none=True);loss_sum=0.
                for example in chosen:
                    tensors={k:v.cuda() for k,v in collator([example]).items()}
                    active=int(tensors['labels'][:,1:].ne(-100).sum())
                    with torch.autocast('cuda',dtype=torch.bfloat16):output=model(**tensors);loss=output.loss*active/count
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Non-finite expert loss')
                    loss.backward();loss_sum+=float(loss.detach());del output,loss,tensors
                if not gradient_checked:
                    checks={n:bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0)
                        for n,p in model.named_parameters() if p.requires_grad and 'cad_rows.modules_to_save.' in n}
                    if len(checks)!=2 or not all(checks.values()):raise RuntimeError('Missing expert row gradients')
                    write_json(arm_root/'first_update_row_gradients.json',checks);gradient_checked=True
                norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
                optimizer.step();updates+=1;tokens+=count
                if updates%10==0 or mode=='smoke':
                    metrics={'cursor':cursor,'global_batches':s['max_updates'],'optimizer_updates':updates,'task':batch['task'],
                        'loss':loss_sum,'gradient_norm':norm,'supervised_tokens':tokens,
                        'seconds_per_update':(time.monotonic()-started)/(updates-start_updates),
                        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
                    status('training',**metrics)
                    with (arm_root/'training.jsonl').open('a') as stream:stream.write(json.dumps(metrics)+'\n')
                    print(json.dumps({'arm':arm,**metrics}),flush=True)
                if updates%s['checkpoint_every']==0:save_arm(model,optimizer,arm_root,cursor,updates,tokens,names,signature)
            save_arm(model,optimizer,arm_root,s['max_updates'],updates,tokens,names,signature)
            status('teacher_after')
            with preserve_rng():after=teacher_evaluate(model,validation,collator)
            write_json(arm_root/'teacher-after.json',after)
            result={'arm':arm,'global_batch_cursor':s['max_updates'],'optimizer_updates':updates,
                'example_presentations':updates*s['effective_batch_size'],'supervised_tokens':tokens,
                'teacher_before':read_config(arm_root/'teacher-before.json'),'teacher_after':after,
                'production_promoted':False}
            if mode=='train':
                if arm in ('joint','editing'):
                    by_id={c['example_id']:c for c in train_cases};recall=[]
                    with sqlite3.connect('data/prepared/cad.sqlite') as db:
                        for i,key in enumerate(read_config(s['recall_plan'])['sample_ids']):
                            target=db.execute('SELECT g.sequence FROM examples e JOIN geometry g ON g.id=e.target_id WHERE e.id=?',(key,)).fetchone()[0]
                            recall.append({**by_id[key],'target':target,'group':'add_numeric' if i<16 else 'add_no_numeric' if i<32 else 'remove'})
                    recall_samples=load_samples(s,recall);status('train_recall')
                    with preserve_rng():result['train_recall']=recall_evaluate(model,recall,recall_samples,codec,{**s,'output':str(arm_root)},s['max_updates'])
                status('validation_generation')
                with preserve_rng():report=generation_evaluate(model,validation_cases,validation,codec,{**s,'output':str(arm_root)},s['max_updates'])
                report['limits']=f"Frozen original validation subset: {len(validation_cases)} prompts for {arm}. Same decoding/kernel settings. Validation reused across experiments; no test cases or promotion."
                write_json(arm_root/'generation'/f"step-{s['max_updates']:06d}"/'results.json',report)
                result['validation']=report
            write_json(arm_root/'results.json',result)
            write_json(arm_root/'state.json',{'status':'complete','arm':arm,'results':str(arm_root/'results.json')})
            write_json(root/'state.json',{'status':'running' if mode=='train' else 'smoke_complete','phase':'arm_complete','arm':arm})
            return result
        except Exception as exc:
            write_json(root/'state.json',{'status':'failed','arm':arm,'error':repr(exc)})
            write_json(arm_root/'state.json',{'status':'failed','error':repr(exc)});raise


def finish_experiment(s,root):
    from .lossless_sft import load_samples
    from .task_router import TaskExpertGenerator
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with (root/'run.lock').open('a') as lock,sdpa_kernel(SDPBackend.MATH):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'results.json').exists():return read_config(root/'results.json')
        manifest=freeze_experiment(s,root)
        write_json(root/'state.json',{'status':'running','phase':'routing_verification'})
        arms={arm:read_config(root/arm/'results.json') for arm in ('joint','generation','editing')}
        for key in ('optimizer_updates','example_presentations','supervised_tokens'):
            if arms['joint'][key]!=arms['generation'][key]+arms['editing'][key]:raise ValueError('Training budget mismatch: '+key)
        routes={task:str(root/task/'checkpoints'/f"step-{s['max_updates']:06d}"/'adapter') for task in ('generation','editing')}
        descriptor={'schema':'cad-task-experts-v1','settings':s,'routes':routes,
            'adapter_sha256':{task:{p.name:file_hash(p) for p in Path(path).iterdir() if p.is_file()} for task,path in routes.items()},
            'routing':'generation if source is None, editing otherwise; one shared frozen backbone',
            'production_promoted':False}
        write_json(root/'router.json',descriptor)
        torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        router=TaskExpertGenerator(root/'router.json');checks=[]
        for task in ('generation','editing'):
            case=next(c for c in manifest['validation'] if c['task']==task)
            expected=next(r for r in read_config(root/task/'generation'/f"step-{s['max_updates']:06d}"/'per_case.json') if r['id']==case['example_id'])
            cached=load_samples(s,[case])[0]
            if router.prompt(case['instruction'],case['source'])!=cached['input_ids'][:cached['prompt_length']]:raise ValueError('Routed prompt differs from cached prompt')
            try:actual=router.predict(case['instruction'],case['source'])['sequence']
            except ValueError:actual=None
            if actual!=expected['sequence']:raise ValueError('Routed expert differs from standalone output')
            checks.append({'task':task,'id':case['example_id'],'same_prompt':True,'same_output':True})
        comparison={}
        baseline={r['id']:r for r in read_config(root/'joint'/'generation'/f"step-{s['max_updates']:06d}"/'per_case.json')}
        parent={r['id']:r for r in read_config(Path(s['parent']).parents[1]/'generation/step-000500/per_case.json')}
        for task in ('generation','editing'):
            rows=read_config(root/task/'generation'/f"step-{s['max_updates']:06d}"/'per_case.json')
            comparison[task]={}
            for name,control in [('matched_joint',baseline),('original_parent',parent)]:
                pairs=[(control[r['id']],r) for r in rows if control[r['id']]['chamfer'] is not None and r['chamfer'] is not None]
                comparison[task][name]={'shared_valid_cases':len(pairs),
                    'control_mean_chamfer':statistics.mean(a['chamfer'] for a,b in pairs) if pairs else None,
                    'expert_mean_chamfer':statistics.mean(b['chamfer'] for a,b in pairs) if pairs else None,
                    'expert_lower_error_cases':sum(b['chamfer']<a['chamfer'] for a,b in pairs)}
        result={'arms':arms,'paired_comparisons':comparison,'routing_verification':checks,'budget_verified':True,
            'production_promoted':False,'limits':'One matched continuation from SFT-500, shared backbone with two explicitly routed adapters. Equal combined exposure/updates; experts have greater total adapter capacity and separate optimizer histories. Small reused validation panel; not learned MoE or proof of task interference.'}
        write_json(root/'results.json',result)
        (root/'findings.md').write_text('# Task expert comparison\n\n'+json.dumps(result,indent=2)+'\n')
        write_json(root/'state.json',{'status':'complete','results':str(root/'results.json')})
        return result
