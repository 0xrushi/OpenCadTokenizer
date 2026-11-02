"""Small, local, reference-reward GRPO pilot for structured CAD editing."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import statistics
import time

from .common import read_config,write_json,file_hash,require_container,load_checkpoint,seed_all
from .grpo_reward import reward,advantages


def run_grpo(mode='train'):
    require_container()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from .grpo_policy import sample,token_logps,clipped_loss,self_test
    if mode=='inspect':
        self_test();return
    from .lossless_sft import build_model,load_samples,optimizer_for,save_pilot,teacher_evaluate,generation_evaluate
    from .lossless_sem import LosslessSemCodec
    from .lossless_editing_eval import read_journal
    from .lossless_continue import recall_evaluate
    from .tokenizer_learning_curve import preserve_rng
    from .bench_runtime import KernelRunner
    from .bench_data import code_hash
    from .sft import InstructionCollator
    s=read_config('configs/lossless-grpo.json')
    if mode=='smoke':
        s.update(output=s['output']+'-smoke',groups=2,reward_audit_prompts=2,
                 minimum_informative_audit_groups=0,checkpoint_every=1)
    root=Path(s['output']);root.mkdir(parents=True,exist_ok=True)
    def status(phase,**extra):write_json(root/'state.json',{'status':'running','phase':phase,**extra})
    with (root/'run.lock').open('a') as lock,sdpa_kernel(SDPBackend.MATH):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if (root/'results.json').exists():return read_config(root/'results.json')
            status('freeze')
            torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
            torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
            parent=Path(s['parent']);pilot=read_config(parent.parents[1]/'manifest.json')
            original=pilot['signature'];codec_root=Path(s['codec_root'])
            if file_hash('data/prepared/cad.sqlite')!=original['source_database_sha256'] or file_hash(codec_root/'tokens.sqlite')!=original['cache_sha256'] or file_hash(codec_root/'codec.json')!=original['codec_sha256']:
                raise ValueError('Pilot data/cache/codec changed')
            for name,digest in original['model_sha256'].items():
                if file_hash(Path(s['model_path'])/name)!=digest:raise ValueError('Base weights changed')
            for name in ('lossless_model.py','lossless_sem.py','lossless_sft.py'):
                path='cadtokenizer/'+name
                if file_hash(path)!=original['code_sha256'][path]:raise ValueError('Original model/decoder changed')
            if os.environ.get('CADTOKENIZER_IMAGE_ID')!=original['image_id']:raise ValueError('Pinned image changed')
            plan=read_config(s['recall_plan']);selected={c['example_id']:c for c in pilot['train']}
            recall=[]
            with sqlite3.connect('data/prepared/cad.sqlite') as db:
                for i,key in enumerate(plan['sample_ids']):
                    instruction,source,target,split=db.execute('SELECT instruction,source_id,target_id,split FROM examples WHERE id=?',(key,)).fetchone()
                    if split!='train' or key not in selected:raise ValueError('GRPO cases must be pilot training data')
                    seq=[]
                    for endpoint in (source,target):
                        value,stored=db.execute('SELECT sequence,split FROM geometry WHERE id=?',(endpoint,)).fetchone()
                        if stored!='train':raise ValueError('Endpoint split mismatch')
                        seq.append(value)
                    recall.append({**selected[key],'instruction':instruction,'source':seq[0],'target':seq[1],
                        'group':'add_numeric' if i<16 else 'add_no_numeric' if i<32 else 'remove'})
            if len(recall)!=64 or len({c['example_id'] for c in recall})!=64:raise ValueError('Recall panel mismatch')
            signature={'settings':s,'parent_manifest':file_hash(parent.parents[1]/'manifest.json'),
                'adapter':{p.name:file_hash(p) for p in sorted((parent/'adapter').iterdir()) if p.is_file()},
                'recall_plan':file_hash(s['recall_plan']),'code_sha256':code_hash(),
                'reward_code_sha256':file_hash('cadtokenizer/grpo_reward.py'),
                'image_id':os.environ.get('CADTOKENIZER_IMAGE_ID')}
            frozen={'signature':signature,'candidate_training_cases':recall,'validation':pilot['validation']}
            if (root/'manifest.json').exists() and read_config(root/'manifest.json')!=frozen:raise ValueError('GRPO provenance changed; choose new output')
            write_json(root/'manifest.json',frozen)
            geometry_settings={**read_config('configs/benchmark.json'),'output':str(root/'reward-cache'),
                'render':False,'point_count':64,'seed':42}
            runner=KernelRunner(geometry_settings)
            def geometries(jobs):
                unique=list(dict.fromkeys(jobs))
                with ThreadPoolExecutor(max_workers=4) as pool:
                    return dict(zip(unique,pool.map(lambda j:runner.reconstruct(j[2],j[0],j[1]),unique)))
            status('reference_geometry_audit')
            jobs=[(c['example_id'],role,c[name]) for c in recall for role,name in [('reference','target'),('prediction','source')]]
            geom=geometries(jobs)
            codec=LosslessSemCodec.from_dict(read_config(codec_root/'codec.json'))
            candidates=[c for c in recall if geom[(c['example_id'],'reference',c['target'])]['status']=='valid'
                and geom[(c['example_id'],'prediction',c['source'])]['status']=='valid'
                and len(codec.encode(c['target']))-1<=s['max_new_tokens']]
            train=[]
            for group,count in s['train_groups'].items():
                pool=[c for c in candidates if c['group']==group]
                if len(pool)<count:raise ValueError('Insufficient valid bounded-length cases: '+group)
                train.extend(pool[:count])
            write_json(root/'training_panel.json',train)
            # Control mutations change a target sketch-offset bin by one. They
            # are reward probes only and never enter the optimizer as examples.
            controls=[];control_jobs=[]
            for c in train:
                words=c['target'].split();index=next(i for i,w in enumerate(words) if w.startswith(('add,','cut,','intersect,')))
                parts=words[index].split(',');number=int(parts[-1]);parts[-1]=str(number+1 if number<63 else number-1)
                words[index]=','.join(parts);wrong=' '.join(words)
                controls.append((c,wrong));control_jobs.append((c['example_id'],'prediction',wrong))
            wrong_geometry=geometries(control_jobs);checks=[]
            for c,wrong in controls:
                rt=reward(c['source'],c['target'],c['target'],True)
                rc=reward(c['source'],c['target'],c['source'],True)
                rw=reward(c['source'],c['target'],wrong,wrong_geometry[(c['example_id'],'prediction',wrong)]['status']=='valid')
                checks.append({'id':c['example_id'],'target':rt,'copy':rc,'wrong_numeric_edit':rw})
                if not(rt>rc and rt>rw and rc==0):raise ValueError('Reward-control ranking failed')
            write_json(root/'reward_controls.json',checks)
            status('load_model')
            checkpoints=sorted(p for p in (root/'checkpoints').glob('step-*') if (p/'complete.json').exists())
            model,_,_=build_model(s,codec,checkpoints[-1] if checkpoints else parent)
            if any(m.p for m in model.modules() if isinstance(m,torch.nn.Dropout)) or getattr(model.config,'attention_dropout',0):
                raise ValueError('GRPO likelihood recomputation requires dropout-free policy')
            optimizer,names=optimizer_for(model,s)
            seed_all(s['seed']);start=updates=0
            if checkpoints:
                saved=load_checkpoint(checkpoints[-1]/'state.pt')
                if saved['signature']!=signature or saved['parameter_names']!=names:raise ValueError('GRPO checkpoint signature mismatch')
                optimizer.load_state_dict(saved['optimizer']);start=saved['step'];updates=saved['epoch']
                torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);del saved
            samples=load_samples(s,train);recall_samples=load_samples(s,recall)
            validation=load_samples(s,pilot['validation']);collator=InstructionCollator(128001)
            def group_rollout(case,prompt):
                rows=[sample(model,prompt,codec,s) for _ in range(s['num_generations'])]
                jobs=[(case['example_id'],'prediction',r['sequence']) for r in rows]
                values=geometries(jobs)
                for row,job in zip(rows,jobs):
                    row['geometry']=values[job]
                    row['reward']=reward(case['source'],case['target'],row['sequence'],row['geometry']['status']=='valid')
                return rows
            if not checkpoints:
                status('reward_rollout_audit',completed=0,total=s['reward_audit_prompts'])
                audit=[]
                for i in range(s['reward_audit_prompts']):
                    case,example=train[i],samples[i]
                    rows=group_rollout(case,example['input_ids'][:example['prompt_length']])
                    _,std=advantages([r['reward'] for r in rows])
                    audit.append({'id':case['example_id'],'reward_std':std,'candidates':rows})
                    write_json(root/'reward_rollout_audit.json',audit)
                    status('reward_rollout_audit',completed=i+1,total=s['reward_audit_prompts'])
                informative=sum(r['reward_std']>1e-6 for r in audit)
                write_json(root/'reward_audit_result.json',{'control_rankings_passed':len(checks),
                    'sampled_groups':len(audit),'informative_groups':informative,'minimum_required':s['minimum_informative_audit_groups']})
                if informative<s['minimum_informative_audit_groups']:raise ValueError('Insufficient reward variation for GRPO pilot')
                with preserve_rng():teacher=teacher_evaluate(model,validation,collator)
                baseline=read_config(parent.parents[1]/'teacher/step-000500.json')
                if abs(teacher['loss']-baseline['loss'])>1e-6:raise ValueError('GRPO parent reload differs from SFT baseline')
                write_json(root/'teacher-before.json',teacher)
                save_pilot(model,optimizer,s,{'signature':signature},0,0,0,names)
            started=time.monotonic();gradient_checked=(root/'first_update_gradients.json').exists()
            for step in range(start,s['groups']):
                # Alternate Add/Remove rather than putting a whole verb first.
                index=(step%2)*16+(step//2)%16
                case,example=train[index],samples[index]
                prompt=example['input_ids'][:example['prompt_length']]
                status('rollout',group=step+1,total=s['groups'],optimizer_updates=updates)
                rows=group_rollout(case,prompt)
                adv,std=advantages([r['reward'] for r in rows])
                write_json(root/'rollouts'/f'group-{step+1:06d}.json',{'id':case['example_id'],'candidates':rows,'advantages':adv})
                loss_value=0.;grad_norm=0.;max_logp_difference=0.
                if std>1e-6:
                    model.train();model.config.use_cache=False
                    with torch.no_grad():old=[token_logps(model,prompt,r['generated_ids'],codec,s).detach().cpu() for r in rows]
                    optimizer.zero_grad(set_to_none=True)
                    for row,old_lp,a in zip(rows,old,adv):
                        lp=token_logps(model,prompt,row['generated_ids'],codec,s)
                        difference=float((lp.detach().cpu()-old_lp).abs().max())
                        max_logp_difference=max(max_logp_difference,difference)
                        if difference>1e-4:raise ValueError('Old and new same-policy log likelihoods disagree')
                        loss=clipped_loss(lp,old_lp.to(lp.device),a,s['clip_epsilon'])/len(rows)
                        if not bool(torch.isfinite(loss)):raise FloatingPointError('Non-finite GRPO loss')
                        loss.backward();loss_value+=float(loss.detach());del lp,loss
                    grad_norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
                    if not gradient_checked:
                        checks={n:bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0)
                            for n,p in model.named_parameters() if p.requires_grad and 'cad_rows.modules_to_save.' in n}
                        if len(checks)!=2 or not all(checks.values()) or grad_norm<=0:raise RuntimeError('GRPO update has no usable row/policy gradients')
                        write_json(root/'first_update_gradients.json',checks);gradient_checked=True
                    optimizer.step();updates+=1
                metrics={'group':step+1,'total':s['groups'],'optimizer_updates':updates,'reward_mean':statistics.mean(r['reward'] for r in rows),
                    'reward_std':std,'valid_candidates':sum(r['geometry']['status']=='valid' for r in rows),
                    'exact_candidates':sum(r['sequence']==case['target'] for r in rows),
                    'gradient_norm':grad_norm,'loss':loss_value,'max_same_policy_logp_difference':max_logp_difference,
                    'seconds_per_group':(time.monotonic()-started)/(step-start+1)}
                with (root/'training.jsonl').open('a') as stream:stream.write(json.dumps(metrics)+'\n')
                status('training',**metrics);print(json.dumps(metrics),flush=True)
                if (step+1)%s['checkpoint_every']==0 or step+1==s['groups']:
                    save_pilot(model,optimizer,s,{'signature':signature},step+1,updates,0,names)
            if not gradient_checked:raise RuntimeError('No informative policy update occurred')
            status('final_teacher')
            with preserve_rng():teacher=teacher_evaluate(model,validation,collator)
            write_json(root/'teacher-after.json',teacher)
            result={'groups':s['groups'],'optimizer_updates':updates,'mode':mode,
                'teacher_before':read_config(root/'teacher-before.json'),'teacher_after':teacher,
                'reward_audit':read_config(root/'reward_audit_result.json'),'production_promoted':False,
                'limits':'Small reference-reward GRPO pilot, 32 existing training prompts, beta=0 and one update per sampled group. Rewards use target programs, not independent instruction judgments. No held-out test data or automatic promotion.'}
            if mode=='train':
                status('final_train_recall')
                with preserve_rng():result['train_recall']=recall_evaluate(model,recall,recall_samples,codec,s,s['groups'])
                status('final_validation')
                with preserve_rng():result['validation']=generation_evaluate(model,pilot['validation'],validation,codec,s,s['groups'])
                previous=read_config(parent.parents[1]/'generation/step-000500/per_case.json')
                current=read_config(root/'generation'/f"step-{s['groups']:06d}"/'per_case.json')
                if [r['id'] for r in previous]!=[r['id'] for r in current]:raise ValueError('Validation IDs changed')
                pairs=[(a,b) for a,b in zip(previous,current) if a['task']=='editing' and a['chamfer'] is not None and b['chamfer'] is not None]
                result['paired_editing_vs_parent']={'cases':len(pairs),'parent_mean_chamfer':statistics.mean(a['chamfer'] for a,b in pairs) if pairs else None,
                    'grpo_mean_chamfer':statistics.mean(b['chamfer'] for a,b in pairs) if pairs else None}
            write_json(root/'results.json',result)
            (root/'findings.md').write_text('# Small CAD GRPO pilot\n\n'+json.dumps(result,indent=2)+'\n')
            write_json(root/'state.json',{'status':'complete','results':str(root/'results.json')})
        except Exception as exc:
            write_json(root/'state.json',{'status':'failed','error':repr(exc)});raise
