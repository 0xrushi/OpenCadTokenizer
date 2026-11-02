"""Assemble the best saved routes and benchmark bounded inference options."""
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

from .common import require_container,read_config,write_json,file_hash,fraction

ROOT=Path('artifacts/lossless-routed-v1')


class GpuMonitor:
    def __enter__(self):
        self.samples=[];self.error=None
        try:
            self.process=subprocess.Popen(['nvidia-smi','--id=0',
                '--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits',
                '--loop-ms=500'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
            def collect():
                for line in self.process.stdout:
                    try:self.samples.append([float(x.strip()) for x in line.split(',')])
                    except ValueError:pass
            self.thread=threading.Thread(target=collect,daemon=True);self.thread.start()
        except OSError as exc:self.error=str(exc);self.process=None
        return self

    def __exit__(self,*args):
        if self.process is not None:
            self.process.terminate();self.process.wait(timeout=5);self.thread.join(timeout=2)

    def report(self):
        return {'samples':len(self.samples),'mean_gpu_utilization_percent':statistics.mean(x[0] for x in self.samples) if self.samples else None,
            'peak_total_gpu_memory_mib':max(x[1] for x in self.samples) if self.samples else None,
            'mean_power_watts':statistics.mean(x[2] for x in self.samples) if self.samples else None,'gpu_monitor_error':self.error}


def run_routed_benchmark(mode='train'):
    require_container()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    if mode=='inspect':
        import unittest
        suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName(name) for name in
            ['tests.test_batch_decode','tests.test_task_experts.ExpertRoutingTests']])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise RuntimeError('Batched routing tests failed')
        return
    from .bench_data import code_hash
    from .task_router import TaskExpertGenerator
    from .lossless_sft import load_samples
    ROOT.mkdir(parents=True,exist_ok=True)
    def status(phase,**extra):write_json(ROOT/'state.json',{'status':'running','phase':phase,**extra})
    with (ROOT/'run.lock').open('a') as lock,sdpa_kernel(SDPBackend.MATH):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            status('freeze')
            s=read_config('configs/lossless-sft-pilot.json')
            pilot=read_config(Path(s['output'])/'manifest.json')
            expected=pilot['signature']
            for name,digest in expected['model_sha256'].items():
                if file_hash(Path(s['model_path'])/name)!=digest:raise ValueError('Cached base changed')
            if file_hash(Path(s['codec_root'])/'codec.json')!=expected['codec_sha256']:raise ValueError('Codec changed')
            if file_hash(Path(s['codec_root'])/'tokens.sqlite')!=expected['cache_sha256']:raise ValueError('Prompt cache changed')
            for name in ('lossless_model.py','lossless_sem.py','lossless_sft.py'):
                path='cadtokenizer/'+name
                if file_hash(path)!=expected['code_sha256'][path]:raise ValueError('Reference inference implementation changed')
            gen_root=Path('artifacts/lossless-generation-validation-v1')
            edit_root=Path('artifacts/lossless-editing-validation-v1')
            generation_results=read_config(gen_root/'results.json')
            if not generation_results['passes_frozen_validation_screen']:raise ValueError('Generation candidate did not pass validation screen')
            gen_manifest=read_config(gen_root/'manifest.json');edit_manifest=read_config(edit_root/'manifest.json')
            routes={'generation':'artifacts/lossless-task-experts-v1/generation/checkpoints/step-000500/adapter',
                    'editing':'artifacts/lossless-sft-pilot-v1/checkpoints/step-000500/adapter'}
            fingerprints={task:{p.name:file_hash(p) for p in sorted(Path(path).iterdir()) if p.is_file()} for task,path in routes.items()}
            for task,key in [('generation','expert'),('editing','parent')]:
                if fingerprints[task]!=gen_manifest['signature']['adapters'][key]:raise ValueError('Selected checkpoint fingerprints changed')
            cases=[]
            for task,pool in [('generation',gen_manifest['cases']),('editing',edit_manifest['cases'])]:
                ordered=sorted(pool,key=lambda c:(fraction(c['example_id'],342),c['example_id']))
                cases.extend({**c,'task':task} for c in ordered[:32])
            frozen={'signature':{'code_sha256':code_hash(),'routing_tests_sha256':file_hash('tests/test_batch_decode.py'),
                'adapter_sha256':fingerprints,'image_id':os.environ.get('CADTOKENIZER_IMAGE_ID'),
                'generation_validation_sha256':file_hash(gen_root/'results.json'),
                'editing_validation_sha256':file_hash(edit_root/'results.json')},
                'cases':cases,'candidate_batch_sizes':[4,8,16,32],'selection_seed':342,
                'precision':'BF16 autocast, deterministic algorithms, math SDPA; no attention-backend changes',
                'selection':'Fastest measured option with exact native-ID and SEM agreement on all 32 prompts for that task, and at least 10% GPU memory margin. Serial reference remains fallback.'}
            if frozen['signature']['image_id']!=expected['image_id']:raise ValueError('Pinned image changed')
            if (ROOT/'manifest.json').exists() and read_config(ROOT/'manifest.json')!=frozen:raise ValueError('Benchmark provenance changed; use a new output')
            write_json(ROOT/'manifest.json',frozen)
            descriptor={'schema':'cad-task-experts-v1','settings':s,'routes':routes,'adapter_sha256':fingerprints,
                'routing':'generation when source is None; original editing adapter otherwise; one shared frozen backbone',
                'inference_options':{'batch_sizes':{'generation':1,'editing':1}},'production_promoted':False,
                'selection_evidence':{'generation':str(gen_root/'results.json'),'editing':'original step-500; editing specialist did not improve shape error'}}
            if (ROOT/'results.json').exists():return read_config(ROOT/'results.json')
            write_json(ROOT/'router.json',descriptor)
            torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
            torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
            status('load_shared_backbone');router=TaskExpertGenerator(ROOT/'router.json')
            gpu_total=torch.cuda.get_device_properties(0).total_memory
            saved={}
            for task,path in [('generation',gen_root/'expert_predictions.jsonl'),('editing',edit_root/'lossless_predictions.jsonl')]:
                saved[task]={r['id']:r for r in [json.loads(line) for line in path.read_text().splitlines()]}
            cached=load_samples(s,cases)
            for c,v in zip(cases,cached):
                if router.prompt(c['instruction'],c['source'])!=v['input_ids'][:v['prompt_length']]:raise ValueError('Combined router prompt differs from cached input')
            reports=[];reference_rows=[]
            for task in ('generation','editing'):
                chosen=[c for c in cases if c['task']==task]
                requests=[{'instruction':c['instruction'],'source':c['source']} for c in chosen]
                # Warm the selected adapter before measuring each task.
                router.predict(**requests[0]);torch.cuda.synchronize()
                status('serial_reference',task=task,completed=0,total=len(chosen))
                torch.cuda.reset_peak_memory_stats();started=time.monotonic();reference=[]
                with GpuMonitor() as monitor:
                    for i,(c,item) in enumerate(zip(chosen,requests),1):
                        row=router.predict(**item);old=saved[task][c['example_id']]
                        if row['sequence']!=old['sequence'] or row['generated_ids']!=old['generated_ids']:
                            raise ValueError('Combined serial route differs from independently evaluated checkpoint: '+c['example_id'])
                        reference.append(row);reference_rows.append({'id':c['example_id'],**row})
                        status('serial_reference',task=task,completed=i,total=len(chosen))
                    torch.cuda.synchronize();elapsed=time.monotonic()-started
                record={'task':task,'batch_size':1,'method':'serial_reference','wall_seconds':elapsed,
                    'shapes_per_second':len(chosen)/elapsed,'generated_tokens_per_second':sum(len(r['generated_ids']) for r in reference)/elapsed,
                    'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'exact_native_matches':len(chosen),'cases':len(chosen),**monitor.report()}
                reports.append(record);write_json(ROOT/'benchmark.json',reports);write_json(ROOT/'reference_outputs.json',reference_rows)
                for size in frozen['candidate_batch_sizes']:
                    status('batched_benchmark',task=task,batch_size=size)
                    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.monotonic();rows=None;error=None
                    with GpuMonitor() as monitor:
                        try:
                            rows=router.predict_many(requests,batch_size=size)
                            torch.cuda.synchronize()
                        except torch.OutOfMemoryError as exc:error=str(exc)
                        elapsed=time.monotonic()-started
                    if error is not None:
                        import gc
                        gc.collect();torch.cuda.empty_cache()
                    matches=sum(a['generated_ids']==b['generated_ids'] and a['sequence']==b['sequence'] for a,b in zip(reference,rows)) if rows else 0
                    record={'task':task,'batch_size':size,'method':'incremental_batched','wall_seconds':elapsed,
                        'shapes_per_second':len(chosen)/elapsed if rows else None,
                        'generated_tokens_per_second':sum(len(r['generated_ids']) for r in rows)/elapsed if rows else None,
                        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'exact_native_matches':matches,'cases':len(chosen),
                        'error':error,'batch_events':list(router.last_batch_stats),
                        'effective_max_batch_size':max((r['batch_size'] for r in router.last_batch_stats if r['event']=='batch'),default=0),**monitor.report()}
                    reports.append(record);write_json(ROOT/'benchmark.json',reports)
                    if rows:write_json(ROOT/f'{task}-batch-{size}.json',[{'id':c['example_id'],**r} for c,r in zip(chosen,rows)])
                    print(json.dumps(record),flush=True)
            selected={};summary={}
            for task in ('generation','editing'):
                measured=[r for r in reports if r['task']==task]
                baseline=measured[0]
                eligible=[r for r in measured if r['exact_native_matches']==r['cases'] and r.get('shapes_per_second')
                    and (r.get('peak_total_gpu_memory_mib') or r['peak_cuda_allocated_bytes']/2**20)*2**20<gpu_total*.9]
                best=max(eligible or [baseline],key=lambda r:r['shapes_per_second'])
                selected[task]=best.get('effective_max_batch_size',best['batch_size'])
                summary[task]={'selected_batch_size':selected[task],'speedup_vs_serial':best['shapes_per_second']/baseline['shapes_per_second'],
                    'serial_shapes_per_second':baseline['shapes_per_second'],'selected_shapes_per_second':best['shapes_per_second'],
                    'exact_reference_matches':best['exact_native_matches'],'cases':best['cases'],
                    'selected_mean_gpu_utilization_percent':best['mean_gpu_utilization_percent'],
                    'selected_peak_total_gpu_memory_mib':best['peak_total_gpu_memory_mib']}
            descriptor['inference_options']={'batch_sizes':selected,'benchmark_sha256':file_hash(ROOT/'benchmark.json'),
                'equivalence_scope':'32 generation and 32 editing validation prompts; exact native IDs and SEM. Not a proof for all inputs.',
                'precision':frozen['precision'],'context_budget':s['context_length'],'oom_policy':'halve batch size and retry; fail if a single prompt cannot fit'}
            write_json(ROOT/'router.json',descriptor)
            result={'routes':routes,'benchmark_summary':summary,'production_promoted':False,
                'limits':'Combined research router only. No new training or quality claims beyond the saved validation results. Batching can change floating-point arithmetic; only exactly matching tested options are selected. Finished historical evaluations are unchanged.'}
            write_json(ROOT/'results.json',result)
            (ROOT/'findings.md').write_text('# Combined CAD router and throughput benchmark\n\n'+json.dumps(result,indent=2)+'\n')
            write_json(ROOT/'state.json',{'status':'complete','results':str(ROOT/'results.json')})
        except Exception as exc:
            write_json(ROOT/'state.json',{'status':'failed','error':repr(exc)});raise


def routed_inference(descriptor,instruction=None,source_path=None,requests_path=None,output='artifacts/routed-output.json',batch_size=None):
    require_container()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from .task_router import TaskExpertGenerator
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    if requests_path:
        if instruction is not None or source_path is not None:raise ValueError('Use requests or a single instruction')
        requests=[json.loads(line) for line in Path(requests_path).read_text().splitlines() if line.strip()]
    elif instruction is not None:
        requests=[{'instruction':instruction,'source':Path(source_path).read_text().strip() if source_path else None}]
    else:raise ValueError('Provide --instruction or --requests')
    with sdpa_kernel(SDPBackend.MATH):
        router=TaskExpertGenerator(descriptor)
        rows=router.predict_many(requests,batch_size=batch_size)
    write_json(output,{'results':rows,'geometry_validity':'not checked; use a CAD kernel to validate solids'})
