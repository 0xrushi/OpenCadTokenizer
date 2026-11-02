"""Recheck the fastest exact editing batch without stale allocator reservations."""
import fcntl
import json
import os
from pathlib import Path
import time

from .common import require_container,read_config,write_json,file_hash


def run_memory_check():
    require_container()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from .task_router import TaskExpertGenerator
    from .routed_release import GpuMonitor
    from .bench_data import code_hash
    root=Path('artifacts/lossless-routed-v2');root.mkdir(parents=True,exist_ok=True)
    old=Path('artifacts/lossless-routed-v1')
    with (root/'run.lock').open('a') as lock,sdpa_kernel(SDPBackend.MATH):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if (root/'results.json').exists():return read_config(root/'results.json')
            write_json(root/'state.json',{'status':'running','phase':'load_shared_backbone'})
            prior=read_config(old/'results.json');records=read_config(old/'benchmark.json')
            eligible=[r for r in records if r['task']=='editing' and r['exact_native_matches']==r['cases'] and r.get('shapes_per_second')]
            best=max(eligible,key=lambda r:r['shapes_per_second'])
            size=best.get('effective_max_batch_size',best['batch_size'])
            if size<=1:raise ValueError('No faster exact editing batch was established')
            cases=[c for c in read_config(old/'manifest.json')['cases'] if c['task']=='editing']
            expected={r['id']:r for r in read_config(old/'reference_outputs.json') if r['task']=='editing'}
            requests=[{'instruction':c['instruction'],'source':c['source']} for c in cases]
            manifest={'prior_manifest_sha256':file_hash(old/'manifest.json'),'prior_benchmark_sha256':file_hash(old/'benchmark.json'),
                'prior_reference_sha256':file_hash(old/'reference_outputs.json'),'code_sha256':code_hash(),
                'image_id':os.environ.get('CADTOKENIZER_IMAGE_ID'),'task':'editing','batch_size':size,
                'case_ids':[c['example_id'] for c in cases],
                'protocol':'Fresh model process, release unused CUDA allocator cache before timing. Retain exact native-ID/SEM and 10% total-device-memory margin requirements.'}
            if (root/'manifest.json').exists() and read_config(root/'manifest.json')!=manifest:raise ValueError('Memory-check provenance changed')
            write_json(root/'manifest.json',manifest)
            torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
            torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
            router=TaskExpertGenerator(old/'router.json')
            router.model.set_adapter('default');router.model.requires_grad_(False);router.model.eval()
            torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            write_json(root/'state.json',{'status':'running','phase':'editing_batch_recheck','batch_size':size})
            with GpuMonitor() as monitor:
                started=time.monotonic()
                rows=router.predict_many(requests,batch_size=size)
                torch.cuda.synchronize();elapsed=time.monotonic()-started
            matches=sum(r['generated_ids']==expected[c['example_id']]['generated_ids'] and r['sequence']==expected[c['example_id']]['sequence'] for c,r in zip(cases,rows))
            measured={'cases':len(cases),'exact_reference_matches':matches,'requested_batch_size':size,
                'effective_batch_size':max(r['batch_size'] for r in router.last_batch_stats if r['event']=='batch'),
                'wall_seconds':elapsed,'shapes_per_second':len(cases)/elapsed,
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),**monitor.report()}
            write_json(root/'memory_recheck.json',measured)
            write_json(root/'outputs.json',[{'id':c['example_id'],**r} for c,r in zip(cases,rows)])
            if matches!=len(cases):raise ValueError('Editing output equivalence failed')
            if measured['peak_total_gpu_memory_mib'] is None:raise ValueError('Total GPU memory measurement unavailable')
            if measured['peak_total_gpu_memory_mib']*2**20>=torch.cuda.get_device_properties(0).total_memory*.9:
                raise ValueError('Editing batch lacks required GPU memory margin')
            descriptor=read_config(old/'router.json')
            descriptor['inference_options']['batch_sizes']['editing']=measured['effective_batch_size']
            descriptor['inference_options']['memory_recheck_sha256']=file_hash(root/'memory_recheck.json')
            descriptor['inference_options']['equivalence_scope']='32 editing prompts at the selected batch; 32 generation prompts serial. Finite-sample validation, not a guarantee for all inputs.'
            write_json(root/'router.json',descriptor)
            baseline=next(r for r in records if r['task']=='editing' and r['method']=='serial_reference')
            result={**prior,'benchmark_summary':{**prior['benchmark_summary'],'editing':{
                'selected_batch_size':measured['effective_batch_size'],'speedup_vs_serial':measured['shapes_per_second']/baseline['shapes_per_second'],
                'serial_shapes_per_second':baseline['shapes_per_second'],'selected_shapes_per_second':measured['shapes_per_second'],
                'exact_reference_matches':matches,'cases':len(cases),
                'selected_mean_gpu_utilization_percent':measured['mean_gpu_utilization_percent'],
                'selected_peak_total_gpu_memory_mib':measured['peak_total_gpu_memory_mib']}},
                'memory_recheck':measured,
                'limits':'V1 evidence preserved. Idle CUDA reservations had contaminated the original total-memory screening. Fresh-process recheck validates editing throughput/headroom. Generation stays serial because batched outputs differed. No new training, quality evaluation or production promotion.'}
            write_json(root/'results.json',result)
            (root/'findings.md').write_text('# Combined router: corrected memory screening\n\n'+json.dumps(result,indent=2)+'\n')
            write_json(root/'state.json',{'status':'complete','results':str(root/'results.json')})
            print(json.dumps(result['benchmark_summary'],indent=2),flush=True)
        except Exception as exc:
            write_json(root/'state.json',{'status':'failed','error':repr(exc)});raise
