"""Read-only existing-model diagnostic, with development-only GPU profiling."""
import fcntl
import gc
import json
import os
from pathlib import Path
import time

from .common import file_hash,read_config,write_json,require_container
from .structured_bore import complete_object,parse,same_intent

ROOT=Path('artifacts/structured-bore-diagnostic-v1')


def run_inference(root=ROOT,text_only=False):
    root=Path(root)
    require_container()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from transformers import AutoTokenizer,StoppingCriteria,StoppingCriteriaList,LogitsProcessor,LogitsProcessorList
    from .lossless_sem import LosslessSemCodec
    from .lossless_sft import build_model
    from .routed_release import GpuMonitor
    protocol=read_config(root/'protocol.json')
    if bool(protocol.get('text_only_vocabulary',False))!=text_only:
        raise ValueError('Decode vocabulary differs from the frozen protocol')
    for p,digest in {**protocol['frozen_code'],**protocol['input_hashes']}.items():
        if file_hash(p)!=digest:raise ValueError('Frozen task/inputs changed')
    router=read_config('artifacts/lossless-routed-v2/router.json');settings=router['settings']
    adapter=Path(router['routes']['editing'])
    for name,digest in router['adapter_sha256']['editing'].items():
        if file_hash(adapter/name)!=digest:raise ValueError('Editing adapter changed')
    original=read_config('artifacts/lossless-sft-pilot-v1/manifest.json')['signature']
    for name,digest in original['model_sha256'].items():
        if file_hash(Path(settings['model_path'])/name)!=digest:raise ValueError('Base model changed')
    if (root/'model_predictions.json').exists():raise ValueError('Predictions already exist; preserve the frozen run')
    def status(phase,**extra):write_json(root/'state.json',{'status':'running','phase':phase,**extra})
    with (root/'inference.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2);torch.set_float32_matmul_precision('high');torch.manual_seed(42)
        torch.cuda.manual_seed_all(42);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        status('load_existing_editing_adapter')
        tokenizer=AutoTokenizer.from_pretrained(Path(settings['codec_root'])/'tokenizer',local_files_only=True)
        tokenizer.padding_side='left'
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        codec=LosslessSemCodec.from_dict(read_config(Path(settings['codec_root'])/'codec.json'))
        model,_,_=build_model(settings,codec,adapter.parent)
        model.requires_grad_(False);model.eval();model.gradient_checkpointing_disable();model.config.use_cache=True
        if hasattr(model,'disable_input_require_grads'):model.disable_input_require_grads()
        main=read_config(root/'main_inputs.json');development=read_config(root/'development_inputs.json')
        for row in main+development:
            row['prompt_ids']=tokenizer.encode(row['prompt'],add_special_tokens=True)
            if len(row['prompt_ids'])+protocol['max_new_tokens']>protocol['context_budget']:
                raise ValueError('No prompt truncation allowed: '+row['id'])
        eos=tokenizer.eos_token_id
        base_vocab_size=read_config(Path(settings['model_path'])/'config.json')['vocab_size']
        class NativeTextOnly(LogitsProcessor):
            def __call__(self,input_ids,scores):
                scores[:,base_vocab_size:]=-float('inf')
                return scores
        processors=LogitsProcessorList([NativeTextOnly()]) if text_only else LogitsProcessorList()
        def equivalent(a,b):
            if protocol.get('profile_comparison')!='strict_intent_or_both_schema_invalid':
                return a['generated_ids']==b['generated_ids']
            try:first=parse(a['text'])
            except ValueError:first=None
            try:second=parse(b['text'])
            except ValueError:second=None
            return first is None and second is None or first is not None and second is not None and same_intent(first,second)
        class StopObject(StoppingCriteria):
            def __init__(self,width):self.width=width
            def __call__(self,input_ids,scores,**kwargs):
                texts=tokenizer.batch_decode(input_ids[:,self.width:].tolist(),skip_special_tokens=False,clean_up_tokenization_spaces=False)
                return torch.tensor([complete_object(t) or (text_only and bool(t.lstrip()) and not t.lstrip().startswith('{')) for t in texts],dtype=torch.bool,device=input_ids.device)
        def generate(chosen):
            width=max(len(r['prompt_ids']) for r in chosen)
            ids=torch.full((len(chosen),width),tokenizer.pad_token_id,dtype=torch.long,device='cuda')
            mask=torch.zeros_like(ids)
            for i,row in enumerate(chosen):
                n=len(row['prompt_ids']);ids[i,-n:]=torch.tensor(row['prompt_ids'],device='cuda');mask[i,-n:]=1
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                output=model.generate(input_ids=ids,attention_mask=mask,do_sample=False,use_cache=True,
                    max_new_tokens=protocol['max_new_tokens'],pad_token_id=tokenizer.pad_token_id,eos_token_id=eos,
                    stopping_criteria=StoppingCriteriaList([StopObject(width)]),logits_processor=processors)
            rows=[]
            for row,tokens in zip(chosen,output[:,width:].tolist()):
                if eos in tokens:tokens=tokens[:tokens.index(eos)]
                rows.append({'id':row['id'],'generated_ids':tokens,
                    'text':tokenizer.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False)})
            return rows
        def measured(pool,size):
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            rows=[];start=time.monotonic();failure=None
            with GpuMonitor() as monitor:
                try:
                    for i in range(0,len(pool),size):rows.extend(generate(pool[i:i+size]))
                    torch.cuda.synchronize()
                except torch.OutOfMemoryError as exc:failure=str(exc)
            elapsed=time.monotonic()-start
            record={'batch_size':size,'requests':len(pool),'wall_seconds':elapsed,'requests_per_second':len(pool)/elapsed if failure is None else None,
                'generated_tokens_per_second':sum(len(r['generated_ids']) for r in rows)/elapsed if failure is None else None,
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'error':failure,**monitor.report()}
            if failure:gc.collect();torch.cuda.empty_cache()
            return rows,record
        status('warmup');generate([max(development,key=lambda r:len(r['prompt_ids']))])
        status('profile_serial_reference',requests=8)
        reference,record=measured(development,1);record['exact_reference_matches']=8
        reports=[record];write_json(root/'profile.json',reports);write_json(root/'development_reference_outputs.json',reference)
        by_id={r['id']:r for r in reference};pool=development*4
        total=torch.cuda.get_device_properties(0).total_memory
        for size in protocol['profile_batch_sizes']:
            status('profile_batch',batch_size=size,requests=len(pool))
            rows,record=measured(pool,size)
            record['exact_token_matches']=sum(r['generated_ids']==by_id[r['id']]['generated_ids'] for r in rows)
            record['exact_reference_matches']=sum(equivalent(r,by_id[r['id']]) for r in rows)
            record['comparison_requests']=len(pool)
            reports.append(record);write_json(root/'profile.json',reports)
            write_json(root/f'development-batch-{size}.json',rows)
        eligible=[r for r in reports if not r['error'] and r['exact_reference_matches']==r['requests']
            and r['peak_total_gpu_memory_mib'] is not None and r['peak_total_gpu_memory_mib']*2**20<total*.9]
        if not eligible:raise RuntimeError('No profile option satisfies memory/equivalence criteria')
        best=max(eligible,key=lambda r:r['requests_per_second']);size=best['batch_size']
        selection={'batch_size':size,'selected_profile':best,'serial_profile':reports[0],
            'selection':protocol['profile_selection'],'peak_prompt_tokens':max(len(r['prompt_ids']) for r in main),
            'development_peak_prompt_tokens':max(len(r['prompt_ids']) for r in development),
            'precision':protocol['generation'],'comparison':protocol.get('profile_comparison','exact_tokens'),'equivalence_scope':'Eight development prompts only; not a guarantee for all main prompts.'}
        write_json(root/'profile_selection.json',selection)
        status('main_model_inference',batch_size=size,completed=0,total=len(main))
        ordered=sorted(main,key=lambda r:(len(r['prompt_ids']),r['id']))
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        predicted=[];start=time.monotonic()
        with GpuMonitor() as monitor:
            for offset in range(0,len(ordered),size):
                predicted.extend(generate(ordered[offset:offset+size]))
                write_json(root/'model_predictions.partial.json',predicted)
                status('main_model_inference',batch_size=size,completed=len(predicted),total=len(main))
            torch.cuda.synchronize()
        elapsed=time.monotonic()-start
        write_json(root/'model_predictions.json',sorted(predicted,key=lambda r:r['id']))
        run={'requests':len(predicted),'batch_size':size,'wall_seconds':elapsed,'requests_per_second':len(predicted)/elapsed,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),**monitor.report(),
            'adapter':str(adapter),'adapter_sha256':router['adapter_sha256']['editing'],
            'prompt_format_changed':True,'training_started':False,'model_weights_modified':False,
            'text_only_vocabulary':text_only,'base_vocab_size':base_vocab_size,
            'inference_code_sha256':file_hash(__file__),'protocol_sha256':file_hash(root/'protocol.json')}
        write_json(root/'model_run.json',run);status('model_predictions_complete',requests=len(predicted))
        print(json.dumps({'profile_selection':selection,'model_run':run},indent=2),flush=True)
