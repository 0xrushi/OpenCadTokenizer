"""GPU profiling/encoding and independent packet-only decoding for the codec test."""
import fcntl
import gc
import json
import os
from pathlib import Path
import time

from .common import require_container,read_config,write_json,file_hash,fraction
from .representation import analyze,split_pairs,tokenize,stringify
from .scale_origin_codec import VQ,HYBRID,LOSSLESS,pack,unpack,metadata_for_pair,decode,lossless_packet

CONFIG='configs/tokenizer-scale-origin.json'


def numerical_context():
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    torch.set_num_threads(2);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    return sdpa_kernel(SDPBackend.MATH)


def profile_model(model,pairs,flags,settings,root,label):
    import torch
    from .scale_origin_runtime import encode_pairs,decode_pairs
    from .routed_release import GpuMonitor
    warm=encode_pairs(model,pairs[:1],1);decode_pairs(model,warm,flags[:1],1)
    reports=[];reference=None
    for size in [1,*settings['profile_batch_sizes']]:
        pool=pairs if size==1 else [pairs[i%len(pairs)] for i in range(settings['profile_requests'])]
        first=flags if size==1 else [flags[i%len(flags)] for i in range(len(pool))]
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        start=time.monotonic();error=None;codes=decoded=None
        with GpuMonitor() as monitor:
            try:
                codes=encode_pairs(model,pool,size);decoded=decode_pairs(model,codes,first,size);torch.cuda.synchronize()
            except torch.OutOfMemoryError as exc:error=str(exc)
        elapsed=time.monotonic()-start
        if size==1:
            if error:raise RuntimeError('Serial profile failed: '+error)
            reference=list(zip(codes,decoded))
        matches=sum(c==reference[i%len(reference)][0] and d==reference[i%len(reference)][1] for i,(c,d) in enumerate(zip(codes or [],decoded or [])))
        record={'arm':label,'batch_size':size,'pairs':len(pool),'seconds':elapsed,
            'pairs_per_second':len(pool)/elapsed if not error else None,'exact_code_and_decoder_matches':matches,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'error':error,**monitor.report()}
        reports.append(record);write_json(root/f'{label}-profile.json',reports)
        write_json(root/'state.json',{'status':'running','phase':'gpu_profile','arm':label,'completed_batch':size})
        del codes,decoded;gc.collect();torch.cuda.empty_cache()
    eligible=[r for r in reports if not r['error'] and r['exact_code_and_decoder_matches']==r['pairs']
        and r['peak_total_gpu_memory_mib'] is not None and r['peak_total_gpu_memory_mib']*2**20<torch.cuda.get_device_properties(0).total_memory*settings['memory_fraction']]
    if not eligible:raise RuntimeError('No profiled option meets equivalence/headroom criteria')
    return max(eligible,key=lambda r:r['pairs_per_second'])


def encode_study():
    require_container()
    import torch
    from .vq import load_vq
    from .scale_origin_runtime import encode_pairs
    settings=read_config(CONFIG);root=Path(settings['output']);root.mkdir(exist_ok=True,parents=True)
    if (root/'encoded_manifest.json').exists():raise ValueError('Encoded streams already exist; preserve this version')
    source=read_config(settings['validation_manifest']);cases=source['validation']
    if len(cases)!=512:raise ValueError('Expected the original 512-shape panel')
    files=[CONFIG,settings['validation_manifest'],settings['baseline_checkpoint'],settings['candidate_checkpoint'],
        'cadtokenizer/scale_origin_codec.py','cadtokenizer/scale_origin_runtime.py',__file__,
        'cadtokenizer/scale_origin_evaluation.py','tests/test_scale_origin_codec.py','cadtokenizer/vq.py','cadtokenizer/representation.py']
    frozen={'settings':settings,'cases':512,'ids':[c['id'] for c in cases],
        'hashes':{str(p):file_hash(p) for p in files},'image_id':os.environ.get('CADTOKENIZER_IMAGE_ID'),
        'selection':'Original full 512 reconstruction validation cases. Profile source pairs selected from training only. No test data.',
        'hypothesis':'Encode three exact discrete scale/origin bins in every pair trailer. Learned decoder otherwise unchanged. Decoder reads packets/checkpoint only.',
        'screen':'More valid reference reconstructions than existing tokenizer and no higher shared-valid mean squared normalized Chamfer. Report token/bit/byte overhead; not a matched-bit-budget claim or automatic promotion.'}
    if (root/'protocol.json').exists() and read_config(root/'protocol.json')!=frozen:raise ValueError('Frozen experiment changed')
    write_json(root/'protocol.json',frozen)
    with (root/'run.lock').open('a') as lock,numerical_context():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        entries=[];profiles={};packet_info=[]
        for label,path in [('existing',settings['baseline_checkpoint']),('curve',settings['candidate_checkpoint'])]:
            write_json(root/'state.json',{'status':'running','phase':'load_checkpoint','arm':label})
            model,_=load_vq(path,torch.device('cuda'));model.requires_grad_(False);data=model.config['data'];size=model.config['vq']['codebook_size']
            # A fixed source-only training pool, diverse in pair token length.
            training=sorted(source['train'],key=lambda c:(fraction(c['id'],settings['profile_seed']),c['id']))[:64]
            available=[]
            for c in training:
                _,parts=analyze(c['sequence'],data['max_pair_tokens'],data['max_primitives'],data['extrusion_partition'],data.get('sketch_partition','curve'))
                available.extend((c['id'],i,p) for i,p in enumerate(parts))
            available.sort(key=lambda item:(len(item[2]['tokens']),item[0],item[1]))
            chosen=[available[round(i*(len(available)-1)/(settings['profile_unique_pairs']-1))] for i in range(settings['profile_unique_pairs'])]
            write_json(root/f'{label}-profile-inputs.json',[{'training_id':c,'ordinal':i,'tokens':p['tokens'],'groups':p['groups']} for c,i,p in chosen])
            best=profile_model(model,[p for _,_,p in chosen],[i==0 for _,i,_ in chosen],settings,root,label)
            profiles[label]=best;write_json(root/'profile_selection.json',profiles)
            flat=[];bounds=[]
            for case in cases:
                _,parts=analyze(case['sequence'],data['max_pair_tokens'],data['max_primitives'],data['extrusion_partition'],data.get('sketch_partition','curve'))
                bounds.append((len(flat),len(parts)));flat.extend(parts)
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
            codes=encode_pairs(model,flat,best['batch_size']);torch.cuda.synchronize()
            write_json(root/f'{label}-encoding.json',{'pairs':len(flat),'seconds':time.monotonic()-start,
                'batch_size':best['batch_size'],'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()})
            streams=[]
            for case,(offset,count) in zip(cases,bounds):
                per_pair=codes[offset:offset+count];metadata=[metadata_for_pair(p['tokens']) for p in flat[offset:offset+count]]
                for arm,kind in ([(label,VQ)] if label=='existing' else [('curve',VQ),('hybrid',HYBRID)]):
                    packet=pack(kind,size,per_pair,metadata if kind==HYBRID else None)
                    directory=root/'packets'/arm;directory.mkdir(exist_ok=True,parents=True);destination=directory/(case['id']+'.ctp');destination.write_bytes(packet)
                    streams.append({'id':case['id'],'arm':arm,'packet':str(destination)})
                    info=unpack(packet);packet_info.append({'id':case['id'],'arm':arm,
                        **{k:info[k] for k in ['token_count','vocabulary_size','bits_per_token','payload_bits','serialized_bytes','framing_tokens','metadata_tokens','fixed_header_crc_bytes']},
                        'code_tokens':sum(map(len,per_pair)),'pairs':count})
            entries.append({'arm':label,'checkpoint':path,'checkpoint_sha256':file_hash(path),'codebook_size':size,
                'batch_size':best['batch_size'],'streams':streams})
            del model,flat,codes;gc.collect();torch.cuda.empty_cache()
        lossless=[]
        for case in cases:
            packet=lossless_packet(case['sequence']);directory=root/'packets'/'lossless';directory.mkdir(exist_ok=True,parents=True)
            destination=directory/(case['id']+'.ctp');destination.write_bytes(packet);info=unpack(packet)
            lossless.append({'id':case['id'],'arm':'lossless','packet':str(destination)})
            packet_info.append({'id':case['id'],'arm':'lossless',**{k:info[k] for k in ['token_count','vocabulary_size','bits_per_token','payload_bits','serialized_bytes','framing_tokens','metadata_tokens','fixed_header_crc_bytes']},'code_tokens':sum(map(len,info['pairs'])),'pairs':len(info['pairs'])})
        entries.append({'arm':'lossless','streams':lossless})
        write_json(root/'packet_costs.json',packet_info)
        inputs={'settings':settings,'variants':entries,'decoder_contract':'Only packet files, checkpoint weights and frozen numerical/batching settings. No source SEM, validation references, saved reconstructions or encoder memory.'}
        write_json(root/'decode_inputs.json',inputs)
        write_json(root/'encoded_manifest.json',{'protocol_sha256':file_hash(root/'protocol.json'),
            'packet_hashes':{str(p):file_hash(p) for p in sorted((root/'packets').rglob('*.ctp'))},
            'decode_inputs_sha256':file_hash(root/'decode_inputs.json')})
        write_json(root/'state.json',{'status':'encoded','packets':2048,'next':'Independent packet-only decoder process'})
        print(json.dumps({'status':'encoded','packets':2048,'profiles':profiles},indent=2),flush=True)


def decode_study():
    require_container()
    import torch
    from .vq import load_vq
    from .scale_origin_runtime import decode_pairs
    from .routed_release import GpuMonitor
    settings=read_config(CONFIG);root=Path(settings['output']);manifest=read_config(root/'encoded_manifest.json')
    if (root/'decoded.json').exists():raise ValueError('Decoded outputs already exist; preserve this version')
    for path,digest in manifest['packet_hashes'].items():
        if file_hash(path)!=digest:raise ValueError('Encoded packet changed')
    if file_hash(root/'decode_inputs.json')!=manifest['decode_inputs_sha256']:raise ValueError('Decode inputs changed')
    inputs=read_config(root/'decode_inputs.json');rows=[];timings=[]
    with numerical_context():
        for variant in inputs['variants']:
            start=time.monotonic();label=variant['arm']
            write_json(root/'state.json',{'status':'running','phase':'packet_only_decode','arm':label})
            if label=='lossless':
                for item in variant['streams']:rows.append({**item,'sequence':decode(Path(item['packet']).read_bytes()),'error':None})
                timings.append({'arm':label,'seconds':time.monotonic()-start});continue
            if file_hash(variant['checkpoint'])!=variant['checkpoint_sha256']:raise ValueError('Checkpoint changed')
            model,_=load_vq(variant['checkpoint'],torch.device('cuda'));model.requires_grad_(False)
            # Memoization is keyed only by code IDs and first-pair grammar flag,
            # within one checkpoint. Hybrid never needs a second neural decode.
            keys={}
            for item in variant['streams']:
                parsed=unpack(Path(item['packet']).read_bytes())
                if parsed['codebook_size']!=variant['codebook_size']:raise ValueError('Codebook mismatch')
                for i,codes in enumerate(parsed['pairs']):keys.setdefault((tuple(codes),i==0),len(keys))
            ordered=list(keys);torch.cuda.reset_peak_memory_stats()
            with GpuMonitor() as monitor:
                predicted=decode_pairs(model,[list(k[0]) for k in ordered],[k[1] for k in ordered],variant['batch_size']);torch.cuda.synchronize()
            def callback(codes,first):
                result=predicted[keys[(tuple(codes),first)]]
                if result['error']:raise ValueError(result['error'])
                return result['tokens']
            for item in variant['streams']:
                try:row={**item,'sequence':decode(Path(item['packet']).read_bytes(),callback,variant['codebook_size']),'error':None}
                except ValueError as exc:row={**item,'sequence':None,'error':str(exc)}
                rows.append(row)
            timings.append({'arm':label,'seconds':time.monotonic()-start,'unique_code_sequences':len(keys),
                'batch_size':variant['batch_size'],'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),**monitor.report()})
            del model,predicted;gc.collect();torch.cuda.empty_cache()
    write_json(root/'decoded.json',rows);write_json(root/'decode_timings.json',timings)
    write_json(root/'decoded_manifest.json',{'encoded_manifest_sha256':file_hash(root/'encoded_manifest.json'),
        'decoded_sha256':file_hash(root/'decoded.json'),'independent_process':True,'source_geometry_or_reference_read_by_decoder':False})
    write_json(root/'state.json',{'status':'decoded','cases':len(rows),'next':'CPU geometry comparison'})
    print(json.dumps({'status':'decoded','outputs':len(rows),'timings':timings},indent=2),flush=True)
