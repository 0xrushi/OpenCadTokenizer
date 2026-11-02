"""Geometry/size comparison, after the independent codec decoder has exited."""
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
from pathlib import Path
import statistics

from .common import read_config,write_json,file_hash,require_container
from .representation import tokenize,split_pairs
from .scale_origin_codec import unpack,metadata_for_pair


def paired(a,b):
    a={r['id']:r for r in a};b={r['id']:r for r in b}
    if a.keys()!=b.keys():raise ValueError('Comparison IDs differ')
    common=[k for k in a if a[k]['paired_valid'] and b[k]['paired_valid']]
    old=statistics.mean(a[k]['chamfer'] for k in common) if common else None
    new=statistics.mean(b[k]['chamfer'] for k in common) if common else None
    av=sum(r['paired_valid'] for r in a.values());bv=sum(r['paired_valid'] for r in b.values())
    return {'shared_valid_cases':len(common),'baseline_valid':av,'candidate_valid':bv,
        'baseline_mean_chamfer':old,'candidate_mean_chamfer':new,
        'candidate_lower_chamfer_cases':sum(b[k]['chamfer']<a[k]['chamfer'] for k in common),
        'baseline_median_volume_error':statistics.median(a[k]['volume_error'] for k in common) if common else None,
        'candidate_median_volume_error':statistics.median(b[k]['volume_error'] for k in common) if common else None,
        'valid_to_invalid':sum(a[k]['paired_valid'] and not b[k]['paired_valid'] for k in a),
        'invalid_to_valid':sum(not a[k]['paired_valid'] and b[k]['paired_valid'] for k in a),
        'passes_reconstruction_screen':bv>av and bool(common) and new<=old}


def evaluate_study():
    require_container()
    import os
    import numpy as np
    from scipy.spatial import cKDTree
    from .bench_runtime import KernelRunner
    from .bench_metrics import normalize_points
    settings=read_config('configs/tokenizer-scale-origin.json');root=Path(settings['output'])
    if (root/'results.json').exists():
        manifest=read_config(root/'evaluation_manifest.json')
        for path,digest in manifest['hashes'].items():
            if file_hash(path)!=digest:raise ValueError('Completed evaluation provenance changed')
        print(json.dumps(read_config(root/'results.json'),indent=2));return
    protocol=read_config(root/'protocol.json')
    for path,digest in protocol['hashes'].items():
        if file_hash(path)!=digest:raise ValueError('Frozen study/code/checkpoint changed')
    if file_hash(root/'decoded.json')!=read_config(root/'decoded_manifest.json')['decoded_sha256']:raise ValueError('Decoded outputs changed')
    cases=read_config(settings['validation_manifest'])['validation'];original={c['id']:c for c in cases}
    rows=read_config(root/'decoded.json');arms=['existing','curve','hybrid','lossless']
    if len(rows)!=2048 or any({r['id'] for r in rows if r['arm']==arm}!=set(original) for arm in arms):raise ValueError('Incomplete original validation coverage')
    old_report=read_config(settings['historical_report'])
    historical={'existing':read_config(settings['historical_baseline']),'curve':read_config(settings['historical_candidate'])}
    geometry_names=['geometry_worker.py','representation.py','bench_runtime.py','bench_metrics.py']
    same_geometry_code=all(file_hash(Path('cadtokenizer')/p)==old_report['signature']['code'][p] for p in geometry_names)
    same_packages=all(importlib.metadata.version(k)==v for k,v in old_report['environment']['versions'].items())
    reuse=(same_geometry_code and same_packages and old_report['signature']['image_id']==os.environ.get('CADTOKENIZER_IMAGE_ID')
           and old_report['signature']['cases']==cases and old_report['signature']['settings']['seed']==settings['seed']
           and old_report['signature']['settings']['point_count']==settings['point_count'])
    cfg={**read_config('configs/benchmark.json'),'output':str(root),'render':False,'seed':settings['seed'],'point_count':settings['point_count']}
    runner=KernelRunner(cfg);references={};cached={};cache_asset_hashes={}
    if reuse:
        for arm,old_rows in historical.items():
            for row in old_rows:
                if row['id'] not in original:raise ValueError('Cached geometry IDs differ')
                for role,key in [('reference','reference'),('reconstruction','reconstruction')]:
                    value=row['geometry'][key]
                    if value['status']=='valid':
                        for field in ['points','step','stl']:
                            path=value[field]
                            if not Path(path).is_file():raise ValueError('Missing cached geometry asset')
                            cache_asset_hashes[path]=file_hash(path)
                    if role=='reference':references.setdefault(row['id'],value)
                    elif row['sequence'] is not None:cached.setdefault((row['id'],row['sequence']),value)
    jobs=[]
    for case in cases:
        if case['id'] not in references:jobs.append((case['id'],'reference',case['sequence']))
    for row in rows:
        if (row['id'],row['sequence']) not in cached:jobs.append((row['id'],'reconstruction',row['sequence']))
    jobs=list(dict.fromkeys(jobs));geometry={}
    write_json(root/'geometry_reuse.json',{'reuse_permitted':reuse,'geometry_source_hashes_match':same_geometry_code,
        'package_versions_match':same_packages,'same_original_cases':old_report['signature']['cases']==cases,
        'cached_reference_count':len(references),'cached_decoded_sequence_count':len(cached),
        'new_unique_geometry_jobs':len(jobs),'historical_report_sha256':file_hash(settings['historical_report']),
        'cached_asset_hashes':cache_asset_hashes,
        'scope':'Reuse only identical sequence/ID/role with unchanged worker/scorer/package/image/seed settings. No historical model-quality numbers substituted for fresh decodes.'})
    with ThreadPoolExecutor(max_workers=settings['kernel_workers']) as pool:
        for index,(job,result) in enumerate(zip(jobs,pool.map(lambda j:runner.reconstruct(j[2],j[0],j[1]),jobs)),1):
            geometry[job]=result
            if index%8==0 or index==len(jobs):write_json(root/'state.json',{'status':'running','phase':'geometry','completed':index,'total':len(jobs)})
    for case in cases:
        if case['id'] not in references:references[case['id']]=geometry[(case['id'],'reference',case['sequence'])]
    metadata_checks=lossless_exact=0;costs={(r['arm'],r['id']):r for r in read_config(root/'packet_costs.json')}
    for row in rows:
        key=row['id'];ref=references[key]
        prediction=cached.get((key,row['sequence']))
        if prediction is None:prediction=geometry[(key,'reconstruction',row['sequence'])]
        row['geometry']={'reference':ref,'reconstruction':prediction}
        row['known_valid_reference']=ref['status']=='valid';row['paired_valid']=ref['status']==prediction['status']=='valid'
        row['geometry_reused']=((key,row['sequence']) in cached)
        row['chamfer']=row['volume_error']=None
        row['cost']=costs[(row['arm'],key)]
        row['exact_sem']=row['sequence'] is not None and tokenize(row['sequence'])==tokenize(original[key]['sequence'])
        if row['paired_valid']:
            a,b=[np.asarray(normalize_points(np.load(g['points']),cfg['normalization'])) for g in [ref,prediction]]
            row['chamfer']=float(np.mean(cKDTree(b).query(a)[0]**2)+np.mean(cKDTree(a).query(b)[0]**2))
            row['volume_error']=abs(prediction['volume']-ref['volume'])/ref['volume']
        if row['arm']=='hybrid':
            encoded=unpack(Path(row['packet']).read_bytes());truth=[metadata_for_pair(p) for p in split_pairs(tokenize(original[key]['sequence']))]
            if encoded['metadata']!=truth:raise AssertionError('Encoder did not preserve original discrete metadata')
            if row['sequence'] is not None:
                restored=[metadata_for_pair(p) for p in split_pairs(tokenize(row['sequence']))]
                if restored!=encoded['metadata']:raise AssertionError('Packet-only decoder lost literal metadata')
                metadata_checks+=len(restored)
        if row['arm']=='lossless':
            if not row['exact_sem']:raise AssertionError('Lossless control changed the original SEM')
            lossless_exact+=1
    summary={}
    for arm in arms:
        chosen=[r for r in rows if r['arm']==arm];valid=[r for r in chosen if r['paired_valid']]
        summary[arm]={'cases':len(chosen),'valid_input_shapes':sum(r['known_valid_reference'] for r in chosen),
            'decoded_sequences':sum(r['sequence'] is not None for r in chosen),'valid_of_valid_inputs':len(valid),
            'valid_reconstructions_all_inputs':sum(r['geometry']['reconstruction']['status']=='valid' for r in chosen),
            'exact_sem':sum(r['exact_sem'] for r in chosen),'conditional_mean_chamfer':statistics.mean(r['chamfer'] for r in valid) if valid else None,
            'median_relative_volume_error':statistics.median(r['volume_error'] for r in valid) if valid else None,
            'total_tokens':sum(r['cost']['token_count'] for r in chosen),'mean_tokens':statistics.mean(r['cost']['token_count'] for r in chosen),
            'total_payload_bits':sum(r['cost']['payload_bits'] for r in chosen),'total_packet_bytes':sum(r['cost']['serialized_bytes'] for r in chosen),
            'framing_tokens':sum(r['cost']['framing_tokens'] for r in chosen),'literal_metadata_tokens':sum(r['cost']['metadata_tokens'] for r in chosen)}
    byarm={a:[r for r in rows if r['arm']==a] for a in arms}
    comparisons={'curve_vs_existing':paired(byarm['existing'],byarm['curve']),
        'hybrid_vs_existing':paired(byarm['existing'],byarm['hybrid']),
        'hybrid_vs_curve':paired(byarm['curve'],byarm['hybrid'])}
    ratio=summary['hybrid']['total_tokens']/summary['curve']['total_tokens']
    results={'status':'complete','summaries':summary,'comparisons':comparisons,
        'hybrid_passes_reconstruction_screen':comparisons['hybrid_vs_existing']['passes_reconstruction_screen'],
        'hybrid_token_overhead_vs_curve':ratio-1,
        'hybrid_byte_overhead_vs_curve':summary['hybrid']['total_packet_bytes']/summary['curve']['total_packet_bytes']-1,
        'hybrid_tokens_vs_lossless_fraction':summary['hybrid']['total_tokens']/summary['lossless']['total_tokens'],
        'metadata_triples_verified_after_decode':metadata_checks,'lossless_sem_exact':lossless_exact,
        'checkpoint_inference':read_config(root/'decode_timings.json'),'profiles':read_config(root/'profile_selection.json'),
        'training_started':False,'new_model_weights':False,'production_promoted':False,
        'limits':'Same repeatedly used 512-shape validation panel; metadata adds real token/bit cost, not equal-rate compression. Exactness covers original discrete SEM bins, not pre-quantization CAD. Reconstruction is not downstream generation quality. Lossless control has independently sampled reconstruction points, so exact geometry can have nonzero Monte Carlo Chamfer.',
        'next':'Consider a separately frozen independent reconstruction check and matched downstream codec study only after reviewing quality versus added rate; do not promote or claim equal-budget superiority.'}
    write_json(root/'per_case.json',rows);write_json(root/'results.json',results)
    files=[str(root/n) for n in ['protocol.json','encoded_manifest.json','decode_inputs.json','decoded_manifest.json','decoded.json',
        'packet_costs.json','profile_selection.json','geometry_reuse.json','per_case.json','results.json']]
    files += [str(p) for p in sorted((root/'packets').rglob('*.ctp'))]
    write_json(root/'evaluation_manifest.json',{'hashes':{p:file_hash(p) for p in files},'cached_geometry_asset_hashes':cache_asset_hashes})
    write_json(root/'state.json',{'status':'complete','hybrid_passes_reconstruction_screen':results['hybrid_passes_reconstruction_screen']})
    print(json.dumps(results,indent=2),flush=True)
