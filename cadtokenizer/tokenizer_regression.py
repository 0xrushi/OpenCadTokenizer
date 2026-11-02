"""Read-only diagnosis of checkpoint drift and quantizer precision sensitivity."""
from __future__ import annotations

import collections
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sqlite3
import statistics

from .common import file_hash, fraction, read_config, require_container, seed_all, write_json
from .representation import PAD, analyze, split_pairs, stringify


def unique_geometry_jobs(jobs):
    """Identical branches share a cache key and must not write it concurrently."""
    return list(dict.fromkeys(jobs))


def select_cases(source, baseline_rows, database, limit, seed):
    known = {r['id'] for r in baseline_rows if r['known_valid_reference']}
    candidates = [r for r in source['validation'] if r['id'] in known]
    selected = sorted(candidates, key=lambda r: (fraction(r['id'], seed), r['id']))[:limit]
    if len(selected) != limit:
        raise ValueError('Not enough known-valid validation originals')
    train_ids = {r['id'] for r in source['train']}
    if train_ids & {r['id'] for r in selected}:
        raise ValueError('Diagnostic selection overlaps training')
    with sqlite3.connect(database) as db:
        for row in selected:
            stored = db.execute('SELECT sequence,split,tokenizer_split,component FROM geometry WHERE id=?', (row['id'],)).fetchone()
            if stored != (row['sequence'], 'val', 'val', row['component']):
                raise ValueError('Diagnostic case is not verified validation geometry')
    return selected


def quantizer_paths(model, pooled, mask):
    """Hold vectors fixed, varying only the nearest-code distance arithmetic."""
    import torch
    _, actual, _ = model.quantize(pooled, mask)
    with torch.autocast(device_type=pooled.device.type, dtype=torch.bfloat16, enabled=pooled.device.type=='cuda'):
        _, mixed, _ = model.quantize(pooled, mask)
    # Float64 distance reference, independent of autocast and TF32 settings.
    x, table = pooled.double(), model.codebook.weight.double()
    distances = (x.square().sum(-1, keepdim=True) + table.square().sum(-1) - 2*x @ table.t()).clamp_min(0)
    reference = distances.argmin(-1)
    smallest = distances.topk(2, largest=False).values
    result = {'continuous': pooled, 'quantized': model.codebook(actual),
              'quantized_amp': model.codebook(mixed), 'quantized_reference': model.codebook(reference)}
    regret = {}
    for name, indices in [('quantized', actual), ('quantized_amp', mixed)]:
        chosen = distances.gather(-1, indices[..., None]).squeeze(-1)
        regret[name] = {'reference_agreement': int(((indices == reference) & mask).sum()),
                        'distance_regret_sum': float((chosen - smallest[..., 0])[mask].sum())}
    stats = {'vectors': int(mask.sum()), 'current_vs_amp_agreement': int(((actual == mixed) & mask).sum()),
             'nearest_margin_sum': float((smallest[..., 1]-smallest[..., 0])[mask].sum()),
             'precision': regret}
    return result, {'quantized': actual, 'quantized_amp': mixed, 'quantized_reference': reference}, stats


def infer_checkpoint(path, cases, settings, output):
    import torch
    import numpy as np
    from .vq import collate_pairs, load_vq, to_device
    from .tokenizer_experiment import field_stats, merge_fields
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'decoded.json').exists():
        return read_config(output/'decoded.json')
    model, checkpoint = load_vq(path, torch.device('cuda'))
    model.eval()
    modes = ('continuous', 'quantized', 'quantized_amp', 'quantized_reference')
    records, pair_records = [], []
    pooled_arrays, code_arrays = [], {name: [] for name in modes if name!='continuous'}
    usage = {name: collections.Counter() for name in code_arrays}
    precision = collections.Counter()
    regrets = {name: collections.Counter() for name in ('quantized','quantized_amp')}
    mse = {name: 0. for name in code_arrays}
    elements = 0
    with torch.inference_mode():
        for index, case in enumerate(cases):
            data = model.config['data']
            _, pairs = analyze(case['sequence'], data['max_pair_tokens'], data['max_primitives'],
                               data['extrusion_partition'], data.get('sketch_partition','curve'))
            batch = to_device(collate_pairs(pairs), torch.device('cuda'))
            pooled, mask = model.pool(batch['tokens'], batch['groups'])
            memories, codes, statistics_ = quantizer_paths(model, pooled, mask)
            precision.update({key: statistics_[key] for key in ('vectors','current_vs_amp_agreement','nearest_margin_sum')})
            for name in regrets: regrets[name].update(statistics_['precision'][name])
            pooled_arrays.append(pooled[mask].cpu().numpy())
            elements += pooled[mask].numel()
            for name in code_arrays:
                values = codes[name][mask].cpu().tolist()
                code_arrays[name].extend(values)
                usage[name].update(values)
                mse[name] += float((pooled[mask]-memories[name][mask]).square().sum())
            for ordinal, pair in enumerate(pairs):
                pair_records.append({'id':case['id'],'ordinal':ordinal,'tokens':pair['tokens'],
                                     'codes':{name:codes[name][ordinal][mask[ordinal]].cpu().tolist() for name in code_arrays}})
            for mode in modes:
                fields, tokens, errors = [], [], []
                for ordinal, pair in enumerate(pairs):
                    memory = memories[mode][ordinal][mask[ordinal]]
                    logits = model.decode(batch['decoder_input'][ordinal:ordinal+1], memory[None],
                                          torch.ones((1,len(memory)), dtype=torch.bool, device=memory.device))
                    fields.append(field_stats(logits,batch['targets'][ordinal:ordinal+1],pair['tokens']))
                    try:
                        tokens.extend(model.reconstruct(memory,force_add=ordinal==0))
                    except ValueError as error:
                        errors.append(str(error))
                sequence = stringify(tokens) if not errors else None
                if sequence is not None:split_pairs(tokens)
                records.append({'id':case['id'],'mode':mode,'sequence':sequence,'errors':errors,
                                'exact_sequence':sequence==case['sequence'],'teacher':merge_fields(fields)})
            if (index+1)%8==0:print(json.dumps({'phase':'decode','checkpoint':str(path),'shapes':index+1,'total':len(cases)}),flush=True)
        arrays = {'pooled':np.concatenate(pooled_arrays), 'codebook':model.codebook.weight.cpu().numpy()}
        arrays.update({name:np.asarray(values,dtype=np.int64) for name,values in code_arrays.items()})
        np.savez(output/'latents.npz',**arrays)
    n = precision['vectors']
    quantization = {'vectors':n,'nearest_margin_mean':precision['nearest_margin_sum']/n,
                    'current_vs_amp_agreement':precision['current_vs_amp_agreement']/n,
                    'mse':{key:value/elements for key,value in mse.items()},'modes':{}}
    for name in code_arrays:
        probabilities = [count/n for count in usage[name].values()]
        import math
        value = {'used_codes':len(usage[name]), 'perplexity':math.exp(-sum(p*math.log(p) for p in probabilities))}
        if name in regrets:
            value.update(reference_agreement=regrets[name]['reference_agreement']/n,
                         mean_squared_distance_regret=regrets[name]['distance_regret_sum']/n)
        quantization['modes'][name] = value
    decoded = {'checkpoint_sha256':file_hash(path),'step':checkpoint['step'],'records':records,
               'pairs':pair_records,'quantization':quantization}
    write_json(output/'decoded.json',decoded)
    del model,checkpoint
    torch.cuda.empty_cache()
    return decoded


def score(decoded, cases, settings, output):
    import numpy as np
    from scipy.spatial import cKDTree
    from .bench_metrics import normalize_points
    from .tokenizer_experiment import merge_fields
    from .tokenizer_overfit import kernel_runner
    output=Path(output)
    if (output/'report.json').exists():return read_config(output/'report.json')
    runner=kernel_runner(settings)
    jobs=[(case['id'],'reference',case['sequence']) for case in cases]
    jobs += [(row['id'],'prediction',row['sequence']) for row in decoded['records']]
    unique_jobs=unique_geometry_jobs(jobs)
    results={}
    with ThreadPoolExecutor(max_workers=settings['kernel_workers']) as pool:
        for index,(job,result) in enumerate(zip(unique_jobs,pool.map(lambda j:runner.reconstruct(j[2],j[0],j[1]),unique_jobs))):
            results[job]=result
            if (index+1)%64==0:print(json.dumps({'phase':'kernel','step':decoded['step'],'jobs':index+1,'total':len(unique_jobs)}),flush=True)
    refs={case['id']:results[(case['id'],'reference',case['sequence'])] for case in cases}
    for reference in refs.values():
        if reference['status']!='valid':raise RuntimeError('A selected original is no longer kernel-valid')
    records=[]
    for index,row in enumerate(decoded['records'],len(cases)):
        row=dict(row)
        reference,prediction=refs[row['id']],results[jobs[index]]
        row.update(geometry=prediction,chamfer=None,relative_volume_error=None)
        if prediction['status']=='valid':
            a,b=[np.asarray(normalize_points(np.load(item['points']))) for item in (reference,prediction)]
            row['chamfer']=float(np.mean(cKDTree(b).query(a)[0]**2)+np.mean(cKDTree(a).query(b)[0]**2))
            row['relative_volume_error']=abs(prediction['volume']-reference['volume'])/reference['volume']
        records.append(row)
    branches={}
    for mode in ('continuous','quantized','quantized_amp','quantized_reference'):
        rows=[row for row in records if row['mode']==mode]
        valid=[row for row in rows if row['geometry']['status']=='valid']
        branches[mode]={'cases':len(rows),'valid_solids':len(valid),'exact_sequences':sum(row['exact_sequence'] for row in rows),
                        'teacher':merge_fields(row['teacher'] for row in rows),
                        'mean_chamfer':statistics.mean(row['chamfer'] for row in valid) if valid else None,
                        'median_relative_volume_error':statistics.median(row['relative_volume_error'] for row in valid) if valid else None,
                        'failures':dict(collections.Counter(row['geometry'].get('error',str(row['errors'])) for row in rows if row['geometry']['status']!='valid'))}
    report={**decoded,'records':records,'branches':branches}
    write_json(output/'report.json',report)
    return report


def drift(earlier, later, earlier_dir, later_dir):
    import numpy as np
    a,b=[np.load(Path(path)/'latents.npz') for path in (earlier_dir,later_dir)]
    assert [(p['id'],p['ordinal'],p['tokens']) for p in earlier['pairs']]==[(p['id'],p['ordinal'],p['tokens']) for p in later['pairs']]
    codes={name:float(np.mean(a[name]==b[name])) for name in ('quantized','quantized_amp','quantized_reference')}
    def norm(x):return float(np.sqrt(np.mean(x.astype(np.float64)**2)))
    return {'same_index_assignment_agreement':codes,
            'encoder_pooled_rms_change':norm(a['pooled']-b['pooled']),
            'encoder_pooled_relative_rms_change':norm(a['pooled']-b['pooled'])/norm(a['pooled']),
            'codebook_rms_change':norm(a['codebook']-b['codebook']),
            'codebook_relative_rms_change':norm(a['codebook']-b['codebook'])/norm(a['codebook']),
            'limits':'Representation drift/reassignment can be harmless coadaptation; these statistics alone do not identify a causal component.'}


def run_regression(config,settings):
    require_container()
    import torch
    from .bench_data import code_hash
    from .bench_runtime import evaluation_doctor
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high');seed_all(settings['seed'])
    environment=evaluation_doctor()
    root=Path(settings['output']);root.mkdir(parents=True,exist_ok=True)
    signature={'config':config,'settings':settings,'code_sha256':code_hash(),'environment':environment,
               'source_sha256':file_hash(settings['source_manifest']),'baseline_sha256':file_hash(settings['baseline_cases']),
               'database_sha256':file_hash(config['data']['database']),
               'checkpoints':{name:file_hash(path) for name,path in settings['checkpoints'].items()}}
    manifest_path=root/'manifest.json'
    if manifest_path.exists():
        manifest=read_config(manifest_path)
        if manifest['signature']!=signature:raise ValueError('Diagnostic inputs/code changed; use a fresh output')
    else:
        cases=select_cases(read_config(settings['source_manifest']),read_config(settings['baseline_cases']),
                           config['data']['database'],settings['limit'],settings['seed'])
        manifest={'signature':signature,'cases':cases,'selection':'Hash-seeded known-valid originals from the fixed validation panel only; no candidate-output filtering.'}
        write_json(manifest_path,manifest)
        source=root/'source';source.mkdir(exist_ok=True)
        for file in Path('cadtokenizer').glob('*.py'):shutil.copy2(file,source/file.name)
    reports={}
    try:
        for name,path in settings['checkpoints'].items():
            write_json(root/'state.json',{'status':'running','checkpoint':name,'phase':'decode_and_score'})
            decoded=infer_checkpoint(path,manifest['cases'],settings,root/name)
            if decoded['step'] != int(name):raise ValueError('Checkpoint label and saved step differ')
            reports[name]=score(decoded,manifest['cases'],settings,root/name)
        keys=list(reports)
        changes=drift(reports[keys[0]],reports[keys[1]],root/keys[0],root/keys[1])
        result={'selection':manifest['selection'],'checkpoints':{name:{'branches':r['branches'],'quantization':r['quantization']} for name,r in reports.items()},'drift':changes,
                'limits':['Continuous vectors are an off-training-distribution probe for a VQ-trained decoder, not a separately trained continuous control.',
                          'AMP and float64-reference assignments hold encoder vectors fixed; they diagnose distance arithmetic, not full AMP encoder execution.',
                          'This small validation diagnostic cannot establish causality or select a full production replacement.']}
        write_json(root/'results.json',result)
        lines=['# Read-only checkpoint regression diagnosis','',manifest['selection'],'',
               '| Checkpoint | Memory path | Valid / cases | Teacher accuracy | Mean CD on valid pairs | Median volume error |',
               '| --- | --- | ---: | ---: | ---: | ---: |']
        for name,report in reports.items():
            for mode,s in report['branches'].items():
                lines.append(f'| {name} | {mode} | {s["valid_solids"]}/{s["cases"]} | {s["teacher"]["all"]["accuracy"]:.4f} | {s["mean_chamfer"]} | {s["median_relative_volume_error"]} |')
        lines+=['',*result['limits']]
        (root/'comparison.md').write_text('\n'.join(lines)+'\n')
        for name,path in settings['checkpoints'].items():
            if file_hash(path)!=signature['checkpoints'][name]:raise RuntimeError('Input checkpoint changed during read-only diagnostic')
        write_json(root/'state.json',{'status':'complete'})
    except BaseException as error:
        write_json(root/'state.json',{'status':'failed','error':f'{type(error).__name__}: {error}'})
        raise
