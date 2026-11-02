"""Validation-only oracle repairs of saved SEM outputs; no model execution."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import collections
import json
import statistics

from .common import file_hash, read_config, require_container, write_json
from .representation import split_pairs, stringify, tokenize

GROUPS = {'depth': [1, 2], 'translation': [3, 4, 5], 'rotation': list(range(6, 15)),
          'scale': [15], 'origin': [16, 17],
          'scale_origin': [15, 16, 17], 'all_extrusion': list(range(19))}


def parts(sequence):
    result = []
    for pair in split_pairs(tokenize(sequence)):
        boundary = next(i for i, token in enumerate(pair) if token in (8, 9, 10))
        result.append((pair[:boundary], pair[boundary:]))
    return result


def repair(original, prediction, mode):
    a, b = parts(original), parts(prediction)
    if len(a) != len(b):
        return None
    output = []
    for (sketch_a, ext_a), (sketch_b, ext_b) in zip(a, b):
        sketch, ext = list(sketch_b), list(ext_b)
        if mode == 'sketch':
            sketch = sketch_a
        elif mode == 'sketch_coordinates':
            if [t for t in sketch_a if t < 12] != [t for t in sketch_b if t < 12]:
                return None
            sketch = [a if a >= 12 else b for a, b in zip(sketch_a, sketch_b)]
        else:
            for index in GROUPS[mode]:
                ext[index] = ext_a[index]
        output.extend(sketch + ext)
    return stringify(output)


def run_shape_audit(settings):
    require_container()
    import numpy as np
    from scipy.spatial import cKDTree
    from .bench_metrics import normalize_points
    from .bench_runtime import KernelRunner
    root = Path(settings['output'])
    root.mkdir(parents=True, exist_ok=True)
    source = read_config(settings['manifest'])['validation']
    originals = {row['id']: row['sequence'] for row in source}
    baseline = {r['id']: r for r in read_config(settings['baseline'])}
    candidate = {r['id']: r for r in read_config(settings['candidate'])}
    if baseline.keys() != candidate.keys() or baseline.keys() != originals.keys():
        raise ValueError('Validation IDs differ')
    common = [key for key in originals if baseline[key]['paired_valid'] and candidate[key]['paired_valid']]
    worse = sorted([key for key in common if candidate[key]['chamfer'] > baseline[key]['chamfer']],
                   key=lambda key: candidate[key]['chamfer']-baseline[key]['chamfer'], reverse=True)
    failures = [key for key in originals if baseline[key]['paired_valid'] and not candidate[key]['paired_valid']]
    chosen = worse[:settings['worst_cases']] + failures[:settings['failure_cases']]
    signature = {'settings': settings, 'inputs': {key: file_hash(settings[key]) for key in
                 ('manifest', 'baseline', 'candidate')}, 'module_sha256': file_hash(__file__)}
    write_json(root / 'manifest.json', {'signature': signature, 'case_ids': chosen,
               'selection': 'Largest positive paired CD regressions, plus baseline-valid/candidate-invalid cases in frozen manifest order. Validation only.'})
    write_json(root / 'state.json', {'status': 'running', 'phase': 'geometry', 'cases': len(chosen)})
    modes = settings.get('repair_modes', ['sketch', 'sketch_coordinates', *GROUPS])
    if not modes or any(mode not in ('sketch', 'sketch_coordinates', *GROUPS) for mode in modes):
        raise ValueError('Unknown or empty oracle repair modes')
    jobs, sequences = [], {}
    for key in chosen:
        for mode, sequence in [('baseline', baseline[key]['sequence']), ('candidate', candidate[key]['sequence'])] + [
                (mode, repair(originals[key], candidate[key]['sequence'], mode)) for mode in modes
                if candidate[key]['sequence'] is not None]:
            sequences[(key, mode)] = sequence
            jobs.append((key, 'reconstruction', sequence))
    unique = list(dict.fromkeys(jobs))
    cfg = {**read_config('configs/benchmark.json'), 'output': str(root), 'render': False,
           'seed': 42, 'point_count': 2000}
    runner = KernelRunner(cfg)
    geometry = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, (job, result) in enumerate(zip(unique, pool.map(lambda j: runner.reconstruct(j[2], j[0], j[1]), unique)), 1):
            geometry[job] = result
            if index % 32 == 0:
                print(json.dumps({'jobs': index, 'total': len(unique)}), flush=True)
    records = []
    for key in chosen:
        reference = candidate[key]['geometry']['reference']
        ref_points = np.asarray(normalize_points(np.load(reference['points'])))
        scores = {}
        for mode in ['baseline', 'candidate', *modes]:
            if (key, mode) not in sequences:
                continue
            sequence = sequences[(key, mode)]
            geom = geometry[(key, 'reconstruction', sequence)]
            score = {'status': geom['status'], 'chamfer': None, 'volume_error': None}
            if geom['status'] == 'valid':
                points = np.asarray(normalize_points(np.load(geom['points'])))
                score.update(chamfer=float(np.mean(cKDTree(points).query(ref_points)[0]**2)
                                          +np.mean(cKDTree(ref_points).query(points)[0]**2)),
                             volume_error=abs(geom['volume']-reference['volume'])/reference['volume'])
            scores[mode] = score
        records.append({'id': key, 'subset': 'shape_regression' if key in worse else 'validity_regression',
                        'original': originals[key], 'baseline': baseline[key]['sequence'],
                        'candidate': candidate[key]['sequence'], 'scores': scores})
    summaries = {}
    for mode in modes:
        comparable = [r for r in records if r['subset']=='shape_regression'
                      and r['scores'].get(mode, {}).get('chamfer') is not None]
        failure_rows = [r for r in records if r['subset']=='validity_regression']
        summaries[mode] = {'valid_shape_repairs': len(comparable),
            'median_chamfer_reduction': statistics.median(r['scores']['candidate']['chamfer']-r['scores'][mode]['chamfer'] for r in comparable) if comparable else None,
            'lower_chamfer_cases': sum(r['scores'][mode]['chamfer'] < r['scores']['candidate']['chamfer'] for r in comparable),
            'rescued_invalid_cases': sum(r['scores'].get(mode, {}).get('status')=='valid' for r in failure_rows)}
    errors = collections.Counter()
    structural = collections.Counter()
    for key, row in candidate.items():
        if row['known_valid_reference'] and not row['paired_valid']:
            errors[row['geometry']['reconstruction'].get('error', 'unknown')] += 1
        if row['sequence'] is not None:
            a, b = parts(originals[key]), parts(row['sequence'])
            structural['cases'] += 1
            structural['different_sketch_structure'] += len(a)!=len(b) or any(
                [t for t in x[0] if t<12] != [t for t in y[0] if t<12] for x,y in zip(a,b))
    result = {'signature': signature, 'common_valid_cases': len(common),
              'candidate_worse_chamfer_cases': len(worse), 'validity_regressions': len(failures),
              'selected_shape_regressions': min(len(worse),settings['worst_cases']),
              'selected_validity_regressions': min(len(failures),settings['failure_cases']),
              'oracle_repairs': summaries, 'invalid_candidate_errors': dict(errors),
              'structural_comparison': dict(structural), 'records': records}
    write_json(root / 'results.json', result)
    lines = ['# Curve 40k validation shape-error diagnosis', '',
             f'{len(common)} shared valid cases; candidate has worse Chamfer on {len(worse)}. Audit covers {min(len(worse),settings["worst_cases"])} largest CD regressions and {min(len(failures),settings["failure_cases"])} validity regressions.', '',
             '| Oracle repair | Valid shape repairs | Median CD reduction | Lower CD cases | Invalid cases rescued |',
             '| --- | ---: | ---: | ---: | ---: |']
    for mode, item in summaries.items():
        lines.append(f'| {mode} | {item["valid_shape_repairs"]} | {item["median_chamfer_reduction"]} | {item["lower_chamfer_cases"]} | {item["rescued_invalid_cases"]} |')
    lines += ['', 'Repairs substitute ground-truth SEM fields into saved predictions. They are oracle diagnostics, not deployable corrections. Effects interact and are not additive. The worst-case selection does not estimate population prevalence. Sketch repair includes topology and coordinates; coordinate-only repair requires matching structural tokens. Separate per-shape normalization makes Chamfer insensitive to pure uniform scaling about the origin; volume errors remain in results.json.', '',
              'All outputs were generated previously; this audit trains no model and consults no test cases. Saved reference points and identical per-case reconstruction sampling seeds keep geometry comparisons aligned.']
    (root / 'findings.md').write_text('\n'.join(lines)+'\n')
    write_json(root / 'state.json', {'status':'complete','results':str(root/'results.json')})
