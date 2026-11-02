"""Scale/origin-trailer (hybrid) packets for coverage tokenizers, scored on the coverage panel."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sqlite3
import statistics

from .common import read_config, require_container, write_json
from .tokenizer_experiment import features, select_cases
from .tokenizer_learning_curve import paired_comparison
from .tokenizer_coverage_eval import _summary


def _roundtrip(checkpoint, cases):
    from .hybrid_tokenizer import HybridTokenizer
    from .scale_origin_codec import unpack
    tokenizer = HybridTokenizer(checkpoint)
    rows = []
    for index, case in enumerate(cases):
        row = {'id': case['id'], 'features': case['features'], 'sequence': None, 'error': None, 'tokens': None,
               'plain_tokens': None, 'bytes': None}
        try:
            packet = tokenizer.encode(case['sequence'])
            parsed = unpack(packet)
            row.update(tokens=parsed['token_count'], bytes=parsed['serialized_bytes'],
                       plain_tokens=parsed['token_count'] - parsed['metadata_tokens'])
            row['sequence'] = tokenizer.decode(packet)
        except ValueError as error:
            row['error'] = str(error)
        rows.append(row)
        if (index + 1) % 32 == 0:
            print(json.dumps({'phase': 'roundtrip', 'checkpoint': checkpoint, 'cases': index + 1}), flush=True)
    del tokenizer
    return rows


def _score(rows, cases, settings, root):
    import numpy as np
    from scipy.spatial import cKDTree
    from .bench_metrics import normalize_points
    from .bench_runtime import KernelRunner
    geometry_settings = {**read_config('configs/benchmark.json'), 'output': str(root), 'render': False,
                         'seed': settings['seed'], 'point_count': settings['point_count']}
    runner = KernelRunner(geometry_settings)
    originals = {c['id']: c['sequence'] for c in cases}
    jobs = [(r['id'], role, originals[r['id']] if role == 'reference' else r['sequence'])
            for r in rows for role in ('reference', 'reconstruction')]
    with ThreadPoolExecutor(max_workers=settings['kernel_workers']) as pool:
        results = list(pool.map(lambda j: runner.reconstruct(j[2], j[0], j[1]), jobs))
    geometry = {(j[0], j[1]): g for j, g in zip(jobs, results)}
    for row in rows:
        reference, prediction = [geometry[(row['id'], role)] for role in ('reference', 'reconstruction')]
        row['geometry'] = {'reference': reference, 'reconstruction': prediction}
        row['known_valid_reference'] = reference['status'] == 'valid'
        row['paired_valid'] = reference['status'] == prediction['status'] == 'valid'
        row['chamfer'] = None
        if row['paired_valid']:
            a, b = [np.asarray(normalize_points(np.load(g['points']), geometry_settings['normalization']))
                    for g in (reference, prediction)]
            row['chamfer'] = float(np.mean(cKDTree(b).query(a)[0] ** 2) + np.mean(cKDTree(a).query(b)[0] ** 2))


def run_coverage_hybrid(settings):
    require_container()
    from .bench_runtime import evaluation_doctor
    from .scale_origin_experiment import numerical_context
    root = Path(settings['output'])
    root.mkdir(parents=True, exist_ok=True)
    split = settings.get('split', 'val')
    with sqlite3.connect(settings['database']) as db:
        pool = [{'id': k, 'sequence': s, 'component': c} for k, s, c in db.execute(
            'SELECT id,sequence,component FROM geometry WHERE tokenizer_candidate=1 AND split=? AND (tokenizer_split=? OR ?="test")',
            (split, split, split))]
    cases = select_cases(pool, settings['case_count'], settings['seed'])
    for case in cases:
        case['features'] = features(case['sequence'])
    panel = read_config(settings['panel_manifest'])['case_ids']
    if [c['id'] for c in cases] != panel:
        raise ValueError('Panel differs from the plain-codec evaluation panel')
    evaluation_doctor()
    reports = {}
    for name, checkpoint in settings['checkpoints'].items():
        out = root / name
        out.mkdir(parents=True, exist_ok=True)
        if (out / 'per_case.json').exists():
            reports[name] = read_config(out / 'per_case.json')
            continue
        with numerical_context():
            rows = _roundtrip(checkpoint, cases)
        import torch
        torch.cuda.empty_cache()
        _score(rows, cases, settings, out)
        write_json(out / 'per_case.json', rows)
        reports[name] = rows
    plain = {name: read_config(path) for name, path in settings['plain_reports'].items()}
    small = {c['id'] for c in cases if c['features']['max_curve_primitives'] <= settings['small_slots']}
    keep = lambda rs: [r for r in rs if r['id'] in small]
    result = {'models': {}, 'paired': {}, 'tokens': {}}
    for name, rows in {**plain, **reports}.items():
        result['models'][name] = {'full': _summary(rows), 'small_slots': _summary(keep(rows))}
    for name, rows in reports.items():
        counted = [r for r in rows if r['tokens'] is not None]
        result['tokens'][name] = {'encoded_cases': len(counted), 'tokens_with_trailer': sum(r['tokens'] for r in counted),
                                  'tokens_without_trailer': sum(r['plain_tokens'] for r in counted),
                                  'bytes': sum(r['bytes'] for r in counted)}
        for ref in settings['paired_against']:
            result['paired'][f'{name} vs {ref}'] = {
                'small_slots': paired_comparison(keep(plain.get(ref) or reports[ref]), keep(rows)),
                'full': paired_comparison(plain.get(ref) or reports[ref], rows)}
    write_json(root / 'results.json', result)
    print(json.dumps(result), flush=True)
