"""Reconstruction evaluation of paper-coverage tokenizer checkpoints on a fresh validation panel."""
from pathlib import Path
import json
import sqlite3
import statistics

from .common import file_hash, read_config, require_container, write_json
from .tokenizer_experiment import features, select_cases, evaluate_checkpoint
from .tokenizer_learning_curve import paired_comparison


def _summary(rows):
    valid_reference = [r for r in rows if r['known_valid_reference']]
    paired = [r for r in valid_reference if r['paired_valid']]
    volume = [abs(r['geometry']['reconstruction']['volume'] - r['geometry']['reference']['volume'])
              / r['geometry']['reference']['volume'] for r in paired]
    return {'cases': len(rows), 'known_valid_references': len(valid_reference), 'valid': len(paired),
            'mean_chamfer': statistics.mean(r['chamfer'] for r in paired) if paired else None,
            'median_relative_volume_error': statistics.median(volume) if volume else None}


def run_coverage_eval(settings):
    require_container()
    root = Path(settings['output'])
    root.mkdir(parents=True, exist_ok=True)
    database = settings['database']
    split = settings.get('split', 'val')
    with sqlite3.connect(database) as db:
        pool = [{'id': key, 'sequence': sequence, 'component': component} for key, sequence, component in db.execute(
            'SELECT id,sequence,component FROM geometry WHERE tokenizer_candidate=1 AND split=? AND (tokenizer_split=? OR ?="test")',
            (split, split, split))]
        train_components = {c for (c,) in db.execute('SELECT DISTINCT component FROM geometry WHERE split="train"')}
    if any(r['component'] in train_components for r in pool):
        raise ValueError('Validation panel overlaps training components')
    cases = select_cases(pool, settings['case_count'], settings['seed'])
    for case in cases:
        case['features'] = features(case['sequence'])
    manifest = {'database_sha256': file_hash(database), 'seed': settings['seed'], 'pool': len(pool),
                'case_ids': [c['id'] for c in cases], 'checkpoints': settings['checkpoints']}
    path = root / 'manifest.json'
    if path.exists() and read_config(path) != manifest:
        raise ValueError('Panel provenance changed; use a fresh output directory')
    write_json(path, manifest)
    eval_settings = {k: settings[k] for k in ('seed', 'point_count', 'kernel_workers')}
    reports = {}
    for name, checkpoint in settings['checkpoints'].items():
        evaluate_checkpoint(checkpoint, cases, eval_settings, root / name)
        reports[name] = read_config(root / name / 'per_case.json')
    small = {c['id'] for c in cases if c['features']['max_curve_primitives'] <= settings['small_slots']}
    result = {'panel': {'cases': len(cases), 'with_at_most_small_slots': len(small)}, 'models': {}, 'paired_vs_baseline': {}}
    for name, rows in reports.items():
        result['models'][name] = {'full': _summary(rows),
                                  'small_slots': _summary([r for r in rows if r['id'] in small])}
    base = settings['baseline']
    for name, rows in reports.items():
        if name != base:
            subset = lambda rs: [r for r in rs if r['id'] in small]
            result['paired_vs_baseline'][name] = {
                'small_slots': paired_comparison(subset(reports[base]), subset(rows)),
                'full': paired_comparison(reports[base], rows)}
    write_json(root / 'results.json', result)
    print(json.dumps(result), flush=True)
