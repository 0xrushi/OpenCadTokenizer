"""Geometry/metric integration check. Uses fixed fixtures, never a trained model."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .bench_data import content_hash, code_hash
from .bench_score import benchmark_score
from .common import file_hash, require_container, write_json


def rectangle_sem(upper=63, operation='add', low=10, high=50):
    sketch = ' '.join(f'line,{x},{y} <curve_end>' for x, y in [(low, low), (high, low), (high, high), (low, high)])
    return sketch + f' <loop_end> <face_end> <sketch_end> {operation},0,{upper},31,31,31,1,0,0,0,1,0,0,0,1,45,31,31 <extrude_end>'


def circle_sem():
    return 'circle,10,31,50,31,31,10,31,50 <curve_end> <loop_end> <face_end> <sketch_end> add,0,63,31,31,31,1,0,0,0,1,0,0,0,1,45,31,31 <extrude_end>'


def benchmark_smoke(config, settings, directory='artifacts/benchmark-smoke'):
    require_container()
    root = Path(directory)
    if root.exists():
        raise FileExistsError('Geometry smoke output exists; use another --directory for a fresh check')
    root.mkdir(parents=True)
    settings = copy.deepcopy(settings)
    settings.update(output=str(root), point_count=200, grid_resolution=8, render=True, distribution_workers=1)
    rectangle, circle = rectangle_sem(), circle_sem()
    edited = rectangle_sem(upper=45)
    manifest = {'schema': 'cad-benchmark-v1', 'paper_test_set_verified': False,
                'comparability_notes': ['Synthetic geometry smoke fixtures; no trained model or benchmark score claim.'],
                'samples': [
                    {'id': 'smoke:g:valid', 'task': 'generation', 'instruction': 'Create a rectangular prism.', 'source_sequence': None, 'target_sequence': rectangle},
                    {'id': 'smoke:g:invalid', 'task': 'generation', 'instruction': 'Create a cylinder.', 'source_sequence': None, 'target_sequence': circle},
                    {'id': 'smoke:e:copied', 'task': 'editing', 'instruction': 'Shorten the extrusion.', 'source_sequence': rectangle, 'target_sequence': edited},
                    {'id': 'smoke:e:failure', 'task': 'editing', 'instruction': 'Shorten the extrusion.', 'source_sequence': rectangle, 'target_sequence': edited},
                ]}
    manifest_path = root / 'manifest.json'
    write_json(manifest_path, manifest)
    signature = {'manifest_sha256': file_hash(manifest_path), 'fixture_run': True, 'code_sha256': code_hash()}
    signature_hash = content_hash(signature)
    rows = []
    for sample, text in zip(manifest['samples'], [rectangle, 'malformed CAD', rectangle, None]):
        rows.append({'sample_id': sample['id'], 'generated_sequence': text,
                     'status': 'syntax_valid' if text is not None else 'inference_error',
                     'error': 'Deliberate fixture failure' if text is None else None,
                     'manifest_sha256': signature['manifest_sha256'], 'run_signature_sha256': signature_hash})
    with (root / 'predictions.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    write_json(root / 'predictions.meta.json', {'schema': 'cad-predictions-v1', 'signature': signature, 'signature_sha256': signature_hash})
    report = benchmark_score(config, settings)
    for task in ('generation', 'editing'):
        if report['tasks'][task]['invalid_predictions'] != 1 or report['tasks'][task]['attempted'] != 2:
            raise AssertionError('Smoke denominator/invalidity check failed')
        if report['tasks'][task]['invalid_references']:
            raise AssertionError('Synthetic reference geometry failed reconstruction')
    if report['tasks']['editing']['copied_source_rate'] != 0.5:
        raise AssertionError('Copied-source check failed')
    write_json(root / 'smoke_result.json', {'status': 'passed', 'scope': 'CAD reconstruction, failure accounting and metric plumbing; not model quality'})
