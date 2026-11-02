"""Sensitivity of the invalid rate to trivial geometry cleanup; diagnostic only, never a benchmark score."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path

from .bench_data import load_manifest, read_predictions
from .bench_runtime import KernelRunner
from .common import write_json

AUDIT_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'audit_codebook_failures.py'


def probe_variant():
    spec = importlib.util.spec_from_file_location('audit_codebook_failures', AUDIT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.probe_variant


def lenient_rescore(settings, manifest_path=None, workers=6):
    root = Path(settings['output'])
    manifest = load_manifest(Path(manifest_path or root / 'manifest.json'))
    predictions = read_predictions(root / 'predictions.jsonl', manifest)
    scored = {}
    for line in (root / 'per_sample.jsonl').read_text().splitlines():
        row = json.loads(line)
        scored[row['sample_id']] = row
    tasks = {sample['id']: sample['task'] for sample in manifest['samples']}
    probe = probe_variant()
    runner = KernelRunner(settings)
    kinds = ('degenerate_curves', 'combined')
    jobs = []
    for key, row in scored.items():
        prediction = predictions[key]
        if prediction['status'] != 'syntax_valid' or row['prediction_geometry'].get('status') == 'valid':
            continue
        for kind in kinds:
            variant, changes = probe(prediction['generated_sequence'], kind)
            if variant is not None:
                jobs.append((key, kind, variant, changes))
    def run(job):
        key, kind, variant, changes = job
        return key, kind, runner.reconstruct(variant, key, f'lenient-{kind}')['status'] == 'valid', [c['field'] for c in changes]
    recovered = {kind: set() for kind in kinds}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (key, kind, valid, _) in enumerate(pool.map(run, jobs), 1):
            if valid:
                recovered[kind].add(key)
            if index % 25 == 0:
                print(json.dumps({'lenient_jobs_completed': index, 'total': len(jobs)}), flush=True)
    report = {'note': 'diagnostic only: cleanup can change design intent and is not part of any benchmark score', 'tasks': {}}
    for task in sorted(set(tasks.values())):
        ids = [key for key in scored if tasks[key] == task]
        valid = sum(scored[key]['prediction_geometry'].get('status') == 'valid' for key in ids)
        entry = {'cases': len(ids), 'valid': valid, 'invalid_rate_percent': round(100 * (len(ids) - valid) / len(ids), 2)}
        for kind in kinds:
            after = valid + sum(tasks[key] == task for key in recovered[kind])
            entry[kind] = {'valid': after, 'invalid_rate_percent': round(100 * (len(ids) - after) / len(ids), 2)}
        report['tasks'][task] = entry
    write_json(root / 'lenient-sensitivity.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report
