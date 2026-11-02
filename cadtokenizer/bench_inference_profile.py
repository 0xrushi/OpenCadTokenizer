"""Profile and validate batched benchmark decoding against the single-prompt reference."""
from __future__ import annotations

from pathlib import Path
import json
import time

from .bench_data import load_manifest
from .common import fraction, require_container, seed_all, write_json


def representative(manifest, count, seed):
    """Seeded, task-proportional subset (no output-based selection)."""
    by_task = {}
    for sample in manifest['samples']:
        if not sample.get('instruction_error'):
            by_task.setdefault(sample['task'], []).append(sample)
    total = sum(len(v) for v in by_task.values())
    chosen = []
    for task, rows in sorted(by_task.items()):
        rows = sorted(rows, key=lambda r: (fraction(r['id'], seed), r['id']))
        chosen += rows[:max(1, round(count * len(rows) / total))]
    return chosen[:count]


def outcome(result):
    if isinstance(result, Exception):
        return {'error': type(result).__name__ + ': ' + str(result)}
    return {'native': result['generated_native_ids'], 'sequence': result['generated_sequence']}


def run_inference_profile(config, settings, manifest_path, sizes, count):
    require_container()
    import torch
    from .inference import CadGenerator
    seed_all(settings['seed'])
    manifest = load_manifest(manifest_path or Path(settings['output']) / 'manifest.json')
    samples = representative(manifest, count, settings['seed'])
    requests = [(s['instruction'], s.get('source_sequence')) for s in samples]
    generator = CadGenerator(config)
    limit = settings['max_new_tokens']

    def timed(function):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        value = function()
        torch.cuda.synchronize()
        return value, time.perf_counter() - start, torch.cuda.max_memory_allocated()

    def reference_one(request):
        try:
            return generator.predict(request[0], request[1], limit)
        except (ValueError, KeyError) as error:
            return error
    generator.predict_batch(requests[:2], limit, 2)       # warm up kernels outside the timings
    reference, seconds, peak = timed(lambda: [reference_one(r) for r in requests])
    reference_outcomes = [outcome(r) for r in reference]
    rows = [{'batch_size': 1, 'path': 'single-prompt reference', 'seconds': seconds, 'samples_per_second': len(requests) / seconds,
             'peak_cuda_gib': peak / 2**30, 'identical_native_ids': 1.0, 'identical_sequences': 1.0}]
    print(json.dumps(rows[-1]), flush=True)
    for size in sizes:
        try:
            batched, seconds, peak = timed(lambda: generator.predict_batch(requests, limit, size))
        except RuntimeError as error:
            if 'out of memory' not in str(error).lower():
                raise
            rows.append({'batch_size': size, 'error': 'CUDA out of memory'})
            torch.cuda.empty_cache()
            print(json.dumps(rows[-1]), flush=True)
            continue
        outcomes = [outcome(r) for r in batched]
        same = lambda key: sum(a.get(key) == b.get(key) and ('error' in a) == ('error' in b) for a, b in zip(reference_outcomes, outcomes)) / len(outcomes)
        rows.append({'batch_size': size, 'path': 'batched', 'seconds': seconds, 'samples_per_second': len(requests) / seconds,
                     'peak_cuda_gib': peak / 2**30, 'identical_native_ids': same('native'), 'identical_sequences': same('sequence'),
                     'identical_error_status': sum(('error' in a) == ('error' in b) for a, b in zip(reference_outcomes, outcomes)) / len(outcomes)})
        print(json.dumps(rows[-1]), flush=True)
    report = {'samples': len(requests), 'tasks': {t: sum(s['task'] == t for s in samples) for t in {s['task'] for s in samples}},
              'gpu': torch.cuda.get_device_name(0), 'total_gib': torch.cuda.get_device_properties(0).total_memory / 2**30,
              'max_new_tokens': limit, 'rows': rows,
              'note': 'Reference is unpadded single-prompt greedy decoding; batched rows are left-padded, length-sorted greedy decoding with the same grammar and decoder.'}
    write_json(Path(settings['output']) / 'inference-profile.json', report)
    print(json.dumps({'written': str(Path(settings['output']) / 'inference-profile.json')}), flush=True)
