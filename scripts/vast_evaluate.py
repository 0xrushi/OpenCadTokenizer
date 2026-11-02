#!/usr/bin/env python3
"""Resumable four-GPU evaluation, followed by parallel CAD cache construction."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from cadtokenizer.bench_data import freeze_manifest, load_manifest, read_predictions
from cadtokenizer.bench_predict import benchmark_predict
from cadtokenizer.bench_runtime import evaluation_doctor, KernelRunner
from cadtokenizer.bench_score import benchmark_score
from cadtokenizer.bench_shards import merge_shards
from cadtokenizer.common import read_config, require_container, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/vastai.json')
    parser.add_argument('--benchmark-config', default='configs/benchmark-vastai.json')
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    require_container()
    config, settings = read_config(args.config), read_config(args.benchmark_config)
    root = Path(settings['output'])
    count = settings.get('inference_workers', 4)
    if args.worker is not None:
        path = root / 'shards' / f'predictions-{args.worker}.jsonl'
        benchmark_predict(config, settings, predictions_path=path,
                          resume=path.exists() or path.with_suffix('.meta.json').exists(),
                          shard_index=args.worker, shard_count=count)
        return
    evaluation_doctor()
    if not (root / 'manifest.json').exists():
        freeze_manifest(config, settings)
    if not (root / 'predictions.jsonl').exists():
        workers = []
        (root / 'shards').mkdir(parents=True, exist_ok=True)
        try:
            for index in range(count):
                log = (root / 'shards' / f'worker-{index}.log').open('a')
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(index), 'OMP_NUM_THREADS': '4'}
                command = [sys.executable, __file__, '--config', args.config,
                           '--benchmark-config', args.benchmark_config, '--worker', str(index)]
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                workers.append((process, log))
            for process, _ in workers:
                if process.wait() != 0:
                    raise RuntimeError('Prediction worker failed; inspect the per-GPU logs')
        finally:
            for process, log in workers:
                if process.poll() is None:
                    process.terminate()
            for process, log in workers:
                if process.poll() is None:
                    process.wait()
                log.close()
    merge_shards(settings, count)
    manifest = load_manifest(root / 'manifest.json')
    predictions = read_predictions(root / 'predictions.jsonl', manifest)
    runner = KernelRunner(settings)
    jobs = []
    for sample in manifest['samples']:
        key = sample['id']
        prediction = predictions[key]
        if prediction['status'] == 'syntax_valid':
            jobs.append((prediction['generated_sequence'], key, 'prediction'))
        jobs.append((sample.get('target_sequence'), key, 'reference'))
        if settings['render'] and sample['task'] == 'editing':
            jobs.append((sample['source_sequence'], key, 'source'))
    print(json.dumps({'phase': 'geometry', 'jobs': len(jobs)}), flush=True)
    with ThreadPoolExecutor(max_workers=settings.get('kernel_workers', 8)) as pool:
        for index, result in enumerate(pool.map(lambda job: runner.reconstruct(*job), jobs), 1):
            if index % 25 == 0:
                print(json.dumps({'geometry_jobs_completed': index, 'total': len(jobs)}), flush=True)
    report = benchmark_score(config, settings)
    write_json(root / 'evaluation_complete.json', {
        'status': 'complete', 'tasks': {task: value['table_metrics'] for task, value in report['tasks'].items()}})


if __name__ == '__main__':
    main()
