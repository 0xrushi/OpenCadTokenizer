#!/usr/bin/env python3
"""Redecode saved native IDs with current VQ-decoder code, scored fresh.

Unlike scripts/redecode_codebook.py, this does not require the VQ-decoder's
grammar/representation code to match a prior baseline's recorded hash: it is
for comparisons where that grammar itself changed (e.g. added numeric decode
guards), so reference and source geometry are rebuilt fresh through the normal
CAD kernel instead of reused from a cache keyed to unchanged code. It still
reuses the saved native token IDs, so no LLM is rerun and no retraining occurs.
"""
import argparse
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time

from concurrent.futures import ThreadPoolExecutor

from cadtokenizer.bench_data import code_hash, content_hash, load_manifest, read_predictions
from cadtokenizer.bench_score import benchmark_score
from cadtokenizer.bench_shards import merge_shards
from cadtokenizer.common import file_hash, load_checkpoint, read_config, require_container, write_json


def replay_worker(config, manifest, metadata, source, root, index, count):
    import torch
    from cadtokenizer.inference import decode_native_ids
    from cadtokenizer.vq import load_vq
    torch.set_num_threads(2)
    vq, _ = load_vq(Path(config['vq']['output']) / 'best.pt', torch.device('cuda'))
    alignment = load_checkpoint(Path(config['alignment']['output']) / 'alignment.pt')
    if alignment['vq_sha256'] != file_hash(Path(config['vq']['output']) / 'best.pt'):
        raise ValueError('Alignment/VQ mismatch')
    path = root / 'shards' / f'predictions-{index}.jsonl'
    meta_path = path.with_suffix('.meta.json')
    if meta_path.exists() and read_config(meta_path)['signature'] != metadata['signature']:
        raise ValueError('Existing replay journal has another signature')
    write_json(meta_path, metadata)
    existing = read_predictions(path, manifest)
    expected = {s['id'] for s in manifest['samples'][index::count]}
    if not set(existing).issubset(expected) or any(r.get('run_signature_sha256') != metadata['signature_sha256'] for r in existing.values()):
        raise ValueError('Incorrect replay journal partition or signature')
    with path.open('a') as stream:
        for sample in manifest['samples'][index::count]:
            key = sample['id']
            if key in existing:
                continue
            old = source[key]
            started = time.monotonic()
            native = old.get('generated_native_ids')
            row = {'sample_id': key, 'generated_native_ids': native,
                   'source_status': old['status'], 'source_elapsed_seconds': old['elapsed_seconds'],
                   'manifest_sha256': metadata['signature']['manifest_sha256'],
                   'run_signature_sha256': metadata['signature_sha256']}
            if old['status'] != 'syntax_valid':
                row.update(status='inference_error', generated_sequence=None,
                           error=old.get('error', 'Baseline inference failed'), error_type=old.get('error_type'))
            else:
                if not isinstance(native, list) or not native:
                    raise ValueError('Baseline did not preserve generated native IDs')
                try:
                    row.update(status='syntax_valid', generated_sequence=decode_native_ids(vq, alignment, native, 'codebook'))
                except (ValueError, KeyError) as error:
                    row.update(status='inference_error', generated_sequence=None,
                               error=str(error), error_type=type(error).__name__)
            row['elapsed_seconds'] = time.monotonic() - started
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            print(json.dumps({'worker': index, 'sample_id': key, 'status': row['status']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/vastai.json')
    parser.add_argument('--benchmark-config', default='configs/benchmark-vastai-codebook-v2.json')
    parser.add_argument('--baseline', default='artifacts/benchmark-vastai-best')
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    require_container()
    config, settings = read_config(args.config), read_config(args.benchmark_config)
    config.setdefault('inference', {})['cad_decoder'] = 'codebook'
    root, baseline = Path(settings['output']), Path(args.baseline)
    if root.resolve() == baseline.resolve():
        raise ValueError('Use a distinct output directory')
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        if file_hash(manifest_path) != file_hash(baseline / 'manifest.json'):
            raise ValueError('Frozen manifest differs')
    else:
        shutil.copyfile(baseline / 'manifest.json', manifest_path)
    manifest = load_manifest(manifest_path)
    manifest_hash = file_hash(manifest_path)
    old_meta = read_config(baseline / 'predictions.meta.json')
    source = read_predictions(baseline / 'predictions.jsonl', manifest)
    if len(source) != len(manifest['samples']):
        raise ValueError('Baseline predictions are incomplete')
    signature = {**old_meta['signature'], 'model_config': config, 'code_sha256': code_hash(),
                 'cad_decoder': 'codebook', 'decoding': 'reused_native_ids_updated_vq_fsa_fresh_geometry',
                 'source_predictions_sha256': file_hash(baseline / 'predictions.jsonl'),
                 'source_signature_sha256': old_meta['signature_sha256'],
                 'replay_script_sha256': file_hash(__file__), 'manifest_sha256': manifest_hash}
    metadata = {'schema': 'cad-predictions-v1', 'signature': signature, 'signature_sha256': content_hash(signature)}
    count = settings.get('inference_workers', 4)
    if args.worker is not None:
        replay_worker(config, manifest, metadata, source, root, args.worker, count)
        return
    (root / 'shards').mkdir(exist_ok=True)
    processes = []
    try:
        for index in range(count):
            log = (root / 'shards' / f'worker-{index}.log').open('a')
            command = [sys.executable, __file__, '--config', args.config, '--benchmark-config', args.benchmark_config,
                       '--baseline', str(baseline), '--worker', str(index)]
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(index), 'OMP_NUM_THREADS': '2'}
            processes.append((subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        for process, _ in processes:
            if process.wait() != 0:
                raise RuntimeError('Replay worker failed; inspect worker logs')
    finally:
        for process, log in processes:
            if process.poll() is None:
                process.terminate()
        for process, log in processes:
            process.wait()
            log.close()
    merge_shards(settings, count)
    print(json.dumps({'phase': 'redecoded', 'cases': len(manifest['samples'])}), flush=True)
    from cadtokenizer.bench_runtime import KernelRunner
    runner = KernelRunner(settings)
    replay = read_predictions(root / 'predictions.jsonl', manifest)
    jobs = []
    for sample in manifest['samples']:
        key = sample['id']
        prediction = replay[key]
        if prediction['status'] == 'syntax_valid':
            jobs.append((prediction['generated_sequence'], key, 'prediction'))
        jobs.append((sample.get('target_sequence'), key, 'reference'))
        if settings['render'] and sample['task'] == 'editing':
            jobs.append((sample['source_sequence'], key, 'source'))
    print(json.dumps({'phase': 'warming_geometry_cache', 'jobs': len(jobs)}), flush=True)
    with ThreadPoolExecutor(max_workers=settings.get('kernel_workers', 16)) as pool:
        for index, _ in enumerate(pool.map(lambda job: runner.reconstruct(*job), jobs), 1):
            if index % 50 == 0:
                print(json.dumps({'warmed_geometry_checks': index, 'total': len(jobs)}), flush=True)
    report = benchmark_score(config, settings, runner_override=runner)
    print(json.dumps({'phase': 'scored', 'tasks': {k: v['table_metrics'] for k, v in report['tasks'].items()}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
