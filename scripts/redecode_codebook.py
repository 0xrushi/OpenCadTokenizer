#!/usr/bin/env python3
"""Controlled exact-codebook redecoding of existing, fingerprinted LLM outputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from cadtokenizer.bench_data import code_hash, content_hash, load_manifest, read_predictions
from cadtokenizer.bench_predict import model_fingerprints
from cadtokenizer.bench_runtime import KernelRunner, evaluation_doctor
from cadtokenizer.bench_score import benchmark_score
from cadtokenizer.bench_shards import merge_shards
from cadtokenizer.common import file_hash, load_checkpoint, read_config, require_container, write_json


class VerifiedReferenceCache:
    """Reuse only baseline reference/source geometry under verified identical code."""
    def __init__(self, settings, baseline, manifest, snapshot):
        self.runner = KernelRunner(settings)
        report = read_config(baseline / 'metrics.json')
        old_settings = report['settings']
        keys = ['seed', 'point_count', 'grid_resolution', 'normalization', 'kernel_backend',
                'kernel_timeout_seconds', 'mesh_tolerance', 'mesh_angular_tolerance',
                'boolean_tolerance', 'minimum_volume', 'render', 'render_axis_limit', 'render_max_triangles']
        if any(settings[key] != old_settings[key] for key in keys):
            raise ValueError('Geometry settings differ from baseline')
        if snapshot['packages'] != self.runner.version['packages']:
            raise ValueError('Geometry package versions differ from baseline')
        if any(file_hash(Path('cadtokenizer') / name) != digest
               for name, digest in snapshot['geometry_components'].items()):
            raise ValueError('Geometry implementation differs from verified baseline')
        if file_hash(baseline / 'per_sample.jsonl') != snapshot['baseline_per_sample_sha256']:
            raise ValueError('Baseline geometry report changed')
        self.cases = {r['sample_id']: r for r in map(json.loads, (baseline / 'per_sample.jsonl').read_text().splitlines())}
        self.samples = {s['id']: s for s in manifest['samples']}
        self.provenance = {'reference_source': str(baseline), 'verification': snapshot,
                           'note': 'Reference/source CAD reused under matching geometry code, versions, settings, and identical input manifest.'}

    def reconstruct(self, sequence, sample_id, role):
        if role == 'prediction':
            return self.runner.reconstruct(sequence, sample_id, role)
        field = {'reference': 'target_sequence', 'source': 'source_sequence'}[role]
        if sequence != self.samples[sample_id].get(field):
            raise ValueError('Reference/source sequence changed')
        result = self.cases[sample_id][{'reference': 'reference_geometry', 'source': 'source_geometry'}[role]]
        if result['status'] == 'valid' and not all(Path(result[key]).is_file() for key in ('points', 'step', 'stl')):
            raise FileNotFoundError('Verified baseline geometry artifacts are missing')
        return result


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
            if key in existing: continue
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
            stream.flush(); os.fsync(stream.fileno())
            print(json.dumps({'worker': index, 'sample_id': key, 'status': row['status']}), flush=True)


def comparison(report, baseline, root, source, replay):
    old_report = read_config(baseline / 'metrics.json')
    old_cases = {r['sample_id']: r for r in map(json.loads, (baseline / 'per_sample.jsonl').read_text().splitlines())}
    cases = [json.loads(line) for line in (root / 'per_sample.jsonl').read_text().splitlines()]
    if set(replay) != set(source) or any(replay[key].get('generated_native_ids') != old.get('generated_native_ids') for key, old in source.items()):
        raise ValueError('Replay changed native IDs or case coverage')
    result = {'same_native_ids_verified': True, 'cases': len(cases),
              'inherited_baseline_failures': sum(r['status'] != 'syntax_valid' for r in source.values()),
              'tasks': {}, 'note': 'Controlled decoder-only comparison. Same test cases, native IDs, and reference point clouds. No LLM rerun or retraining.'}
    for task, new in report['tasks'].items():
        old = old_report['tasks'][task]
        selected = [r for r in cases if r['task'] == task]
        shared = [r for r in selected if r.get('paired_cd_raw') is not None and old_cases[r['sample_id']].get('paired_cd_raw') is not None]
        result['tasks'][task] = {
            'cases': new['attempted'], 'baseline_valid_solids': old['attempted'] - old['invalid_predictions'],
            'codebook_valid_solids': new['attempted'] - new['invalid_predictions'],
            'became_valid': sum(r['prediction_geometry']['status']=='valid' and old_cases[r['sample_id']]['prediction_geometry']['status']!='valid' for r in selected),
            'became_invalid': sum(r['prediction_geometry']['status']!='valid' and old_cases[r['sample_id']]['prediction_geometry']['status']=='valid' for r in selected),
            'baseline_metrics': old['table_metrics'], 'codebook_metrics': new['table_metrics'],
            'common_valid_pairs': len(shared),
            'common_set_mean_cd_before': statistics.mean(old_cases[r['sample_id']]['paired_cd_raw'] for r in shared) if shared else None,
            'common_set_mean_cd_after': statistics.mean(r['paired_cd_raw'] for r in shared) if shared else None}
    write_json(root / 'decoder_comparison.json', result)
    lines = ['# Exact codebook decoder comparison', '', result['note'], '',
             '| Task | Cases | Baseline valid solids | Exact codebook valid solids | Became valid | Became invalid |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for task, value in result['tasks'].items():
        n = value['cases']
        lines.append(f'| {task} | {n} | {value["baseline_valid_solids"]} ({value["baseline_valid_solids"]/n:.1%}) | {value["codebook_valid_solids"]} ({value["codebook_valid_solids"]/n:.1%}) | {value["became_valid"]} | {value["became_invalid"]} |')
    lines += ['', 'This is an independent benchmark, not the original paper test set. Validity alone does not prove instruction following.']
    (root / 'decoder_comparison.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/vastai.json')
    parser.add_argument('--benchmark-config', default='configs/benchmark-vastai-codebook.json')
    parser.add_argument('--baseline', default='artifacts/benchmark-vastai-best')
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    require_container()
    config, settings = read_config(args.config), read_config(args.benchmark_config)
    config.setdefault('inference', {})['cad_decoder'] = 'codebook'
    root, baseline = Path(settings['output']), Path(args.baseline)
    if root.resolve() == baseline.resolve(): raise ValueError('Use a distinct output directory')
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        if file_hash(manifest_path) != file_hash(baseline / 'manifest.json'): raise ValueError('Frozen manifest differs')
    else: shutil.copyfile(baseline / 'manifest.json', manifest_path)
    manifest = load_manifest(manifest_path)
    manifest_hash = file_hash(manifest_path)
    old_meta = read_config(baseline / 'predictions.meta.json')
    snapshot = read_config('.runtime/baseline-geometry-provenance.json')
    if (old_meta['signature_sha256'] != content_hash(old_meta['signature'])
            or old_meta['signature']['code_sha256'] != snapshot['baseline_code_sha256']
            or file_hash(baseline / 'predictions.jsonl') != snapshot['baseline_predictions_sha256']
            or old_meta['signature']['manifest_sha256'] != manifest_hash):
        raise ValueError('Baseline prediction provenance mismatch')
    if model_fingerprints(config) != old_meta['signature']['model_files']:
        raise ValueError('Model weights changed since baseline inference')
    source = read_predictions(baseline / 'predictions.jsonl', manifest)
    if len(source) != len(manifest['samples']) or any(r['run_signature_sha256'] != old_meta['signature_sha256'] or r['manifest_sha256'] != manifest_hash for r in source.values()):
        raise ValueError('Baseline rows are incomplete or inconsistent')
    signature = {**old_meta['signature'], 'model_config': config, 'code_sha256': code_hash(),
                 'cad_decoder': 'codebook', 'decoding': 'reused_native_ids_exact_codebook_vq_fsa',
                 'source_predictions_sha256': snapshot['baseline_predictions_sha256'],
                 'source_signature_sha256': old_meta['signature_sha256'],
                 'replay_script_sha256': file_hash(__file__)}
    metadata = {'schema': 'cad-predictions-v1', 'signature': signature, 'signature_sha256': content_hash(signature)}
    count = settings.get('inference_workers', 4)
    if args.worker is not None:
        replay_worker(config, manifest, metadata, source, root, args.worker, count)
        return
    environment = evaluation_doctor()
    runner = VerifiedReferenceCache(settings, baseline, manifest, snapshot)
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
            if process.wait() != 0: raise RuntimeError('Codebook replay worker failed; inspect worker logs')
    finally:
        for process, log in processes:
            if process.poll() is None: process.terminate()
        for process, log in processes:
            process.wait(); log.close()
    merge_shards(settings, count)
    replay = read_predictions(root / 'predictions.jsonl', manifest)
    jobs = [(r['generated_sequence'], key, 'prediction') for key, r in replay.items() if r['status']=='syntax_valid']
    print(json.dumps({'phase': 'geometry', 'new_prediction_checks': len(jobs), 'reference_source_checks': 'verified baseline cache'}), flush=True)
    with ThreadPoolExecutor(max_workers=settings.get('kernel_workers',16)) as pool:
        for index, _ in enumerate(pool.map(lambda job: runner.reconstruct(*job), jobs), 1):
            if index % 25 == 0: print(json.dumps({'new_geometry_checks': index,'total':len(jobs)}),flush=True)
    report = benchmark_score(config, settings, runner_override=runner)
    result = comparison(report, baseline, root, source, replay)
    write_json(root / 'evaluation_complete.json', {'status':'complete','comparison':result})
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
