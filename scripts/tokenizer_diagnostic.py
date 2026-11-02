#!/usr/bin/env python3
"""Compare original CAD, direct VQ reconstruction, and aligned reconstruction."""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import subprocess
import sys

from cadtokenizer.bench_data import code_hash, content_hash, load_manifest
from cadtokenizer.common import file_hash, fraction, read_config, require_container, seed_all, write_json
from cadtokenizer.representation import NAMES, stringify, split_pairs


def freeze_cases(config, settings, benchmark, root, limit):
    signature = {'vq': file_hash(Path(config['vq']['output']) / 'best.pt'),
                 'alignment': file_hash(Path(config['alignment']['output']) / 'alignment.pt'),
                 'database': file_hash(config['data']['database']),
                 'benchmark_manifest': file_hash(benchmark / 'manifest.json'),
                 'benchmark_metrics': file_hash(benchmark / 'metrics.json'),
                 'code': code_hash(), 'diagnostic_code': file_hash(__file__),
                 'settings': settings, 'limit': limit, 'seed': config['seed']}
    path = root / 'manifest.json'
    if path.exists():
        manifest = read_config(path)
        if manifest['signature'] != signature:
            raise ValueError('Diagnostic inputs/settings/code changed; use a fresh output directory')
        return manifest
    source = load_manifest(benchmark / 'manifest.json')
    report = read_config(benchmark / 'metrics.json')
    if report['manifest_sha256'] != file_hash(benchmark / 'manifest.json') or report['evaluation_code_sha256'] != code_hash():
        raise ValueError('Source benchmark provenance does not match current inputs/code')
    cases = {r['sample_id']: r for r in map(json.loads, (benchmark / 'per_sample.jsonl').read_text().splitlines())}
    candidates, excluded = {}, collections.Counter()
    with sqlite3.connect(config['data']['database']) as db:
        for sample in source['samples']:
            if cases[sample['id']]['reference_geometry']['status'] != 'valid':
                excluded['original_not_kernel_valid'] += 1
                continue
            key = hashlib.sha256(sample['target_sequence'].encode()).hexdigest()
            row = db.execute('SELECT sequence,split,tokenizer_split FROM geometry WHERE id=?', (key,)).fetchone()
            if row is None:
                excluded['outside_prepared_representation'] += 1
                continue
            if row[1] == 'train' or row[2] == 'train':
                raise ValueError('Diagnostic candidate overlaps training')
            if row[1] != 'test':
                excluded['not_in_prepared_test_split'] += 1
                continue
            if key in candidates:
                excluded['duplicate_geometry'] += 1
                continue
            pairs = [{'tokens': json.loads(tokens), 'groups': json.loads(groups)} for tokens, groups in
                     db.execute('SELECT tokens,groups FROM pairs WHERE geometry_id=? ORDER BY ordinal', (key,))]
            if not pairs:
                raise ValueError('Prepared geometry has no pairs')
            candidates[key] = {'id': key, 'source_sample_id': sample['id'], 'task': sample['task'],
                               'sequence': row[0], 'pairs': pairs}
    ordered = sorted(candidates.values(), key=lambda c: (fraction(c['id'], config['seed']), c['id']))
    if len(ordered) < limit:
        raise ValueError(f'Only {len(ordered)} eligible unique test geometries; requested {limit}')
    manifest = {'signature': signature, 'signature_sha256': content_hash(signature),
                'selection': 'Seeded unique test geometries with known kernel-valid originals and supported representation; no model-output filtering',
                'candidate_geometries': len(ordered), 'excluded_source_cases': dict(excluded),
                'samples': ordered[:limit]}
    write_json(path, manifest)
    print(json.dumps({'selected_geometries': limit, 'candidate_geometries': len(ordered),
                      'excluded_source_cases': dict(excluded)}), flush=True)
    return manifest


def teacher_stats(vq, decoder_input, targets, memory):
    import torch
    from torch.nn import functional as F
    from cadtokenizer.representation import PAD
    logits = vq.decode(decoder_input, memory[None], torch.ones((1, len(memory)), dtype=torch.bool, device=memory.device))
    active = targets.ne(PAD)
    predicted = logits.argmax(-1)
    numeric = active & targets.ge(12) & targets.le(76)
    curve = active & targets.ge(1) & targets.le(3)
    return {'tokens': int(active.sum()), 'correct': int(((predicted == targets) & active).sum()),
            'numeric_tokens': int(numeric.sum()), 'numeric_correct': int(((predicted == targets) & numeric).sum()),
            'curve_tokens': int(curve.sum()), 'curve_correct': int(((predicted == targets) & curve).sum()),
            'ce_sum': float(F.cross_entropy(logits[active].float(), targets[active], reduction='sum'))}


def diagnose_case(vq, alignment, case, device):
    import torch
    from cadtokenizer.vq import collate_pairs, to_device
    batch = to_device(collate_pairs(case['pairs']), device)
    native = alignment['native_ids']
    inverse = {int(value): index for index, value in enumerate(native.tolist())}
    if len(inverse) != len(native):
        raise ValueError('Alignment native IDs are not injective')
    results = {mode: {'tokens': [], 'teacher': collections.Counter(), 'errors': []}
               for mode in ('direct', 'aligned')}
    with torch.inference_mode():
        _, codes, mask = vq.encode(batch['tokens'], batch['groups'])
        used, squared_error, elements = [], 0.0, 0
        for index in range(len(case['pairs'])):
            selected = codes[index][mask[index]].cpu()
            restored = torch.tensor([inverse[int(native[code])] for code in selected], dtype=torch.long)
            if not torch.equal(selected, restored):
                raise ValueError('Native-ID round trip changed code identity')
            direct = vq.codebook.weight[selected.to(device)]
            aligned = alignment['reconstructed_code_vectors'][restored].to(device)
            squared_error += float((direct - aligned).square().sum())
            elements += direct.numel()
            used.extend(selected.tolist())
            for mode, memory in [('direct', direct), ('aligned', aligned)]:
                results[mode]['teacher'].update(teacher_stats(
                    vq, batch['decoder_input'][index:index+1], batch['targets'][index:index+1], memory))
                try:
                    tokens = vq.reconstruct(memory, constrained=True)
                    split_pairs(tokens)
                    results[mode]['tokens'].extend(tokens)
                except ValueError as error:
                    results[mode]['errors'].append({'pair': index, 'error': str(error)})
    for value in results.values():
        value['sequence'] = stringify(value['tokens']) if not value['errors'] else None
        value['exact_sequence'] = value['sequence'] == case['sequence']
        value['curve_counts'] = dict(collections.Counter(NAMES[t] for t in value['tokens'] if t in (1, 2, 3)))
        del value['tokens']
    return {'id': case['id'], 'source_sample_id': case['source_sample_id'], 'pairs': len(case['pairs']),
            'used_codes': used, 'alignment_squared_error': squared_error, 'alignment_elements': elements,
            'branches': results}


def worker(config, manifest, root, index, count):
    import torch
    from cadtokenizer.common import load_checkpoint
    from cadtokenizer.vq import load_vq
    seed_all(config['seed'])
    torch.set_num_threads(2)
    vq, _ = load_vq(Path(config['vq']['output']) / 'best.pt', torch.device('cuda'))
    alignment = load_checkpoint(Path(config['alignment']['output']) / 'alignment.pt')
    if alignment['vq_sha256'] != manifest['signature']['vq']:
        raise ValueError('Alignment/VQ checkpoint mismatch')
    samples = manifest['samples'][index::count]
    path = root / f'worker-{index}.jsonl'
    existing = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['signature_sha256'] != manifest['signature_sha256'] or row['id'] in existing:
                raise ValueError('Corrupt or incompatible diagnostic journal')
            existing[row['id']] = row
    if not set(existing).issubset({s['id'] for s in samples}):
        raise ValueError('Journal contains another worker\'s cases')
    with path.open('a') as stream:
        for case in samples:
            if case['id'] in existing:
                continue
            row = diagnose_case(vq, alignment, case, torch.device('cuda'))
            row['signature_sha256'] = manifest['signature_sha256']
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            print(json.dumps({'worker': index, 'geometry': case['id']}), flush=True)


def summarize(manifest, records, geometry, settings):
    import numpy as np
    from scipy.spatial import cKDTree
    from cadtokenizer.bench_metrics import normalize_points
    count = len(records)
    original_valid = sum(geometry[(r['id'], 'original')]['status'] == 'valid' for r in records)
    summary = {'geometries': count, 'pairs': sum(r['pairs'] for r in records),
               'original_valid_solids': original_valid, 'branches': {}, 'transitions': collections.Counter()}
    for mode in ('direct', 'aligned'):
        teacher, curves = collections.Counter(), collections.Counter()
        valid, exact, syntax, distances, volume_errors = 0, 0, 0, [], []
        for row in records:
            branch = row['branches'][mode]
            teacher.update(branch['teacher']); curves.update(branch['curve_counts'])
            exact += branch['exact_sequence']; syntax += branch['sequence'] is not None
            prediction, original = geometry[(row['id'], mode)], geometry[(row['id'], 'original')]
            valid += prediction['status'] == 'valid'
            if prediction['status'] == original['status'] == 'valid':
                a = np.asarray(normalize_points(np.load(prediction['points']), settings['normalization']))
                b = np.asarray(normalize_points(np.load(original['points']), settings['normalization']))
                distances.append(float(np.mean(cKDTree(b).query(a)[0]**2) + np.mean(cKDTree(a).query(b)[0]**2)))
                volume_errors.append(abs(prediction['volume'] - original['volume']) / original['volume'])
        summary['branches'][mode] = {
            'syntax_valid': syntax, 'valid_solids': valid, 'valid_solid_rate': valid/count,
            'exact_sequences': exact, 'paired_geometry_count': len(distances),
            'mean_normalized_squared_chamfer': statistics.mean(distances) if distances else None,
            'median_normalized_squared_chamfer': statistics.median(distances) if distances else None,
            'median_relative_volume_error': statistics.median(volume_errors) if volume_errors else None,
            'teacher_forced_token_accuracy': teacher['correct']/teacher['tokens'],
            'teacher_forced_numeric_accuracy': teacher['numeric_correct']/teacher['numeric_tokens'] if teacher['numeric_tokens'] else None,
            'teacher_forced_curve_type_accuracy': teacher['curve_correct']/teacher['curve_tokens'] if teacher['curve_tokens'] else None,
            'teacher_forced_cross_entropy': teacher['ce_sum']/teacher['tokens'],
            'decoded_curve_counts': dict(curves),
            'geometry_failures': dict(collections.Counter(geometry[(r['id'], mode)]['status'] for r in records
                                                         if geometry[(r['id'], mode)]['status'] != 'valid'))}
    for row in records:
        direct = geometry[(row['id'], 'direct')]['status'] == 'valid'
        aligned = geometry[(row['id'], 'aligned')]['status'] == 'valid'
        summary['transitions'][f'direct_{"valid" if direct else "invalid"}_aligned_{"valid" if aligned else "invalid"}'] += 1
    usage = collections.Counter(code for r in records for code in r['used_codes'])
    total = sum(usage.values())
    summary['codebook'] = {'used_codes': len(usage), 'primitive_vectors': total,
                           'perplexity': math.exp(-sum((n/total)*math.log(n/total) for n in usage.values())),
                           'most_frequent_codes': usage.most_common(10)}
    summary['alignment_latent_mse'] = sum(r['alignment_squared_error'] for r in records)/sum(r['alignment_elements'] for r in records)
    summary['native_id_roundtrip_accuracy'] = 1.0
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/vastai.json')
    parser.add_argument('--benchmark', default='artifacts/benchmark-vastai-best')
    parser.add_argument('--output', default='artifacts/tokenizer-diagnostic')
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    require_container()
    if args.limit < 1 or args.workers < 1:
        parser.error('limit and workers must be positive')
    config, root = read_config(args.config), Path(args.output)
    settings = read_config('configs/benchmark-vastai.json')
    settings.update(output=str(root), render=False)
    if args.worker is not None:
        worker(config, read_config(root / 'manifest.json'), root, args.worker, args.workers)
        return
    from cadtokenizer.bench_runtime import evaluation_doctor, KernelRunner
    environment = evaluation_doctor()
    root.mkdir(parents=True, exist_ok=True)
    manifest = freeze_cases(config, settings, Path(args.benchmark), root, args.limit)
    processes = []
    try:
        for index in range(args.workers):
            log = (root / f'worker-{index}.log').open('a')
            command = [sys.executable, __file__, '--config', args.config, '--output', str(root),
                       '--workers', str(args.workers), '--worker', str(index)]
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(index), 'OMP_NUM_THREADS': '2'}
            processes.append((subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        for process, _ in processes:
            if process.wait() != 0:
                raise RuntimeError('Diagnostic worker failed; inspect worker logs')
    finally:
        for process, log in processes:
            if process.poll() is None: process.terminate()
        for process, log in processes:
            process.wait(); log.close()
    records = []
    for index in range(args.workers):
        shard = [json.loads(line) for line in (root / f'worker-{index}.jsonl').read_text().splitlines()]
        if {r['id'] for r in shard} != {s['id'] for s in manifest['samples'][index::args.workers]}:
            raise ValueError('Incomplete/incorrect diagnostic shard')
        records.extend(shard)
    if len({r['id'] for r in records}) != len(manifest['samples']):
        raise ValueError('Diagnostic case coverage mismatch')
    original = {s['id']: s['sequence'] for s in manifest['samples']}
    jobs = [(r['id'], mode, original[r['id']] if mode == 'original' else r['branches'][mode]['sequence'])
            for r in records for mode in ('original', 'direct', 'aligned')]
    runner, geometry = KernelRunner(settings), {}
    print(json.dumps({'phase': 'geometry', 'jobs': len(jobs)}), flush=True)
    with ThreadPoolExecutor(max_workers=settings.get('kernel_workers', 16)) as pool:
        for index, (job, result) in enumerate(zip(jobs, pool.map(lambda j: runner.reconstruct(j[2], j[0], j[1]), jobs)), 1):
            geometry[(job[0], job[1])] = result
            if index % 25 == 0: print(json.dumps({'geometry_checks': index, 'total': len(jobs)}), flush=True)
    report = {'schema': 'cad-tokenizer-diagnostic-v1', 'signature': manifest['signature'],
              'selection': manifest['selection'], 'environment': environment,
              'summary': summarize(manifest, records, geometry, settings),
              'limits': ['Conditional diagnostic on known-valid, representable test geometry; not the full benchmark.',
                         'No LLM is loaded. Both decodes use the same FSA and original pair boundaries.',
                         'Chamfer uses independent surface samples and per-shape max-absolute normalization; volume error also reported.']}
    write_json(root / 'report.json', report)
    write_json(root / 'per_case.json', [{**r, 'geometry': {mode: geometry[(r['id'], mode)]
                                                       for mode in ('original', 'direct', 'aligned')}} for r in records])
    s = report['summary']
    lines = ['# Tokenizer/alignment reconstruction diagnostic', '',
             manifest['selection'], '', f'Original valid solids: {s["original_valid_solids"]}/{s["geometries"]}.', '',
             '| Path | Syntax-valid | Valid solids | Teacher-forced token accuracy | Mean normalized squared CD |',
             '| --- | ---: | ---: | ---: | ---: |']
    for mode, b in s['branches'].items():
        lines.append(f'| {mode} | {b["syntax_valid"]}/{s["geometries"]} | {b["valid_solids"]}/{s["geometries"]} | {b["teacher_forced_token_accuracy"]:.4f} | {b["mean_normalized_squared_chamfer"]} |')
    lines += ['', f'Native-ID round-trip accuracy: {s["native_id_roundtrip_accuracy"]:.1%}.',
              f'Alignment latent MSE: {s["alignment_latent_mse"]:.6f}.',
              f'Used codebook entries: {s["codebook"]["used_codes"]}.', '', *report['limits']]
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    write_json(root / 'complete.json', {'status': 'complete', 'summary': s})
    print(json.dumps(s, indent=2), flush=True)


if __name__ == '__main__':
    main()
