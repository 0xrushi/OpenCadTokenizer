"""Local, paired tokenizer ablations. Selection never consults the test benchmark."""
from __future__ import annotations

import collections
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
import os
from pathlib import Path
import pickle
import sqlite3
import statistics
import time

from .common import file_hash, fraction, read_config, require_container, seed_all, write_json
from .representation import analyze, primitive_ids, skexgen_to_sem, split_pairs, stringify, tokenize

SCHEMA = 'tokenizer-local-v1'


def features(sequence):
    pairs = split_pairs(tokenize(sequence))
    counts = collections.Counter(t for pair in pairs for t in pair)
    return {'pairs': len(pairs), 'curves': sum(counts[t] for t in (1, 2, 3)),
            'arcs': counts[2], 'circles': counts[3], 'loops': counts[5],
            'faces': counts[6], 'holes': counts[5] - counts[6],
            'max_curve_primitives': max(max(primitive_ids(p)) + 1 for p in pairs),
            'max_loop_primitives': max(max(primitive_ids(p, sketch_partition='loop')) + 1 for p in pairs)}


def select_cases(rows, limit, seed):
    """Hash-seeded uniform selection; distribution is independent of model outputs."""
    ordered = sorted(rows, key=lambda r: (fraction(r['id'], seed), r['id']))
    if len(ordered) < limit:
        raise ValueError(f'Need {limit} geometries, found {len(ordered)}')
    return ordered[:limit]


def diverse_subset(rows, limit):
    """Include training-only arc/hole/multi-loop/multi-operation examples in overfit probe."""
    chosen = {}
    predicates = [lambda f: f['arcs'] > 0, lambda f: f['holes'] > 0,
                  lambda f: f['loops'] > 1, lambda f: f['pairs'] > 1,
                  lambda f: f['circles'] > 0, lambda f: f['max_curve_primitives'] >= 10]
    for predicate in predicates:
        for row in rows:
            if predicate(row['features']):
                chosen[row['id']] = row
                break
    for row in rows:
        if len(chosen) >= limit:
            break
        chosen[row['id']] = row
    if len(chosen) > limit:
        raise ValueError('Overfit size too small for required diversity')
    return list(chosen.values())


def code_signature():
    names = ('tokenizer_experiment.py', 'representation.py', 'vq.py', 'geometry_worker.py',
             'bench_runtime.py', 'bench_metrics.py')
    return {name: file_hash(Path('cadtokenizer') / name) for name in names}


def make_database(path, train, val, data):
    """Repool identical geometry IDs without recomputing split components."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.unlink(missing_ok=True)
    with sqlite3.connect(temporary) as db:
        db.executescript('''
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE geometry(id TEXT PRIMARY KEY,sequence TEXT,split TEXT,
                                  tokenizer_candidate INTEGER,tokenizer_split TEXT,component TEXT);
            CREATE TABLE pairs(id INTEGER PRIMARY KEY,geometry_id TEXT,ordinal INTEGER,tokens TEXT,groups TEXT);
            CREATE INDEX pairs_geometry ON pairs(geometry_id,ordinal);
        ''')
        db.execute('INSERT INTO metadata VALUES(?,?)', ('data_config', json.dumps(data)))
        db.execute('INSERT INTO metadata VALUES(?,?)', ('schema_version', json.dumps(SCHEMA)))
        for split, rows in [('train', train), ('val', val)]:
            for row in rows:
                canonical, pairs = analyze(row['sequence'], data['max_pair_tokens'], data['max_primitives'],
                                           data['extrusion_partition'], data['sketch_partition'])
                db.execute('INSERT INTO geometry VALUES(?,?,?,?,?,?)',
                           (row['id'], canonical, split, 1, split, row['component']))
                db.executemany('INSERT INTO pairs(geometry_id,ordinal,tokens,groups) VALUES(?,?,?,?)',
                               [(row['id'], i, json.dumps(p['tokens']), json.dumps(p['groups'])) for i, p in enumerate(pairs)])
    temporary.replace(path)


def raw_coverage(config):
    """Coverage of released SkexGen records, including ones excluded by prepared DB."""
    root = Path(config['data']['root']) / 'skexgen/cad_data'
    report = {}
    with (root / 'train_invalid.pkl').open('rb') as stream:
        invalid = {str(v) for v in pickle.load(stream)}
    for split, name in [('train', 'train_deduplicate_s.pkl'), ('val', 'val.pkl'), ('test', 'test.pkl')]:
        with (root / name).open('rb') as stream:
            records = pickle.load(stream)
        counts, histogram = collections.Counter(), collections.Counter()
        for record in records:
            counts['records'] += 1
            if str(record['name']) in invalid:
                counts['upstream_invalid'] += 1
                continue
            try:
                pairs = split_pairs(tokenize(skexgen_to_sem(record)))
                if any(len(p) > config['data']['max_pair_tokens'] for p in pairs):
                    counts['over_pair_token_budget'] += 1
                    continue
                for mode in ('curve', 'loop'):
                    maximum = max(max(primitive_ids(p, config['data']['extrusion_partition'], mode)) + 1 for p in pairs)
                    histogram[f'{mode}:{maximum}'] += 1
                    counts[f'{mode}_supported'] += maximum <= config['data']['max_primitives']
                counts['well_formed_within_token_budget'] += 1
            except (ValueError, KeyError, IndexError, TypeError):
                counts['conversion_or_syntax_invalid'] += 1
        report[split] = {'counts': dict(counts), 'maximum_primitives_histogram': dict(histogram)}
    return {'scope': 'Released SkexGen records before grouped splitting; record counts, not deduplicated geometry. Test counts are descriptive coverage only.',
            'splits': report}


def prepare_experiment(config, settings):
    root = Path(settings['output'])
    root.mkdir(parents=True, exist_ok=True)
    raw = Path(config['data']['root']) / 'skexgen/cad_data'
    signature = {'schema': SCHEMA, 'source_database_sha256': file_hash(config['data']['database']),
                 'raw_source_sha256': {name: file_hash(raw / name) for name in (
                     'train_invalid.pkl', 'train_deduplicate_s.pkl', 'val.pkl', 'test.pkl')},
                 'image_id': os.environ.get('CADTOKENIZER_IMAGE_ID'),
                 'base_config': config, 'settings': settings, 'code': code_signature()}
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        manifest = read_config(manifest_path)
        if manifest['signature'] != signature:
            raise ValueError('Experiment inputs/code changed; choose a fresh output directory')
        for variant in manifest['variants']:
            for phase in ('overfit', 'screen'):
                spec = variant[phase]
                if file_hash(spec['config_path']) != spec['config_sha256'] or file_hash(spec['database']) != spec['database_sha256']:
                    raise ValueError('Experiment config/database changed')
        return manifest
    with sqlite3.connect(config['data']['database']) as db:
        candidates = {split: [] for split in ('train', 'val')}
        for key, sequence, split, component in db.execute(
                'SELECT id,sequence,tokenizer_split,component FROM geometry WHERE tokenizer_candidate=1 AND tokenizer_split IN ("train","val")'):
            candidates[split].append({'id': key, 'sequence': sequence, 'component': component})
    train = select_cases(candidates['train'], settings['train_geometries'], settings['seed'])
    val = select_cases(candidates['val'], settings['validation_geometries'], settings['seed'])
    for row in train + val:
        row['features'] = features(row['sequence'])
    if {r['component'] for r in train} & {r['component'] for r in val}:
        raise ValueError('Train/validation component overlap')
    overfit = diverse_subset(train, settings['overfit_geometries'])
    coverage = raw_coverage(config)
    variants = []
    for mode in ('curve', 'loop'):
        for loss in ('emd', 'cross_entropy'):
            name = f'{mode}-{loss}'
            variant = {'name': name, 'sketch_partition': mode, 'loss': loss}
            for phase, selected in [('overfit', overfit), ('screen', train)]:
                cfg = copy.deepcopy(config)
                directory = root / name / phase
                cfg['seed'] = settings['seed']
                cfg['data'].update(database=str(directory / 'cad.sqlite'), sketch_partition=mode)
                cfg['vq'].update(output=str(directory / 'vq'), reconstruction_loss=loss,
                                 batch_size=settings['microbatch_size'],
                                 gradient_accumulation_steps=settings['gradient_accumulation_steps'],
                                 num_workers=0, epochs=settings[f'{phase}_epochs'],
                                 patience=settings[f'{phase}_epochs'])
                cfg['vq']['overfit_validation'] = phase == 'overfit'
                make_database(cfg['data']['database'], selected, val if phase == 'screen' else [], cfg['data'])
                config_path = directory / 'config.json'
                write_json(config_path, cfg)
                variant[phase] = {'config_path': str(config_path), 'config_sha256': file_hash(config_path),
                                  'database': cfg['data']['database'], 'database_sha256': file_hash(cfg['data']['database'])}
            variants.append(variant)
    manifest = {'signature': signature, 'train': train, 'validation': val, 'overfit': overfit,
                'variants': variants, 'coverage': coverage,
                'selection': 'Hash-seeded prepared tokenizer train/val; fixed IDs and components, common supported subset. No benchmark/test outputs used.',
                'limits': ['Prepared candidates were filtered by the original curve representation. Raw coverage audit describes additional capacity but does not add those cases to this controlled comparison.',
                           'Overfit validation deliberately reuses the small training set and is not a generalization score.',
                           'Screening uses a limited training subset/budget and cannot establish full-data convergence.']}
    write_json(manifest_path, manifest)
    return manifest


def field_stats(logits, targets, tokens):
    """Teacher-forced field accuracy, with per-field rather than pooled numeric counts."""
    from .representation import PAD
    target = targets[0].tolist()
    predicted = logits[0].argmax(-1).tolist()
    boundary = next(i for i, t in enumerate(tokens) if t in (8, 9, 10))
    labels = ['sketch'] * len(target)
    fields = ['boolean', 'lower', 'upper', 'tx', 'ty', 'tz',
              'xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz', 'scale', 'ox', 'oy', 'extrusion_end']
    labels[boundary:boundary+len(fields)] = fields
    result = {}
    for index, actual in enumerate(target):
        if actual == PAD:
            continue
        for label in ('all', labels[index], 'numeric' if 12 <= actual <= 76 else 'structural'):
            item = result.setdefault(label, {'correct': 0, 'count': 0})
            item['count'] += 1
            item['correct'] += predicted[index] == actual
    return result


def merge_fields(rows):
    result = {}
    for row in rows:
        for key, value in row.items():
            item = result.setdefault(key, {'correct': 0, 'count': 0})
            for name in item:
                item[name] += value[name]
    return {key: {**value, 'accuracy': value['correct'] / value['count']} for key, value in result.items()}


def evaluate_checkpoint(checkpoint_path, cases, settings, output):
    """Ground-truth codes to free-running VQ decoder, with isolated kernel workers."""
    require_container()
    import torch
    import numpy as np
    from scipy.spatial import cKDTree
    from .bench_metrics import normalize_points
    from .bench_runtime import KernelRunner, evaluation_doctor
    from .vq import collate_pairs, load_vq, to_device
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    signature = {'checkpoint_sha256': file_hash(checkpoint_path), 'case_ids': [r['id'] for r in cases],
                 'cases': cases, 'settings': settings, 'code': code_signature(),
                 'image_id': os.environ.get('CADTOKENIZER_IMAGE_ID')}
    if (root / 'report.json').exists():
        report = read_config(root / 'report.json')
        if report['signature'] != signature:
            raise ValueError('Reconstruction report provenance changed; use a fresh output')
        return report
    environment = evaluation_doctor()
    seed_all(settings['seed'])
    torch.set_num_threads(2)
    device = torch.device('cuda')
    vq, checkpoint = load_vq(checkpoint_path, device)
    vq.eval()
    data = vq.config['data']
    torch.cuda.reset_peak_memory_stats()
    decode_start = time.monotonic()
    records, usage = [], collections.Counter()
    journal = root / 'decoded.jsonl'
    from .bench_data import content_hash
    signature_hash = content_hash(signature)
    existing = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            item = json.loads(line)
            if item.pop('signature_sha256') != signature_hash or item['id'] in existing:
                raise ValueError('Incompatible reconstruction journal')
            existing[item['id']] = item
        if not set(existing) <= set(signature['case_ids']):
            raise ValueError('Unexpected reconstruction cases')
    with journal.open('a') as stream, torch.inference_mode():
        for index, case in enumerate(cases):
            if case['id'] in existing:
                row = existing[case['id']]
            else:
                row = {'id': case['id'], 'features': case['features'], 'sequence': None, 'error': None,
                       'fields': {}, 'codes': []}
                try:
                    _, pairs = analyze(case['sequence'], data['max_pair_tokens'], data['max_primitives'],
                                       data['extrusion_partition'], data.get('sketch_partition', 'curve'))
                    batch = to_device(collate_pairs(pairs), device)
                    _, codes, mask = vq.encode(batch['tokens'], batch['groups'])
                    reconstructed, fields = [], []
                    for i, pair in enumerate(pairs):
                        chosen = codes[i][mask[i]]
                        memory = vq.codebook.weight[chosen]
                        row['codes'].extend(chosen.tolist())
                        logits = vq.decode(batch['decoder_input'][i:i+1], memory[None],
                                           torch.ones((1, len(memory)), dtype=torch.bool, device=device))
                        fields.append(field_stats(logits, batch['targets'][i:i+1], pair['tokens']))
                        reconstructed.extend(vq.reconstruct(memory, constrained=True, force_add=i == 0))
                    row['fields'] = merge_fields(fields)
                    row['sequence'] = stringify(reconstructed)
                    split_pairs(reconstructed)
                except ValueError as error:
                    row['error'] = str(error)
                stream.write(json.dumps({**row, 'signature_sha256': signature_hash}, allow_nan=False) + '\n')
                stream.flush()
            usage.update(row['codes'])
            records.append(row)
            if (index + 1) % 16 == 0:
                print(json.dumps({'phase': 'reconstruct', 'output': str(root), 'cases': index + 1}), flush=True)
    decode_seconds = time.monotonic() - decode_start
    peak = torch.cuda.max_memory_allocated()
    del vq, checkpoint
    torch.cuda.empty_cache()
    geometry_settings = {**read_config('configs/benchmark.json'), 'output': str(root), 'render': False,
                         'seed': settings['seed'], 'point_count': settings['point_count']}
    runner = KernelRunner(geometry_settings)
    originals = {r['id']: r['sequence'] for r in cases}
    jobs = [(r['id'], role, originals[r['id']] if role == 'reference' else r['sequence'])
            for r in records for role in ('reference', 'reconstruction')]
    geometry = {}
    geometry_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=settings['kernel_workers']) as pool:
        for index, (job, result) in enumerate(zip(jobs, pool.map(lambda j: runner.reconstruct(j[2], j[0], j[1]), jobs)), 1):
            geometry[(job[0], job[1])] = result
            if index % 32 == 0:
                print(json.dumps({'phase': 'kernel', 'output': str(root), 'jobs': index, 'total': len(jobs)}), flush=True)
    valid_reference = valid = exact = syntax = 0
    distances, volume_errors = [], []
    for row in records:
        reference, prediction = [geometry[(row['id'], role)] for role in ('reference', 'reconstruction')]
        row['geometry'] = {'reference': reference, 'reconstruction': prediction}
        syntax += row['sequence'] is not None
        exact += row['sequence'] == originals[row['id']]
        row['known_valid_reference'] = reference['status'] == 'valid'
        valid_reference += row['known_valid_reference']
        row['paired_valid'] = reference['status'] == prediction['status'] == 'valid'
        valid += row['paired_valid']
        row['chamfer'] = None
        if row['paired_valid']:
            a, b = [np.asarray(normalize_points(np.load(g['points']), geometry_settings['normalization']))
                    for g in (reference, prediction)]
            row['chamfer'] = float(np.mean(cKDTree(b).query(a)[0] ** 2) + np.mean(cKDTree(a).query(b)[0] ** 2))
            distances.append(row['chamfer'])
            volume_errors.append(abs(prediction['volume'] - reference['volume']) / reference['volume'])
        if row['sequence']:
            predicted = features(row['sequence'])
            row['count_exact'] = {key: predicted[key] == row['features'][key] for key in ('pairs', 'curves', 'loops', 'faces')}
    strata = {}
    predicates = {'arcs': lambda f: f['arcs'] > 0, 'circles': lambda f: f['circles'] > 0,
                  'holes': lambda f: f['holes'] > 0, 'multiple_loops': lambda f: f['loops'] > 1,
                  'multiple_extrusions': lambda f: f['pairs'] > 1,
                  'high_primitive_count': lambda f: f['max_curve_primitives'] >= 10}
    for name, predicate in predicates.items():
        selected = [r for r in records if r['known_valid_reference'] and predicate(r['features'])]
        strata[name] = {'known_valid_references': len(selected), 'valid_reconstructions': sum(r['paired_valid'] for r in selected)}
    total_usage = sum(usage.values())
    rate = valid / valid_reference if valid_reference else None
    # Wilson 95% interval, conditional on valid references.
    interval = None
    if valid_reference:
        z, n = 1.96, valid_reference
        center = (rate + z*z/(2*n)) / (1 + z*z/n)
        half = z * math.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / (1 + z*z/n)
        interval = [center-half, center+half]
    summary = {'cases': len(records), 'known_valid_references': valid_reference,
               'valid_reconstructions_of_valid_references': valid, 'conditional_valid_rate': rate,
               'conditional_valid_rate_wilson95': interval, 'syntax_valid': syntax, 'exact_sequences': exact,
               'paired_geometry_count': len(distances),
               'mean_normalized_squared_chamfer': statistics.mean(distances) if distances else None,
               'median_normalized_squared_chamfer': statistics.median(distances) if distances else None,
               'median_relative_volume_error': statistics.median(volume_errors) if volume_errors else None,
               'teacher_forced_fields': merge_fields(r['fields'] for r in records),
               'exact_count_rates_all_cases': {key: sum(r.get('count_exact', {}).get(key, False) for r in records)/len(records)
                                               for key in ('pairs', 'curves', 'loops', 'faces')},
               'strata': strata, 'reference_statuses': dict(collections.Counter(r['geometry']['reference']['status'] for r in records)),
               'reconstruction_statuses': dict(collections.Counter(r['geometry']['reconstruction']['status'] for r in records)),
               'used_codes': len(usage), 'codebook_perplexity': math.exp(-sum((n/total_usage)*math.log(n/total_usage) for n in usage.values())) if total_usage else None,
               'decode_seconds_this_invocation': decode_seconds, 'peak_cuda_allocated_bytes': peak,
               'geometry_seconds_this_invocation': time.monotonic() - geometry_start}
    write_json(root / 'per_case.json', records)
    report = {'signature': signature, 'environment': environment, 'summary': summary,
              'limits': ['Ground-truth codes and original pair boundaries, no LLM; extrusion count is supplied, not predicted.',
                         'Geometry distance uses valid pairs only and independent max-absolute normalization; volume error is also reported.',
                         'Cases originate in the curve-filtered prepared validation pool; raw coverage is reported separately.']}
    write_json(root / 'report.json', report)
    print(json.dumps({'output': str(root), 'summary': summary}), flush=True)
    return report


def run_experiment(config, settings):
    require_container()
    import torch
    from .vq import train_vq
    torch.set_num_threads(2)
    manifest = prepare_experiment(config, settings)
    root = Path(settings['output'])
    write_json(root / 'state.json', {'status': 'running', 'phase': 'baseline'})
    try:
        baseline = evaluate_checkpoint(Path(config['vq']['output']) / 'best.pt', manifest['validation'], settings, root / 'baseline')
        results = []
        for variant in manifest['variants']:
            result = {'name': variant['name']}
            for phase in ('overfit', 'screen'):
                spec = variant[phase]
                cfg = read_config(spec['config_path'])
                output = Path(cfg['vq']['output'])
                write_json(root / 'state.json', {'status': 'running', 'variant': variant['name'], 'phase': phase})
                if not (output / 'config.json').exists():
                    resume = output / 'last.pt'
                    train_vq(cfg, str(resume) if resume.exists() else None)
                    torch.cuda.empty_cache()
                cases = manifest['overfit'] if phase == 'overfit' else manifest['validation']
                report = evaluate_checkpoint(output / 'best.pt', cases, settings, Path(spec['config_path']).parent / 'evaluation')
                result[phase] = report['summary']
                metrics = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
                result[phase]['training'] = {'epochs': len(metrics), 'total_epoch_seconds': sum(m['epoch_seconds'] for m in metrics),
                                              'peak_cuda_allocated_bytes': max(m['peak_cuda_allocated_bytes'] for m in metrics),
                                              'last_epoch': metrics[-1]}
            results.append(result)
            write_json(root / 'results.json', {'baseline': baseline['summary'], 'variants': results})
        lines = ['# Local tokenizer reconstruction comparison', '', manifest['selection'], '',
                 '| Variant | Phase | Valid / known-valid | Mean squared CD (valid pairs) | Teacher accuracy | Used codes |',
                 '| --- | --- | ---: | ---: | ---: | ---: |']
        for name, phase, summary in [('existing-epoch72', 'baseline', baseline['summary'])] + [
                (r['name'], phase, r[phase]) for r in results for phase in ('overfit', 'screen')]:
            accuracy = summary['teacher_forced_fields'].get('all', {}).get('accuracy')
            lines.append(f'| {name} | {phase} | {summary["valid_reconstructions_of_valid_references"]}/{summary["known_valid_references"]} | {summary["mean_normalized_squared_chamfer"]} | {accuracy} | {summary["used_codes"]} |')
        lines += ['', *manifest['limits'], '',
                  'This is a screening experiment, not a full-data retraining or a paper reproduction. No downstream SFT is scheduled automatically.',
                  'Approximate 98% reconstruction validity is an engineering target, not a paper-reported result. Compare fidelity on common valid case IDs before selecting a winner when valid subsets differ.']
        (root / 'comparison.md').write_text('\n'.join(lines) + '\n')
        write_json(root / 'state.json', {'status': 'complete', 'results': str(root / 'results.json')})
    except BaseException as error:
        write_json(root / 'state.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise
