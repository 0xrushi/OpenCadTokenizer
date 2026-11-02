"""Tiny-set diagnosis of continuous memory, VQ memory, and autoregressive decoding."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import collections
import copy
import json
import os
from pathlib import Path
import shutil
import sqlite3
import statistics
import time

from .common import file_hash, load_checkpoint, read_config, require_container, save_checkpoint, seed_all, write_json
from .representation import PAD, analyze, split_pairs, stringify


def select_training_candidates(source, database):
    """Read prior TRAIN IDs only; never use validation/test results for selection."""
    manifest = read_config(source)
    rows = {r['id']: r for r in manifest['train']}
    preferred = [r['id'] for r in manifest['overfit']]
    ordered = preferred + [key for key in rows if key not in preferred]
    with sqlite3.connect(database) as db:
        for key in ordered:
            stored = db.execute('SELECT sequence,split,tokenizer_split,component FROM geometry WHERE id=?', (key,)).fetchone()
            row = rows[key]
            if not stored or stored[:3] != (row['sequence'], 'train', 'train') or stored[3] != row['component']:
                raise ValueError('Overfit candidate is not verified tokenizer training geometry')
    return [rows[key] for key in ordered]


def source_signature(config, settings):
    names = ('tokenizer_overfit.py', 'tokenizer_experiment.py', 'vq.py', 'representation.py',
             'geometry_worker.py', 'bench_runtime.py', 'bench_metrics.py')
    return {'schema': 'tokenizer-overfit-v1', 'config': config, 'settings': settings,
            'database_sha256': file_hash(config['data']['database']),
            'source_manifest_sha256': file_hash(settings['source_manifest']),
            'code': {name: file_hash(Path('cadtokenizer') / name) for name in names},
            'image_id': os.environ.get('CADTOKENIZER_IMAGE_ID')}


def kernel_runner(settings):
    from .bench_runtime import KernelRunner
    cfg = read_config('configs/benchmark.json')
    cfg.update(output=str(Path(settings['output']) / 'kernel-cache'), render=False,
               seed=settings['seed'], point_count=settings['point_count'])
    return KernelRunner(cfg)


def freeze_cases(config, settings, signature):
    from .tokenizer_experiment import diverse_subset
    root = Path(settings['output'])
    path = root / 'manifest.json'
    if path.exists():
        manifest = read_config(path)
        if manifest['signature'] != signature:
            raise ValueError('Overfit inputs/code changed; choose a new output directory')
        return manifest
    candidates = select_training_candidates(settings['source_manifest'], config['data']['database'])
    runner = kernel_runner(settings)
    valid, rejected = [], []
    for case in candidates:
        result = runner.reconstruct(case['sequence'], case['id'], 'reference')
        if result['status'] == 'valid':
            valid.append(case)
        else:
            rejected.append({'id': case['id'], 'geometry': result})
        if len(valid) >= settings['diverse_geometries']:
            break
    if len(valid) < settings['diverse_geometries']:
        raise ValueError('Insufficient known-valid training candidates')
    diverse = diverse_subset(valid, settings['diverse_geometries'])
    single_pairs = [r for r in diverse if r['features']['pairs'] == 1]
    if not single_pairs:
        raise ValueError('Need a single-pair shape for first probe')
    single = min(single_pairs, key=lambda r: (len(r['sequence']), r['id']))
    manifest = {'signature': signature, 'one': [single], 'diverse': diverse, 'rejected_originals': rejected,
                'selection': 'Known-kernel-valid tokenizer training geometry only. These are memorization probes, not held-out quality estimates.'}
    write_json(path, manifest)
    return manifest


def prepared_pairs(cases, config):
    data = config['data']
    pairs, locations = [], []
    for case in cases:
        _, parts = analyze(case['sequence'], data['max_pair_tokens'], data['max_primitives'],
                           data['extrusion_partition'], data.get('sketch_partition', 'curve'))
        locations.append((len(pairs), len(parts)))
        pairs.extend(parts)
    return pairs, locations


def memory_paths(model, batch):
    pooled, mask = model.pool(batch['tokens'], batch['groups'])
    _, codes, vq_loss = model.quantize(pooled, mask)
    return pooled, model.codebook(codes), mask, codes, vq_loss


def training_parts(model, batch, mode):
    """Return separately normalized CE and VQ terms for exact full-set accumulation."""
    import torch
    from torch.nn import functional as F
    pooled, mask = model.pool(batch['tokens'], batch['groups'])
    if mode == 'quantized':
        memory, codes, vq_loss = model.quantize(pooled, mask)
    elif mode == 'continuous':
        memory, codes, vq_loss = pooled, None, pooled.new_zeros(())
    else:
        raise ValueError('Memory mode must be continuous or quantized')
    logits = model.decode(batch['decoder_input'], memory, mask)
    active = batch['targets'].ne(PAD)
    ce_sum = F.cross_entropy(logits[active].float(), batch['targets'][active], reduction='sum')
    return {'ce_sum': ce_sum, 'token_count': int(active.sum()), 'vq_loss': vq_loss,
            'primitive_count': int(mask.sum()), 'correct': int(((logits.argmax(-1) == batch['targets']) & active).sum())}


def evaluate(model, cases, settings, step, output, reference_mode):
    import torch
    import numpy as np
    from scipy.spatial import cKDTree
    from .bench_metrics import normalize_points
    from .tokenizer_experiment import field_stats, merge_fields
    from .vq import collate_pairs, to_device
    path = Path(output) / f'evaluation-{step}.json'
    if path.exists():
        return read_config(path)
    model.eval()
    device = next(model.parameters()).device
    pairs, locations = prepared_pairs(cases, model.config)
    batch = to_device(collate_pairs(pairs), device)
    rows, fields, usage = [], {'continuous': [], 'quantized': []}, collections.Counter()
    with torch.inference_mode():
        pooled, quantized, mask, codes, vq_loss = memory_paths(model, batch)
        usage.update(codes[mask].tolist())
        latent_mse = float((pooled[mask] - quantized[mask]).square().mean())
        for index, case in enumerate(cases):
            start, count = locations[index]
            paths = [('continuous', pooled)]
            if reference_mode == 'quantized':
                paths.append(('quantized', quantized))
            for mode, vectors in paths:
                row = {'id': case['id'], 'mode': mode, 'sequence': None, 'error': None}
                output_tokens, case_fields = [], []
                for ordinal, pair_index in enumerate(range(start, start + count)):
                    memory = vectors[pair_index][mask[pair_index]]
                    logits = model.decode(batch['decoder_input'][pair_index:pair_index+1], memory[None],
                                         torch.ones((1, len(memory)), dtype=torch.bool, device=device))
                    case_fields.append(field_stats(logits, batch['targets'][pair_index:pair_index+1], pairs[pair_index]['tokens']))
                    try:
                        output_tokens.extend(model.reconstruct(memory, force_add=ordinal == 0))
                    except ValueError as error:
                        row['error'] = str(error)
                row['fields'] = merge_fields(case_fields)
                fields[mode].append(row['fields'])
                if row['error'] is None:
                    row['sequence'] = stringify(output_tokens)
                    split_pairs(output_tokens)
                row['exact_sequence'] = row['sequence'] == case['sequence']
                # Only run unmasked decoding once teacher predictions are near exact.
                # This distinguishes forced grammar choices from learned termination.
                row['unconstrained_exact_sequence'] = None
                if row['fields']['all']['accuracy'] >= settings['unconstrained_accuracy_threshold']:
                    try:
                        tokens = []
                        for pair_index in range(start, start + count):
                            tokens.extend(model.reconstruct(vectors[pair_index][mask[pair_index]], constrained=False))
                        row['unconstrained_exact_sequence'] = stringify(tokens) == case['sequence']
                    except ValueError:
                        row['unconstrained_exact_sequence'] = False
                rows.append(row)
    from .tokenizer_regression import unique_geometry_jobs
    runner = kernel_runner(settings)
    jobs = [(case['id'], 'reference', case['sequence']) for case in cases]
    jobs += [(row['id'], 'prediction', row['sequence']) for row in rows]
    unique_jobs = unique_geometry_jobs(jobs)
    with ThreadPoolExecutor(max_workers=settings['kernel_workers']) as pool:
        geometry = {job: result for job, result in zip(
            unique_jobs, pool.map(lambda job: runner.reconstruct(job[2], job[0], job[1]), unique_jobs))}
    references = {case['id']: geometry[(case['id'], 'reference', case['sequence'])] for case in cases}
    for row in rows:
        prediction = geometry[(row['id'], 'prediction', row['sequence'])]
        reference = references[row['id']]
        if reference['status'] != 'valid':
            raise RuntimeError('Frozen training reference is no longer kernel-valid')
        row['geometry'] = prediction
        row['chamfer'] = row['relative_volume_error'] = None
        if prediction['status'] == 'valid':
            a, b = [np.asarray(normalize_points(np.load(g['points']))) for g in (reference, prediction)]
            row['chamfer'] = float(np.mean(cKDTree(b).query(a)[0]**2) + np.mean(cKDTree(a).query(b)[0]**2))
            row['relative_volume_error'] = abs(prediction['volume'] - reference['volume']) / reference['volume']
    branches = {}
    for mode in ('continuous', 'quantized'):
        selected = [row for row in rows if row['mode'] == mode]
        if not selected:
            continue
        distances = [row['chamfer'] for row in selected if row['chamfer'] is not None]
        branches[mode] = {'cases': len(cases), 'valid_solids': sum(r['geometry']['status'] == 'valid' for r in selected),
                          'exact_sequences': sum(r['exact_sequence'] for r in selected),
                          'unconstrained_exact_sequences': sum(r['unconstrained_exact_sequence'] is True for r in selected),
                          'unconstrained_evaluated': sum(r['unconstrained_exact_sequence'] is not None for r in selected),
                          'teacher': merge_fields(fields[mode]),
                          'mean_normalized_squared_chamfer': statistics.mean(distances) if distances else None,
                          'mean_relative_volume_error': statistics.mean(r['relative_volume_error'] for r in selected if r['relative_volume_error'] is not None) if distances else None,
                          'failures': dict(collections.Counter(r['geometry'].get('error', r['error']) for r in selected if r['geometry']['status'] != 'valid'))}
    own = branches[reference_mode]
    gate = (own['exact_sequences'] == len(cases) and own['valid_solids'] == len(cases)
            and own['teacher']['all']['accuracy'] >= settings['gate_teacher_accuracy'])
    report = {'step': step, 'training_mode': reference_mode, 'branches': branches, 'gate_passed': gate,
              'latent_mse': latent_mse, 'used_codes': len(usage), 'rows': rows,
              'limits': ['Continuous-control training does not train its codebook, so its quantized cross-path is not evaluated.',
                         'Within VQ training, the decoder is optimized on codebook vectors, so continuous cross-path results are diagnostic, not a separately trained continuous model.',
                         'Exact SEM reconstruction and kernel validity define the gate; sampled Chamfer has a nonzero sampling floor even for exact geometry.']}
    write_json(path, report)
    print(json.dumps({'phase': 'evaluation', 'output': str(output), 'step': step, 'gate_passed': gate,
                      'branches': {key: {k: value[k] for k in ('valid_solids', 'exact_sequences', 'mean_normalized_squared_chamfer')} for key, value in branches.items()}}), flush=True)
    return report


def train_probe(config, settings, cases, phase, mode, signature):
    import torch
    from .vq import PrimitiveVQVAE, collate_pairs, to_device
    root = Path(settings['output']) / phase / mode
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'complete.json').exists():
        return read_config(root / 'complete.json')
    cfg = copy.deepcopy(config)
    cfg['vq'].update(dropout=0.0, reconstruction_loss='cross_entropy', output=str(root))
    seed_all(settings['seed'])
    device = torch.device('cuda')
    model = PrimitiveVQVAE(cfg).to(device)
    pairs, _ = prepared_pairs(cases, cfg)
    batches = [to_device(collate_pairs(pairs[start:start+settings['microbatch_size']]), device)
               for start in range(0, len(pairs), settings['microbatch_size'])]
    total_tokens = sum(int(batch['targets'].ne(PAD).sum()) for batch in batches)
    total_primitives = sum(max(p['groups']) + 1 for p in pairs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=0.0)
    start = 0
    checkpoint_path = root / 'last.pt'
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint['signature'] != signature or checkpoint['mode'] != mode or checkpoint['case_ids'] != [r['id'] for r in cases]:
            raise ValueError('Probe checkpoint is incompatible')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        start = checkpoint['step']
        del checkpoint
    else:
        model.eval()
        with torch.no_grad():
            pools = []
            for batch in batches:
                pooled, mask = model.pool(batch['tokens'], batch['groups'])
                pools.append(pooled[mask])
            values = torch.cat(pools)
            choices = torch.randint(len(values), (cfg['vq']['codebook_size'],), device=device)
            model.codebook.weight.copy_(values[choices] + torch.randn_like(model.codebook.weight)*0.01)
    torch.cuda.reset_peak_memory_stats()
    elapsed_start = time.monotonic()
    final = None
    for step in range(start, settings['max_updates'] + 1):
        if step != start:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            ce_total = vq_total = 0.0
            correct = 0
            for batch in batches:
                parts = training_parts(model, batch, mode)
                loss = parts['ce_sum']/total_tokens + parts['vq_loss']*parts['primitive_count']/total_primitives
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite overfit loss')
                loss.backward()
                ce_total += float(parts['ce_sum'].detach())/total_tokens
                vq_total += float(parts['vq_loss'].detach())*parts['primitive_count']/total_primitives
                correct += parts['correct']
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['vq']['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            if step % settings['logging_updates'] == 0:
                metrics = {'step': step, 'ce': ce_total, 'vq_loss': vq_total, 'teacher_accuracy_before_update': correct/total_tokens,
                           'gradient_norm': float(norm), 'wall_seconds': time.monotonic()-elapsed_start,
                           'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}
                with (root/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(metrics, allow_nan=False)+'\n')
                print(json.dumps({'phase': phase, 'mode': mode, **metrics}), flush=True)
        if step in settings['evaluation_updates'] or step == settings['max_updates']:
            final = evaluate(model, cases, settings, step, root, mode)
            # RNG is saved after diagnostics. Deterministic no-dropout probes use
            # no randomness during an update, so resume remains equivalent.
            save_checkpoint(checkpoint_path, {'signature': signature, 'config': cfg, 'model': model.state_dict(),
                                             'optimizer': optimizer.state_dict(), 'mode': mode,
                                             'case_ids': [r['id'] for r in cases], 'step': step,
                                             'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()})
            if final['gate_passed'] and step >= settings['minimum_updates']:
                break
    if final is None:
        raise RuntimeError('No diagnostic evaluation was produced')
    summary = {'phase': phase, 'mode': mode, 'steps': final['step'], 'gate_passed': final['gate_passed'],
               'branches': final['branches'], 'latent_mse': final['latent_mse'], 'used_codes': final['used_codes'],
               'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
               'wall_seconds_this_invocation': time.monotonic()-elapsed_start, 'checkpoint': str(checkpoint_path)}
    write_json(root/'complete.json', summary)
    del model, optimizer, batches
    torch.cuda.empty_cache()
    return summary


def write_report(root, results, manifest):
    lines = ['# Tiny-set tokenizer diagnosis', '', manifest['selection'], '',
             'Cross-entropy, dropout zero, weight decay zero, FP32 tensors with high matmul precision (TF32 permitted), identical initial model seeds. One optimizer update accumulates the entire selected set with token-weighted CE and primitive-weighted VQ loss. These settings test memorization and differ from production training.', '',
             '| Training set | Trained memory | Updates | Exact primary / cases | Valid primary / cases | Teacher accuracy | Gate |',
             '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for item in results:
        branch = item['branches'][item['mode']]
        lines.append(f'| {item["phase"]} | {item["mode"]} | {item["steps"]} | {branch["exact_sequences"]}/{branch["cases"]} | {branch["valid_solids"]}/{branch["cases"]} | {branch["teacher"]["all"]["accuracy"]:.5f} | {item["gate_passed"]} |')
    lines += ['', 'Both memory paths, per-field teacher accuracy, constrained and near-converged unconstrained decoding, latent MSE, per-case geometry failures, volume error and Chamfer are in the evaluation JSON files.', '',
              'The continuous control does not train its codebook; that untrained cross-path is not evaluated. The VQ-trained model is evaluated with both pooled and quantized memory. Original pair boundaries are supplied. All results concern training-set memorization; no held-out quality claim is made.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    write_json(root/'results.json', {'runs': results})


def run_overfit(config, settings):
    require_container()
    import torch
    from .bench_runtime import evaluation_doctor
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    if not torch.cuda.is_available():
        raise RuntimeError('The local overfit probe requires CUDA')
    if settings['max_updates'] < settings['minimum_updates'] or settings['minimum_updates'] < 1:
        raise ValueError('Invalid optimizer-update budget')
    root = Path(settings['output'])
    root.mkdir(parents=True, exist_ok=True)
    environment = evaluation_doctor()
    signature = source_signature(config, settings)
    manifest = freeze_cases(config, settings, signature)
    write_json(root/'environment.json', {**environment, 'gpu': torch.cuda.get_device_name(),
                                         'torch': torch.__version__, 'matmul_precision': torch.get_float32_matmul_precision()})
    source = root/'source'
    source.mkdir(exist_ok=True)
    for name in signature['code']:
        shutil.copy2(Path('cadtokenizer')/name, source/name)
    results = []
    try:
        for phase in ('one', 'diverse'):
            for mode in ('continuous', 'quantized'):
                write_json(root/'state.json', {'status': 'running', 'phase': phase, 'mode': mode})
                results.append(train_probe(config, settings, manifest[phase], phase, mode, signature))
                write_report(root, results, manifest)
            if phase == 'one' and not all(item['gate_passed'] for item in results):
                write_json(root/'state.json', {'status': 'complete', 'one_shape_gate_passed': False,
                                               'diverse_probe': 'not_started_pending_one_shape_diagnosis'})
                return
        write_json(root/'state.json', {'status': 'complete', 'one_shape_gate_passed': True,
                                       'diverse_gate_passed': all(item['gate_passed'] for item in results if item['phase']=='diverse')})
    except BaseException as error:
        write_json(root/'state.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise
