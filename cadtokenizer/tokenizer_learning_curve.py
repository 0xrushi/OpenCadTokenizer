"""Fixed-update, held-out tokenizer learning curves in the cached local image."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import statistics
import time

from .common import file_hash, load_checkpoint, read_config, require_container, save_checkpoint, seed_all, write_json
from .representation import PAD


def verify_cases(manifest, database, train_count, val_count):
    train, val = manifest['train'], manifest['validation']
    if len(train) != train_count or len(val) != val_count:
        raise ValueError('Frozen training/validation counts differ from requested experiment')
    ids = [r['id'] for r in train + val]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate or overlapping training/validation IDs')
    if {r['component'] for r in train} & {r['component'] for r in val}:
        raise ValueError('Training/validation component overlap')
    with sqlite3.connect(database) as db:
        for split, cases in [('train', train), ('val', val)]:
            for row in cases:
                stored = db.execute('SELECT sequence,split,tokenizer_split,component FROM geometry WHERE id=?', (row['id'],)).fetchone()
                if stored != (row['sequence'], split, split, row['component']):
                    raise ValueError('Frozen geometry or split differs from prepared source')
    return train, val


def paired_comparison(baseline_rows, candidate_rows):
    a, b = [{r['id']: r for r in rows} for rows in (baseline_rows, candidate_rows)]
    if a.keys() != b.keys() or any(a[k]['known_valid_reference'] != b[k]['known_valid_reference'] for k in a):
        raise ValueError('Paired reports have different cases or reference validity')
    common = [k for k in a if a[k]['paired_valid'] and b[k]['paired_valid']]
    old = statistics.mean(a[k]['chamfer'] for k in common) if common else None
    new = statistics.mean(b[k]['chamfer'] for k in common) if common else None
    baseline_valid = sum(r['paired_valid'] for r in a.values())
    candidate_valid = sum(r['paired_valid'] for r in b.values())
    return {'common_valid_cases': len(common), 'baseline_valid': baseline_valid, 'candidate_valid': candidate_valid,
            'baseline_mean_chamfer_common': old, 'candidate_mean_chamfer_common': new,
            'baseline_valid_to_candidate_invalid': sum(a[k]['paired_valid'] and not b[k]['paired_valid'] for k in a),
            'baseline_invalid_to_candidate_valid': sum(not a[k]['paired_valid'] and b[k]['paired_valid'] for k in a),
            'candidate_lower_chamfer_cases': sum(b[k]['chamfer'] < a[k]['chamfer'] for k in common),
            'improves_validity_and_geometry': candidate_valid > baseline_valid and bool(common) and new <= old}


@contextmanager
def preserve_rng():
    import numpy as np
    import torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state_all(cuda_state)


def epoch_order(size, seed, epoch):
    import torch
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(size, generator=generator).tolist()


def initialization_pairs(settings, pairs, config):
    """Keep bootstrap examples fixed when profiling/changing training microbatch."""
    from .representation import primitive_ids
    from .vq import PairDataset
    count = settings['bootstrap_batch_size'] if 'bootstrap_batch_size' in settings else settings['microbatch_size']
    if settings.get('bootstrap_database'):
        source = PairDataset(settings['bootstrap_database'], 'train')
        selected = [source[i] for i in epoch_order(len(source), settings['seed'], 0)[:count]]
        training_ids = {pair['geometry_id'] for pair in pairs}
        if any(pair['geometry_id'] not in training_ids for pair in selected):
            raise ValueError('Codebook bootstrap contains geometry outside training')
        for pair in selected:
            pair['groups'] = primitive_ids(pair['tokens'], config['data']['extrusion_partition'],
                                           config['data'].get('sketch_partition', 'curve'))
        return selected
    return [pairs[i] for i in epoch_order(len(pairs), settings['seed'], 0)[:count]]


def initialize_codebook(model, pairs, device):
    import torch
    from .vq import collate_pairs, to_device
    with torch.no_grad():
        batch = to_device(collate_pairs(pairs), device)
        pooled, mask = model.pool(batch['tokens'], batch['groups'])
        values = pooled[mask]
        indices = torch.randint(len(values), (model.config['vq']['codebook_size'],), device=device)
        model.codebook.weight.copy_(values[indices] + torch.randn_like(model.codebook.weight)*0.01)


def share_geometry_cache(sources, target):
    """Reuse only fingerprint-addressed caches; KernelRunner still verifies every request."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for source in map(Path, sources):
        if not source.exists():
            continue
        for item in source.iterdir():
            link = target / item.name
            if item.is_dir() and (item / 'result.json').exists() and not link.exists() and not link.is_symlink():
                link.symlink_to(os.path.relpath(item, target), target_is_directory=True)


def freeze(config, settings, environment):
    from .bench_data import code_hash
    from .tokenizer_experiment import make_database
    root = Path(settings['output'])
    root.mkdir(parents=True, exist_ok=True)
    signature = {'schema': 'tokenizer-learning-curve-v1', 'config': config, 'settings': settings,
                 'source_manifest_sha256': file_hash(settings['source_manifest']),
                 'source_database_sha256': file_hash(config['data']['database']),
                 'baseline_checkpoint_sha256': file_hash(Path(config['vq']['output']) / 'best.pt'),
                 'bootstrap_database_sha256': file_hash(settings['bootstrap_database']) if settings.get('bootstrap_database') else None,
                 'code_sha256': code_hash(), 'environment': environment}
    path = root / 'manifest.json'
    if path.exists():
        manifest = read_config(path)
        if manifest['signature'] != signature:
            raise ValueError('Learning-curve code/inputs/environment changed; choose a fresh output')
        for variant in manifest['variants'].values():
            if file_hash(variant['config_path']) != variant['config_sha256'] or file_hash(variant['database']) != variant['database_sha256']:
                raise ValueError('Learning-curve config/database changed')
        return manifest
    source = read_config(settings['source_manifest'])
    train, val = verify_cases(source, config['data']['database'], settings['train_geometries'], settings['validation_geometries'])
    variants = {}
    for grouping in ('curve', 'loop'):
        cfg = copy.deepcopy(config)
        cfg['seed'] = settings['seed']
        cfg['data'].update(sketch_partition=grouping, database=str(root / grouping / 'cad.sqlite'))
        cfg['vq'].update(output=str(root / grouping), reconstruction_loss=settings.get('reconstruction_loss', 'cross_entropy'),
                         batch_size=settings['microbatch_size'],
                         gradient_accumulation_steps=settings['effective_batch_size'] // settings['microbatch_size'])
        make_database(cfg['data']['database'], train, val, cfg['data'])
        config_path = root / grouping / 'config.json'
        write_json(config_path, cfg)
        variants[grouping] = {'config_path': str(config_path), 'config_sha256': file_hash(config_path),
                              'database': cfg['data']['database'], 'database_sha256': file_hash(cfg['data']['database'])}
    manifest = {'signature': signature, 'train': train, 'validation': val, 'variants': variants,
                'selection': source['selection'],
                'loop_trigger': 'After curve completes: any evaluated curve checkpoint has more valid solids than the baseline and no higher mean CD on shared valid case IDs. Exploratory screen, not statistical proof.',
                'limits': ['512 training geometries are a subset of the full-data baseline corpus; this is not a matched full-data reproduction.',
                           'Validation IDs are fixed and originate in the original curve-supported prepared pool.',
                           'No test benchmark or LLM is used. Original pair boundaries are supplied.']}
    write_json(path, manifest)
    source_root = root / 'source'
    source_root.mkdir(exist_ok=True)
    for file in Path('cadtokenizer').glob('*.py'):
        shutil.copy2(file, source_root / file.name)
    shutil.copy2('scripts/cad.sh', source_root / 'cad.sh')
    return manifest


def evaluation_settings(settings):
    """Keep geometry sampling fixed when replicating a training trajectory."""
    if 'evaluation_seed' in settings:
        return {**settings, 'seed': settings['evaluation_seed']}
    return settings


def evaluate_point(checkpoint, manifest, settings, grouping, step, root):
    import torch
    from .tokenizer_experiment import evaluate_checkpoint
    directory = root / grouping / f'evaluation-{step}'
    sources = [root / 'baseline' / 'geometry']
    sources += [root / grouping / f'evaluation-{previous}' / 'geometry'
                for previous in settings['evaluation_updates'] if previous < step]
    share_geometry_cache(sources, directory / 'geometry')
    with preserve_rng():
        report = evaluate_checkpoint(checkpoint, manifest['validation'], evaluation_settings(settings), directory)
    baseline_rows = read_config(root / 'baseline' / 'per_case.json')
    candidate_rows = read_config(directory / 'per_case.json')
    paired = paired_comparison(baseline_rows, candidate_rows)
    result = {'step': step, 'summary': report['summary'], 'paired': paired,
              'checkpoint': str(checkpoint), 'checkpoint_sha256': file_hash(checkpoint)}
    write_json(directory / 'paired_comparison.json', paired)
    print(json.dumps({'phase': 'comparison', 'grouping': grouping, 'step': step, **paired}), flush=True)
    torch.cuda.reset_peak_memory_stats()
    return result


def train_variant(manifest, settings, grouping, root):
    import torch
    from .vq import LOSS_NORMALIZATION, PairDataset, PrimitiveVQVAE, accumulated_loss, collate_pairs, to_device
    directory = root / grouping
    if (directory / 'complete.json').exists():
        return read_config(directory / 'complete.json')
    spec = manifest['variants'][grouping]
    cfg = read_config(spec['config_path'])
    seed_all(settings['seed'])
    device = torch.device('cuda')
    dataset = PairDataset(cfg['data']['database'], 'train')
    pairs = [dataset[i] for i in range(len(dataset))]
    model = PrimitiveVQVAE(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['vq']['learning_rate'], weight_decay=cfg['vq']['weight_decay'])
    step = epoch = offset = 0
    peak = 0
    last = directory / 'last.pt'
    if last.exists():
        checkpoint = load_checkpoint(last)
        if checkpoint['experiment_signature'] != manifest['signature'] or checkpoint['database_sha256'] != spec['database_sha256']:
            raise ValueError('Learning-curve checkpoint mismatch')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step, epoch, offset, peak = [checkpoint[key] for key in ('step', 'epoch', 'offset', 'training_peak_cuda_allocated_bytes')]
        torch.set_rng_state(checkpoint['rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        del checkpoint
    else:
        # Match production's first-microbatch codebook bootstrap, with the
        # deterministic epoch order recorded by this experiment protocol.
        initialize_codebook(model, initialization_pairs(settings, pairs, cfg), device)
    use_amp = torch.cuda.is_bf16_supported()
    torch.cuda.reset_peak_memory_stats()
    points = []
    for previous in settings['evaluation_updates']:
        checkpoint_path = directory / f'checkpoint-{previous}.pt'
        if previous <= step:
            if previous == step and not checkpoint_path.exists():
                checkpoint_path.hardlink_to(last)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f'Missing earlier milestone checkpoint: {checkpoint_path}')
            points.append(evaluate_point(checkpoint_path, manifest, settings, grouping, previous, root))
    started = time.monotonic()
    order = epoch_order(len(pairs), settings['seed'], epoch)
    while step < settings['max_updates']:
        indices = order[offset:offset+settings['effective_batch_size']]
        chosen = [pairs[i] for i in indices]
        batches = [collate_pairs(chosen[start:start+settings['microbatch_size']])
                   for start in range(0, len(chosen), settings['microbatch_size'])]
        tokens = sum(int(batch['targets'].ne(PAD).sum()) for batch in batches)
        primitives = sum(int((batch['groups'].max(dim=1).values+1).sum()) for batch in batches)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        reconstruction = vq_loss = 0.0
        correct = 0
        update_started = time.monotonic()
        for batch in batches:
            batch = to_device(batch, device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                result = model(**batch)
                loss = accumulated_loss(result, tokens, primitives)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite learning-curve training loss')
            loss.backward()
            reconstruction += float(result['reconstruction'].detach()) * int(result['token_count']) / tokens
            vq_loss += float(result['vq_loss'].detach()) * result['indices'].numel() / primitives
            correct += int(result['correct'])
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['vq']['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        step += 1
        offset += len(indices)
        if offset == len(pairs):
            epoch, offset = epoch+1, 0
            order = epoch_order(len(pairs), settings['seed'], epoch)
        peak = max(peak, torch.cuda.max_memory_allocated())
        if step % settings['logging_updates'] == 0:
            metrics = {'step': step, 'epoch': epoch, 'offset': offset, 'reconstruction': reconstruction, 'vq_loss': vq_loss,
                       'teacher_accuracy_on_last_training_batch': correct/tokens, 'gradient_norm': float(norm),
                       'last_update_seconds': time.monotonic()-update_started, 'wall_seconds_this_invocation': time.monotonic()-started,
                       'training_peak_cuda_allocated_bytes': peak}
            with (directory / 'training.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics, allow_nan=False)+'\n')
            write_json(root / 'state.json', {'status': 'running', 'phase': 'training', 'grouping': grouping, 'step': step})
            print(json.dumps({'grouping': grouping, **metrics}), flush=True)
        if step % settings['save_updates'] == 0 or step in settings['evaluation_updates'] or step == settings['max_updates']:
            state = {'config': cfg, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'step': step, 'epoch': epoch, 'offset': offset, 'training_peak_cuda_allocated_bytes': peak,
                     'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                     'experiment_signature': manifest['signature'], 'database_sha256': spec['database_sha256'],
                     'loss_normalization': LOSS_NORMALIZATION}
            save_checkpoint(last, state)
            del state
        if step in settings['evaluation_updates']:
            checkpoint_path = directory / f'checkpoint-{step}.pt'
            if checkpoint_path.exists():
                raise FileExistsError(f'Unexpected pre-existing checkpoint: {checkpoint_path}')
            checkpoint_path.hardlink_to(last)
            write_json(root / 'state.json', {'status': 'running', 'phase': 'evaluation', 'grouping': grouping, 'step': step})
            points.append(evaluate_point(checkpoint_path, manifest, settings, grouping, step, root))
    result = {'grouping': grouping, 'points': points, 'updates': step, 'epochs_completed': epoch,
              'train_pairs': len(pairs), 'training_peak_cuda_allocated_bytes': peak,
              'wall_seconds_this_invocation': time.monotonic()-started,
              'qualified_at_steps': [point['step'] for point in points if point['paired']['improves_validity_and_geometry']]}
    write_json(directory / 'complete.json', result)
    del optimizer, model, pairs
    torch.cuda.empty_cache()
    return result


def write_report(root, manifest, baseline, variants, loop_started):
    objective = manifest['signature']['settings'].get('reconstruction_loss', 'cross_entropy')
    label = 'EMD' if objective == 'emd' else 'CE'
    lines = ['# Held-out tokenizer learning curve', '',
             f'Fixed original 512 training geometries and 128 validation geometries. Production dropout and weight decay, {objective}, corrected token/primitive normalization. All new models start from scratch.', '',
             '| Model | Updates | Valid / known-valid | Mean CD (own valid subset) | Teacher accuracy | Paired CD: existing / new |',
             '| --- | ---: | ---: | ---: | ---: | --- |']
    s = baseline['summary']
    lines.append(f'| Existing epoch-72 | — | {s["valid_reconstructions_of_valid_references"]}/{s["known_valid_references"]} | {s["mean_normalized_squared_chamfer"]} | {s["teacher_forced_fields"]["all"]["accuracy"]:.4f} | — |')
    for variant in variants:
        for point in variant['points']:
            s, p = point['summary'], point['paired']
            lines.append(f'| {variant["grouping"]} + {label} | {point["step"]} | {s["valid_reconstructions_of_valid_references"]}/{s["known_valid_references"]} | {s["mean_normalized_squared_chamfer"]} | {s["teacher_forced_fields"]["all"]["accuracy"]:.4f} | {p["baseline_mean_chamfer_common"]} / {p["candidate_mean_chamfer_common"]} (n={p["common_valid_cases"]}) |')
    lines += ['', f'Loop comparison launched: {loop_started}.', '', 'Loop trigger: '+manifest['loop_trigger'], '', *manifest['limits'], '',
              'CD uses independently sampled, per-shape-normalized surfaces and only valid pairs. Paired CD uses identical case IDs valid in both models; volume errors and failure strata remain in individual reports. A screen trigger is not statistical proof of superiority.']
    (root / 'comparison.md').write_text('\n'.join(lines)+'\n')
    write_json(root / 'results.json', {'baseline': baseline['summary'], 'variants': variants, 'loop_started': loop_started})


def run_learning_curve(config, settings):
    require_container()
    import torch
    from .bench_runtime import evaluation_doctor
    from .tokenizer_experiment import evaluate_checkpoint
    if settings.get('reconstruction_loss', 'cross_entropy') not in ('emd', 'cross_entropy'):
        raise ValueError('Reconstruction loss must be emd or cross_entropy')
    if settings['microbatch_size'] < 1 or settings['effective_batch_size'] < 1 or settings['effective_batch_size'] % settings['microbatch_size']:
        raise ValueError('Effective batch must be divisible by positive microbatch size')
    if (not settings['evaluation_updates'] or any(step < 1 for step in settings['evaluation_updates'])
            or sorted(set(settings['evaluation_updates'])) != settings['evaluation_updates']
            or settings['evaluation_updates'][-1] != settings['max_updates']):
        raise ValueError('Evaluation updates must be unique, increasing, and end at max_updates')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the local learning curve')
    environment = evaluation_doctor()
    environment = {**environment, 'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(),
                   'matmul_precision': torch.get_float32_matmul_precision()}
    manifest = freeze(config, settings, environment)
    root = Path(settings['output'])
    try:
        write_json(root / 'state.json', {'status': 'running', 'phase': 'baseline'})
        baseline = evaluate_checkpoint(Path(config['vq']['output']) / 'best.pt', manifest['validation'], settings, root / 'baseline')
        curve = train_variant(manifest, settings, 'curve', root)
        variants, loop_started = [curve], False
        write_report(root, manifest, baseline, variants, loop_started)
        if curve['qualified_at_steps'] and settings['run_loop_if_improved']:
            loop_started = True
            variants.append(train_variant(manifest, settings, 'loop', root))
        write_report(root, manifest, baseline, variants, loop_started)
        write_json(root / 'state.json', {'status': 'complete', 'loop_started': loop_started,
                                         'curve_qualified_at_steps': curve['qualified_at_steps']})
    except BaseException as error:
        write_json(root / 'state.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise
