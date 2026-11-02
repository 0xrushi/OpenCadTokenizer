"""Profile local batches and run a full-corpus CE tokenizer candidate."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import sqlite3
import statistics
import time

from .common import file_hash, fraction, read_config, require_container, seed_all, write_json


def full_source(config, settings):
    from .tokenizer_experiment import features
    root = Path(settings['output'])
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'full-input-manifest.json'
    signature = {'database_sha256': file_hash(config['data']['database']),
                 'anchor_manifest_sha256': file_hash(settings['anchor_manifest']),
                 'seed': settings['seed'], 'validation_geometries': settings['validation_geometries']}
    if path.exists():
        source = read_config(path)
        if source['signature'] != signature:
            raise ValueError('Full-data source selection changed; choose a fresh output')
        return source
    anchor = read_config(settings['anchor_manifest'])['validation']
    with sqlite3.connect(config['data']['database']) as db:
        train = [{'id': key, 'sequence': sequence, 'component': component} for key, sequence, component in db.execute(
            'SELECT id,sequence,component FROM geometry WHERE tokenizer_candidate=1 AND tokenizer_split="train" AND split="train" ORDER BY id')]
        candidates = {key: {'id': key, 'sequence': sequence, 'component': component} for key, sequence, component in db.execute(
            'SELECT id,sequence,component FROM geometry WHERE tokenizer_candidate=1 AND tokenizer_split="val" AND split="val" ORDER BY id')}
    if len(train) != settings['train_geometries']:
        raise ValueError('Full tokenizer training count changed')
    chosen = []
    for row in anchor:
        if row['id'] not in candidates or any(candidates[row['id']][key] != row[key] for key in ('id', 'sequence', 'component')):
            raise ValueError('Historical validation anchor changed')
        chosen.append(candidates.pop(row['id']))
    remaining = sorted(candidates.values(), key=lambda row: (fraction(row['id'], settings['seed']), row['id']))
    needed = settings['validation_geometries'] - len(chosen)
    if needed < 0 or len(remaining) < needed:
        raise ValueError('Invalid full validation panel size')
    chosen.extend(remaining[:needed])
    for row in chosen:
        row['features'] = features(row['sequence'])
    if {r['component'] for r in train} & {r['component'] for r in chosen}:
        raise ValueError('Full training/validation component overlap')
    source = {'signature': signature, 'train': train, 'validation': chosen,
              'anchor_validation_ids': [r['id'] for r in anchor],
              'selection': 'All prepared tokenizer training geometries; historical 128 validation IDs plus hash-seeded additional validation geometries. No output/kernel-quality filtering or test selection.'}
    write_json(path, source)
    return source


def profile_batch(config, settings, pairs, microbatch):
    import torch
    from .tokenizer_learning_curve import epoch_order, initialization_pairs, initialize_codebook
    from .vq import PrimitiveVQVAE, accumulated_loss, collate_pairs, to_device
    seed_all(settings['seed'])
    device = torch.device('cuda')
    model = PrimitiveVQVAE(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['vq']['learning_rate'], weight_decay=config['vq']['weight_decay'])
    initialize_codebook(model, initialization_pairs(settings, pairs, config), device)
    amp = torch.cuda.is_bf16_supported()
    order = epoch_order(len(pairs), settings['seed'], 0)
    effective = settings['effective_batch_size']

    def update(items):
        model.train()
        batches = [collate_pairs(items[start:start+microbatch]) for start in range(0, len(items), microbatch)]
        tokens = sum(int(batch['targets'].ne(0).sum()) for batch in batches)
        primitives = sum(int((batch['groups'].max(dim=1).values+1).sum()) for batch in batches)
        optimizer.zero_grad(set_to_none=True)
        for batch in batches:
            batch = to_device(batch, device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp):
                result = model(**batch)
                loss = accumulated_loss(result, tokens, primitives)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite batch profiling loss')
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['vq']['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()

    times = []
    torch.cuda.reset_peak_memory_stats()
    total = settings['profile_warmup_updates'] + settings['profile_measured_updates']
    for step in range(total):
        items = [pairs[i] for i in order[step*effective:(step+1)*effective]]
        if len(items) != effective:
            raise ValueError('Profiling corpus is too small for measured updates')
        torch.cuda.synchronize()
        started = time.monotonic()
        update(items)
        torch.cuda.synchronize()
        if step >= settings['profile_warmup_updates']:
            times.append(time.monotonic()-started)
    longest = sorted(pairs, key=lambda item: len(item['tokens']), reverse=True)[:effective]
    update(longest)
    torch.cuda.synchronize()
    result = {'microbatch_size': microbatch, 'effective_batch_size': effective,
              'median_update_seconds': statistics.median(times), 'mean_update_seconds': statistics.mean(times),
              'pairs_per_second': effective/statistics.median(times),
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
              'longest_pair_tokens_checked': max(len(p['tokens']) for p in longest),
              'longest_batch_guard_passed': True}
    del model, optimizer
    return result


def profile(config, settings):
    require_container()
    import torch
    from .bench_data import code_hash
    from .vq import PairDataset
    root = Path(settings['output'])
    root.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps(config))
    cfg['vq']['reconstruction_loss'] = 'cross_entropy'
    signature = {'code_sha256': code_hash(), 'config': cfg, 'settings': settings,
                 'database_sha256': file_hash(cfg['data']['database']),
                 'bootstrap_database_sha256': file_hash(settings['bootstrap_database']),
                 'image_id': os.environ.get('CADTOKENIZER_IMAGE_ID'), 'torch': str(torch.__version__),
                 'gpu': torch.cuda.get_device_name()}
    path = root / 'profile.json'
    if path.exists():
        report = read_config(path)
        if report['signature'] != signature:
            raise ValueError('Profiling inputs/code changed; choose a fresh output')
        return report
    dataset = PairDataset(cfg['data']['database'], 'train')
    pairs = [dataset[i] for i in range(len(dataset))]
    available, total = torch.cuda.mem_get_info()
    results = []
    for microbatch in settings['profile_microbatches']:
        if microbatch < 1 or settings['effective_batch_size'] % microbatch:
            raise ValueError('Invalid profiling microbatch')
        try:
            result = profile_batch(cfg, settings, pairs, microbatch)
            result['fits_memory_margin'] = result['peak_cuda_reserved_bytes'] <= available*settings['profile_memory_fraction']
        except torch.cuda.OutOfMemoryError:
            result = {'microbatch_size': microbatch, 'out_of_memory': True, 'fits_memory_margin': False}
        gc.collect()
        torch.cuda.empty_cache()
        results.append(result)
        print(json.dumps({'phase': 'profile', **result}), flush=True)
    safe = [r for r in results if r['fits_memory_margin']]
    if not safe:
        raise RuntimeError('No measured microbatch fits the local memory margin')
    selected = min(safe, key=lambda r: r['median_update_seconds'])
    report = {'signature': signature, 'train_pairs': len(pairs), 'available_vram_before_profile': available,
              'total_vram': total, 'results': results, 'selected_microbatch_size': selected['microbatch_size'],
              'selected_median_update_seconds': selected['median_update_seconds'],
              'estimated_training_seconds_excluding_evaluation_and_checkpoints': selected['median_update_seconds']*settings['max_updates']}
    write_json(path, report)
    return report


def _run_full(config, settings, profile_only=False):
    require_container()
    import torch
    from .tokenizer_learning_curve import run_learning_curve
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    if not torch.cuda.is_available():
        raise RuntimeError('Local full-data training requires CUDA')
    root = Path(settings['output'])
    root.mkdir(parents=True, exist_ok=True)
    try:
        write_json(root/'state.json', {'status': 'running', 'phase': 'profiling'})
        measured = profile(config, settings)
        if profile_only:
            write_json(root/'state.json', {'status': 'profile_complete', 'selected_microbatch_size': measured['selected_microbatch_size']})
            return
        write_json(root/'state.json', {'status': 'running', 'phase': 'preparing_full_data'})
        full_source(config, settings)
        protocol = {key: settings[key] for key in ('seed','train_geometries','validation_geometries','max_updates',
                                                  'evaluation_updates','effective_batch_size','logging_updates','save_updates',
                                                  'point_count','kernel_workers','bootstrap_database','bootstrap_batch_size')}
        protocol.update(output=str(root/'training'), source_manifest=str(root/'full-input-manifest.json'),
                        microbatch_size=measured['selected_microbatch_size'], reconstruction_loss='cross_entropy',
                        run_loop_if_improved=False)
        write_json(root/'training-settings.json', protocol)
        write_json(root/'state.json', {'status': 'running', 'phase': 'training',
                                       'selected_microbatch_size': measured['selected_microbatch_size'],
                                       'progress': str(root/'training/state.json')})
        run_learning_curve(config, protocol)
        write_json(root/'state.json', {'status': 'complete', 'selected_microbatch_size': measured['selected_microbatch_size'],
                                       'results': str(root/'training/results.json')})
    except BaseException as error:
        write_json(root/'state.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise


def run_full(config, settings, profile_only=False):
    require_container()
    import fcntl
    root = Path(settings['output'])
    root.mkdir(parents=True, exist_ok=True)
    with (root/'run.lock').open('a+') as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('A full-data profile/training job already holds this run lock; inspect state.json instead of launching another copy') from error
        return _run_full(config, settings, profile_only)
