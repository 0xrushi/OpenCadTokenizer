"""Training-only throughput/memory profile for the VQ tokenizer (no checkpoints written)."""
import json
import threading
import time
from pathlib import Path

import torch

from .common import device_for, require_container, seed_all, write_json
from .vq import PairDataset, PrimitiveVQVAE, collate_pairs, to_device


def _sample_utilization(samples, stop):
    try:
        while not stop.is_set():
            samples.append(torch.cuda.utilization())
            time.sleep(0.25)
    except Exception:  # NVML can be unavailable; report that rather than guess.
        samples.append(None)


def _measure(config, items, microbatch, settings, device):
    seed_all(config.get('seed', 42))
    model = PrimitiveVQVAE(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['vq']['learning_rate'])
    batches = [to_device(collate_pairs(items[i:i + microbatch]), device)
               for i in range(0, len(items) - microbatch + 1, microbatch)]
    with torch.no_grad():
        pooled, mask = model.pool(batches[0]['tokens'], batches[0]['groups'])
        values = pooled[mask]
        model.codebook.weight.copy_(values[torch.randint(len(values), (config['vq']['codebook_size'],), device=device)])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    steps = settings['warmup_updates'] + settings['measured_updates']
    samples, stop = [], threading.Event()
    sampler = threading.Thread(target=_sample_utilization, args=(samples, stop))
    for step in range(steps):
        if step == settings['warmup_updates']:
            torch.cuda.synchronize(device)
            start = time.monotonic()
            sampler.start()
        batch = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            result = model(**batch)
        result['loss'].backward()
        optimizer.step()
    torch.cuda.synchronize(device)
    seconds = time.monotonic() - start
    stop.set()
    sampler.join()
    utilization = [s for s in samples if s is not None]
    return {'pairs_per_second': settings['measured_updates'] * microbatch / seconds,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'mean_gpu_utilization': sum(utilization) / len(utilization) if utilization else None}


def run_profile(config, output=None):
    require_container()
    settings = config['profile']
    device = device_for(config)
    total = torch.cuda.get_device_properties(device).total_memory
    dataset = PairDataset(config['data']['database'], 'train')
    # The longest-pair groups bound memory; a seeded random sample estimates typical throughput.
    items = [dataset[i] for i in range(len(dataset))]
    longest = sorted(items, key=lambda item: -len(item['tokens']))
    generator = torch.Generator().manual_seed(config.get('seed', 42))
    typical = [items[i] for i in torch.randperm(len(items), generator=generator)[:max(settings['microbatches']) * 8].tolist()]
    result = {'database': config['data']['database'], 'max_primitives': config['data']['max_primitives'],
              'total_memory_bytes': total, 'memory_fraction_limit': settings['memory_fraction'],
              'longest_pair_tokens': len(longest[0]['tokens']), 'options': []}
    for microbatch in settings['microbatches']:
        row = {'microbatch': microbatch}
        for name, pool in (('longest', longest[:microbatch * 4]), ('typical', typical)):
            try:
                row[name] = _measure(config, pool, microbatch, settings, device)
            except torch.OutOfMemoryError:
                row[name] = {'error': 'out of memory'}
            torch.cuda.empty_cache()
        peak = row['longest'].get('peak_allocated_bytes')
        row['eligible'] = peak is not None and peak <= settings['memory_fraction'] * total and 'error' not in row['typical']
        result['options'].append(row)
        print(json.dumps(row), flush=True)
    eligible = [r for r in result['options'] if r['eligible']]
    result['selected_microbatch'] = max(eligible, key=lambda r: r['typical']['pairs_per_second'])['microbatch'] if eligible else None
    out = Path(output or config['vq']['output']) / 'profile.json'
    write_json(out, result)
    print(json.dumps({'selected_microbatch': result['selected_microbatch'], 'written': str(out)}), flush=True)
