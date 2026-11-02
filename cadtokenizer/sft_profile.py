"""Measure SFT microbatch memory (worst-case batches) and throughput (typical batches) on one GPU."""
from __future__ import annotations

import json
import random
import sqlite3
import statistics
import time
from pathlib import Path

from .common import load_checkpoint, require_container, seed_all, write_json


def load_lengths(tokens_path):
    with sqlite3.connect(tokens_path) as db:
        return [(i, n) for i, n in db.execute('SELECT id,LENGTH(input_ids)-LENGTH(REPLACE(input_ids,",",""))+1 FROM samples WHERE split="train"')]


def make_batches(tokens_path, lengths, microbatch, count, seed):
    """Longest-B, random-B and length-grouped-B batches of sample IDs (ID lists, not tensors)."""
    rng = random.Random(seed)
    ordered = sorted(lengths, key=lambda x: (x[1], x[0]))
    ids = [i for i, _ in ordered]
    return {'longest': [ids[-microbatch:]] * count,
            'random': [[ids[j] for j in rng.sample(range(len(ids)), microbatch)] for _ in range(count)],
            'grouped': [ids[start:start + microbatch] for start in
                        (rng.randrange(0, len(ids) - microbatch) for _ in range(count))]}


def collate(tokens_path, sample_ids, pad_id, device):
    import torch
    from .sft import InstructionCollator
    with sqlite3.connect(tokens_path) as db:
        items = []
        for key in sample_ids:
            text, prefix = db.execute('SELECT input_ids,prompt_length FROM samples WHERE id=?', (key,)).fetchone()
            ids = json.loads(text)
            items.append({'input_ids': ids, 'labels': [-100] * prefix + ids[prefix:]})
    batch = InstructionCollator(pad_id)(items)
    real = int(batch['attention_mask'].sum())
    return {k: v.to(device) for k, v in batch.items()}, real


def run_updates(model, optimizer, batches, device, warmup, measured):
    """Return samples/s, real tokens/s, padded tokens/s and peak allocated bytes for the measured updates."""
    import torch
    cuda = device.type == 'cuda'
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    elapsed, samples, real_tokens, padded = 0.0, 0, 0, 0
    for index, (batch, real) in enumerate(batches[:warmup + measured]):
        if cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda):
            loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite profiling loss')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.synchronize(device)
        if index >= warmup:
            elapsed += time.perf_counter() - start
            samples += batch['input_ids'].shape[0]
            real_tokens += real
            padded += batch['input_ids'].numel()
    return {'samples_per_second': samples / elapsed, 'real_tokens_per_second': real_tokens / elapsed,
            'padded_tokens_per_second': padded / elapsed, 'padding_fraction': 1 - real_tokens / padded,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if cuda else 0}


def sweep(model, tokens_path, lengths, settings, pad_id, device):
    import torch
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-6)
    total = torch.cuda.get_device_properties(device).total_memory if device.type == 'cuda' else None
    rows = []
    for checkpointing in settings['checkpointing']:
        (model.gradient_checkpointing_enable if checkpointing else model.gradient_checkpointing_disable)()
        for microbatch in settings['microbatches']:
            row = {'microbatch': microbatch, 'gradient_checkpointing': checkpointing}
            try:
                groups = make_batches(tokens_path, lengths, microbatch, settings['warmup_updates'] + settings['measured_updates'], settings['seed'])
                prepared = {name: [collate(tokens_path, ids, pad_id, device) for ids in batches] for name, batches in groups.items()}
                for name in ('longest', 'random', 'grouped'):
                    row[name] = run_updates(model, optimizer, prepared[name], device, settings['warmup_updates'], settings['measured_updates'])
                row['eligible'] = total is None or row['longest']['peak_allocated_bytes'] <= settings['memory_fraction'] * total
            except RuntimeError as error:
                if 'out of memory' not in str(error).lower():
                    raise
                row.update(eligible=False, error='CUDA out of memory')
            optimizer.zero_grad(set_to_none=True)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(json.dumps(row), flush=True)
            rows.append(row)
    return rows


def recommend(rows):
    """Best eligible setting by samples/s under each sampling mode (peak memory judged on the longest batches)."""
    eligible = [r for r in rows if r.get('eligible')]
    result = {}
    for label, key in (('with_group_by_length', 'grouped'), ('with_random_sampling', 'random')):
        best = max(eligible, key=lambda r: r[key]['samples_per_second'], default=None)
        result['best_' + label] = None if best is None else {
            'microbatch': best['microbatch'], 'gradient_checkpointing': best['gradient_checkpointing'],
            'samples_per_second_per_gpu': best[key]['samples_per_second'],
            'padding_fraction': best[key]['padding_fraction'],
            'longest_batch_peak_gib': best['longest']['peak_allocated_bytes'] / 2**30}
    return result


def run_sft_profile(config, settings):
    require_container()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .instruction_data import prepare_sft_tokens
    seed_all(config.get('seed', 42))
    device = torch.device('cuda')
    sft = config['sft']
    output = Path(sft['output'])
    alignment = load_checkpoint(Path(config['alignment']['output']) / 'alignment.pt')
    tokenizer = AutoTokenizer.from_pretrained(Path(config['alignment']['output']) / 'llm_tokenizer')
    tokens = prepare_sft_tokens(config, tokenizer, alignment)
    lengths = load_lengths(tokens)
    model = AutoModelForCausalLM.from_pretrained(alignment['model_name'], revision=alignment['model_revision'],
                                                 torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(device)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=sft['lora_rank'], lora_alpha=sft['lora_alpha'], target_modules=sft['lora_targets'],
                                             lora_dropout=sft['lora_dropout'], bias='none', task_type='CAUSAL_LM'))
    model.enable_input_require_grads()
    model.train()
    values = [n for _, n in lengths]
    rows = sweep(model, tokens, lengths, {**settings, 'seed': config.get('seed', 42)}, tokenizer.pad_token_id, device)
    report = {'gpu': torch.cuda.get_device_name(device), 'total_memory_bytes': torch.cuda.get_device_properties(device).total_memory,
              'train_samples': len(lengths), 'length_tokens': {'median': statistics.median(values), 'p99': sorted(values)[int(0.99 * (len(values) - 1))], 'max': max(values)},
              'settings': settings, 'rows': rows, 'recommendation': recommend(rows),
              'note': 'Single-GPU, one process, LoRA updates with synthetic optimizer lr; DDP adds only a LoRA-gradient all-reduce. '
                      'Peak memory is from the longest batches; throughput from random and length-grouped typical batches.'}
    write_json(output / 'sft-profile.json', report)
    print(json.dumps({'recommendation': report['recommendation'], 'written': str(output / 'sft-profile.json')}), flush=True)
