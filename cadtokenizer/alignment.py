from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .common import device_for, file_hash, load_checkpoint, require_container, save_checkpoint, seed_all, write_json
from .matching import unique_assignment
from .vq import load_vq


class Adapters(nn.Module):
    def __init__(self, vq_width, llm_width):
        super().__init__()
        self.forward_adapter = nn.Linear(vq_width, llm_width)
        self.backward_adapter = nn.Linear(llm_width, vq_width)


def boundary_ids(tokenizer):
    candidates = sorted(
        (i for name, i in tokenizer.get_vocab().items() if 'reserved_special_token_' in name),
    )
    excluded = {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}
    candidates = [i for i in candidates if i not in excluded]
    if len(candidates) < 3:
        raise ValueError('Backbone needs at least three existing reserved_special_token_* IDs; no vocabulary expansion is performed')
    return dict(zip(('start', 'pair_end', 'end'), candidates[:3]))


TRAILER_BINS = 64


def allocate_trailer_ids(blocked, native_ids, vocabulary_size, count=TRAILER_BINS, eligible=None):
    """Deterministic native IDs for scale/origin trailer bins.

    Highest-numbered eligible IDs are taken (rarest tokens, least likely to collide with prompt text).
    They are disjoint from blocked IDs and from every code ID. `eligible` (boolean per vocabulary ID)
    must exclude tokens with untrained input/output embeddings: the SFT LoRA does not touch embeddings,
    so bins that share a zero embedding and identical output rows could never be told apart.
    """
    used = set(blocked) | set(int(i) for i in native_ids)
    free = [i for i in range(vocabulary_size - 1, -1, -1) if i not in used and (eligible is None or bool(eligible[i]))]
    if len(free) < count:
        raise ValueError('Not enough unused native vocabulary IDs for trailer bins')
    return free[:count]


def trained_embedding_mask(input_weight, output_weight, threshold=0.25):
    """True where both the input embedding and output-head row have a norm comparable to the median."""
    import torch
    rows = min(len(input_weight), len(output_weight))
    ok = []
    for weight in (input_weight, output_weight):
        norms = torch.cat([chunk.float().norm(dim=1) for chunk in weight[:rows].split(8192)])
        ok.append(norms >= threshold * norms.median())
    return (ok[0] & ok[1]).cpu()


def train_alignment(config, resume=None):
    require_container()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    seed_all(config.get('seed', 42))
    device = device_for(config)
    settings = config['alignment']
    if settings['collision_policy'] not in ('error', 'unique_topk'):
        raise ValueError('collision_policy must be error or unique_topk')
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'alignment.pt').exists():
        raise FileExistsError('Completed alignment exists; choose a new output directory to retrain')
    if (output / 'last.pt').exists() and not resume:
        raise FileExistsError('Alignment run exists; use --resume or choose a new output directory')
    vq_path = Path(config['vq']['output']) / 'best.pt'
    vq_hash = file_hash(vq_path)
    vq, _ = load_vq(vq_path, device)
    vectors = vq.codebook.weight.detach().float().clone()
    del vq
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(settings['model_name'], revision=settings['revision'])
    if tokenizer.eos_token_id is None:
        raise ValueError('LLM tokenizer requires EOS')
    tokenizer.pad_token = tokenizer.eos_token
    boundaries = boundary_ids(tokenizer)
    backbone = AutoModelForCausalLM.from_pretrained(settings['model_name'], revision=settings['revision'], torch_dtype=dtype, attn_implementation='eager')
    resolved_revision = getattr(backbone.config, '_commit_hash', None) or settings['revision']
    embedding = backbone.get_input_embeddings().to(device).requires_grad_(False)
    head = backbone.get_output_embeddings().to(device).requires_grad_(False)
    del backbone
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    blocked = sorted(set(tokenizer.all_special_ids) | set(boundaries.values()))
    blocked = [i for i in blocked if i < head.weight.shape[0]]
    if head.weight.shape[0] - len(blocked) < len(vectors):
        raise ValueError('Not enough native vocabulary IDs for an injective code mapping')
    adapters = Adapters(vectors.shape[1], embedding.weight.shape[1]).to(device)
    optimizer = torch.optim.AdamW(adapters.parameters(), lr=settings['learning_rate'])
    order = torch.randperm(len(vectors), device=device)
    val_count = max(1, int(len(order) * settings['validation_fraction']))
    train_ids, val_ids = order[val_count:], order[:val_count]
    if not len(train_ids):
        raise ValueError('Codebook too small for alignment split')

    def select(projected):
        with torch.no_grad():
            logits = head(projected.to(head.weight.dtype)).float()
            logits[:, blocked] = -torch.inf
            return logits.argmax(-1)

    def objective(indices, fixed=None):
        raw = vectors[indices]
        projected = adapters.forward_adapter(raw)
        native = select(projected) if fixed is None else fixed[indices]
        native_embeddings = embedding(native).float().detach()
        reconstructed = adapters.backward_adapter(native_embeddings)
        # Argmax is discrete. The second term explicitly trains the forward adapter;
        # no unspecified straight-through estimator is hidden in this implementation.
        return F.mse_loss(reconstructed, raw) + F.mse_loss(projected, native_embeddings)

    start, best, stale = 0, float('inf'), 0
    if resume:
        state = load_checkpoint(resume)
        if state['config'] != config or state['vq_sha256'] != vq_hash:
            raise ValueError('Alignment resume mismatch')
        adapters.load_state_dict(state['adapters'])
        optimizer.load_state_dict(state['optimizer'])
        start, best, stale = state['epoch'] + 1, state['best_loss'], state['stale']
        train_ids, val_ids = state['train_ids'].to(device), state['val_ids'].to(device)
        torch.set_rng_state(state['rng'])
        if device.type == 'cuda' and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])

    def batches(indices):
        return indices.split(settings['batch_size'])

    for epoch in range(start, settings['epochs']):
        adapters.train()
        total = 0.0
        for indices in batches(train_ids[torch.randperm(len(train_ids), device=device)]):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(indices)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite alignment loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in adapters.parameters()):
                raise RuntimeError('Missing or non-finite adapter gradients')
            nn.utils.clip_grad_norm_(adapters.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        adapters.eval()
        with torch.no_grad():
            score = sum(float(objective(ids)) * len(ids) for ids in batches(val_ids)) / len(val_ids)
        improved = score < best
        best, stale = (score, 0) if improved else (best, stale + 1)
        metrics = {'epoch': epoch + 1, 'train_loss': total / len(train_ids), 'validation_loss': score}
        print(json.dumps(metrics), flush=True)
        with (output / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(metrics) + '\n')
        state = {'config': config, 'vq_sha256': vq_hash, 'adapters': adapters.state_dict(),
                 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'best_loss': best, 'stale': stale,
                 'train_ids': train_ids.cpu(), 'val_ids': val_ids.cpu(), 'rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}
        save_checkpoint(output / 'last.pt', state)
        if improved:
            save_checkpoint(output / 'best_adapters.pt', {'adapters': adapters.state_dict()})
        if stale >= settings['patience']:
            break
    adapters.load_state_dict(load_checkpoint(output / 'best_adapters.pt')['adapters'])
    adapters.eval()
    with torch.no_grad():
        original = torch.cat([select(adapters.forward_adapter(batch)) for batch in vectors.split(settings['batch_size'])])
    collisions = len(original) - len(set(original.tolist()))
    native_ids = original
    if collisions:
        if settings['collision_policy'] == 'error':
            raise ValueError(f'{collisions} code collisions. Select collision_policy=unique_topk explicitly to resolve; this is a reproduction extension.')
        if settings['collision_policy'] != 'unique_topk':
            raise ValueError('Unknown collision policy')
        available = head.weight.shape[0] - len(blocked)
        k = min(settings['candidate_count'], available)
        if k < 1:
            raise ValueError('candidate_count must be positive')
        while True:
            with torch.no_grad():
                candidate_ids, candidate_scores = [], []
                for batch in vectors.split(settings['batch_size']):
                    scores = head(adapters.forward_adapter(batch).to(head.weight.dtype)).float()
                    scores[:, blocked] = -torch.inf
                    top = scores.topk(k, dim=-1)
                    candidate_ids.extend(top.indices.cpu().tolist())
                    candidate_scores.extend(top.values.cpu().tolist())
            try:
                assigned = unique_assignment(candidate_ids, candidate_scores)
                native_ids = torch.tensor(assigned, device=device)
                break
            except ValueError:
                if k >= available:
                    raise
                k = min(k * 2, available)
                print(json.dumps({'expanded_assignment_candidates': k}), flush=True)
        for epoch in range(settings['refinement_epochs']):
            adapters.train()
            for indices in batches(train_ids[torch.randperm(len(train_ids), device=device)]):
                optimizer.zero_grad(set_to_none=True)
                loss = objective(indices, fixed=native_ids)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite fixed-mapping refinement loss')
                loss.backward()
                nn.utils.clip_grad_norm_(adapters.parameters(), 1.0)
                optimizer.step()
            print(json.dumps({'refinement_epoch': epoch + 1}), flush=True)
    adapters.eval()
    with torch.no_grad():
        reconstructed = adapters.backward_adapter(embedding(native_ids).float()).cpu()
        final_val = sum(float(objective(ids, native_ids)) * len(ids) for ids in batches(val_ids)) / len(val_ids)
    tokenizer.save_pretrained(output / 'llm_tokenizer')
    trailer_ids = None
    if settings.get('trailer'):
        reserved = [i for name, i in tokenizer.get_vocab().items() if 'reserved_special_token_' in name]
        trailer_ids = allocate_trailer_ids(sorted(set(blocked) | set(reserved)), native_ids.tolist(), head.weight.shape[0],
                                           eligible=trained_embedding_mask(embedding.weight, head.weight))
        for name, weight in (('input embedding', embedding.weight), ('output head', head.weight)):
            rows = weight[torch.tensor(trailer_ids, device=weight.device)].float()
            gaps = torch.cdist(rows, rows) + torch.eye(len(rows), device=rows.device) * 1e9
            if float(gaps.min()) < 1e-3:
                raise ValueError(f'Trailer IDs have indistinguishable {name} rows')
    report = {'initial_collisions': collisions, 'collision_policy': settings['collision_policy'],
              'reassigned_codes': int((native_ids != original).sum()), 'final_validation_loss': final_val,
              'boundaries': boundaries, 'model_name': settings['model_name'], 'model_revision': resolved_revision,
              'vq_sha256': vq_hash, 'trailer_ids': trailer_ids}
    state = {'config': config, **report, 'adapters': adapters.state_dict(), 'native_ids': native_ids.cpu(),
             'reconstructed_code_vectors': reconstructed}
    save_checkpoint(output / 'alignment.pt', state)
    write_json(output / 'report.json', report)
    print(json.dumps(report, indent=2), flush=True)
