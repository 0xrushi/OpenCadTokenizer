from __future__ import annotations

import json
import math
import sqlite3
import time
from itertools import islice
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from .common import device_for, file_hash, load_checkpoint, require_container, save_checkpoint, seed_all, write_json
from .representation import BOS, CURVES, EOS, PAD, VOCAB_SIZE, Grammar

LOSS_NORMALIZATION = 'tokens-and-primitives-v1'


def origin_ordinal_loss(logits, targets):
    """Sum expected squared normalized bin error at ox/oy only.

    Targets contain one SEM pair plus EOS per row. The origin slots are
    boolean+16 and boolean+17. CE still supervises the entire vocabulary;
    this auxiliary term conditions its distances on the 64 legal numbers.
    """
    operation = (targets >= 8) & (targets <= 10)
    if not bool((operation.sum(dim=1) == 1).all()):
        raise ValueError('Origin loss requires exactly one extrusion per row')
    boundary = operation.long().argmax(dim=1)
    positions = boundary[:, None] + torch.tensor([16, 17], device=targets.device)
    rows = torch.arange(targets.shape[0], device=targets.device)[:, None]
    truth = targets[rows, positions] - 13
    if not bool(((truth >= 0) & (truth < 64)).all()):
        raise ValueError('Origin target is not a coordinate bin')
    probabilities = logits[rows, positions, 13:77].float().softmax(-1)
    bins = torch.arange(64, device=logits.device, dtype=torch.float32)
    return (probabilities * ((bins-truth[..., None])/63).square()).sum()


def accumulation_windows(loader, count):
    iterator = iter(loader)
    while window := list(islice(iterator, count)):
        yield window


def accumulated_loss(result, window_tokens, window_primitives):
    """Match a full window's token-mean reconstruction and primitive-mean VQ loss."""
    return (result['reconstruction'] * result['token_count'] / window_tokens
            + result['vq_loss'] * result['indices'].numel() / window_primitives)


class PairDataset(Dataset):
    def __init__(self, path, tokenizer_split=None, limit=None):
        self.path, self.connection = str(path), None
        with sqlite3.connect(path) as db:
            query = 'SELECT p.id FROM pairs p JOIN geometry g ON g.id=p.geometry_id'
            args = []
            if tokenizer_split is not None:
                query += ' WHERE g.tokenizer_candidate=1 AND g.tokenizer_split=?'
                args.append(tokenizer_split)
            query += ' ORDER BY p.geometry_id,p.ordinal'
            if limit:
                query += ' LIMIT ?'
                args.append(limit)
            self.ids = [r[0] for r in db.execute(query, args)]
        if not self.ids:
            raise ValueError(f'No pairs for {tokenizer_split or "all"}')

    def __len__(self):
        return len(self.ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['connection'] = None
        return state

    def __getitem__(self, index):
        if self.connection is None:
            self.connection = sqlite3.connect(self.path)
        row = self.connection.execute('SELECT geometry_id,ordinal,tokens,groups FROM pairs WHERE id=?', (self.ids[index],)).fetchone()
        return {'geometry_id': row[0], 'ordinal': row[1], 'tokens': json.loads(row[2]), 'groups': json.loads(row[3])}


def collate_pairs(items):
    length = max(len(x['tokens']) for x in items)
    tokens = torch.full((len(items), length), PAD, dtype=torch.long)
    groups = torch.full_like(tokens, -1)
    targets = torch.full((len(items), length + 1), PAD, dtype=torch.long)
    decoder_input = torch.full_like(targets, PAD)
    decoder_input[:, 0] = BOS
    for i, item in enumerate(items):
        n = len(item['tokens'])
        tokens[i, :n] = torch.tensor(item['tokens'])
        groups[i, :n] = torch.tensor(item['groups'])
        targets[i, :n] = tokens[i, :n]
        targets[i, n] = EOS
        decoder_input[i, 1:n+1] = tokens[i, :n]
    return {'tokens': tokens, 'groups': groups, 'targets': targets, 'decoder_input': decoder_input}


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


class PrimitiveVQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        vq, data = config['vq'], config['data']
        width = vq['width']
        self.embedding = nn.Embedding(VOCAB_SIZE, width, padding_idx=PAD)
        self.positions = nn.Embedding(data['max_pair_tokens'] + 2, width)
        self.primitive_positions = nn.Embedding(data['max_primitives'], width)
        enc = nn.TransformerEncoderLayer(width, vq['heads'], vq['feedforward'], vq['dropout'], batch_first=True, norm_first=True)
        dec = nn.TransformerDecoderLayer(width, vq['heads'], vq['feedforward'], vq['dropout'], batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, vq['encoder_layers'], norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec, vq['decoder_layers'], norm=nn.LayerNorm(width))
        self.codebook = nn.Embedding(vq['codebook_size'], width)
        self.head = nn.Linear(width, VOCAB_SIZE)

    def embed(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        return self.embedding(ids) * math.sqrt(self.embedding.embedding_dim) + self.positions(positions)

    def pool(self, tokens, groups):
        hidden = self.encoder(self.embed(tokens), src_key_padding_mask=tokens.eq(PAD))
        count = int(groups.max().item()) + 1
        mask = torch.arange(count, device=tokens.device)[None, :] <= groups.max(dim=1).values[:, None]
        pooled = hidden.new_full((tokens.shape[0], count, hidden.shape[-1]), -torch.inf)
        index = groups.clamp_min(0)[..., None].expand_as(hidden)
        values = hidden.masked_fill(groups.lt(0)[..., None], -torch.inf)
        pooled = pooled.scatter_reduce(1, index, values, reduce='amax', include_self=True)
        return pooled.masked_fill(~mask[..., None], 0), mask

    def quantize(self, pooled, mask):
        vectors = self.codebook.weight
        distance = pooled.float().square().sum(-1, keepdim=True) + vectors.float().square().sum(-1) - 2 * pooled.float() @ vectors.float().t()
        indices = distance.argmin(-1)
        quantized = self.codebook(indices)
        vq_loss = F.mse_loss(quantized[mask], pooled.detach()[mask])
        commitment = F.mse_loss(pooled[mask], quantized.detach()[mask])
        straight_through = pooled + (quantized - pooled).detach()
        return straight_through, indices, vq_loss + self.config['vq']['commitment_weight'] * commitment

    def decode(self, decoder_input, memory, memory_mask):
        length = decoder_input.shape[1]
        if memory.shape[1] > self.primitive_positions.num_embeddings:
            raise ValueError('Latent sequence exceeds the configured primitive limit')
        memory = memory + self.primitive_positions(torch.arange(memory.shape[1], device=memory.device))
        causal = torch.ones(length, length, dtype=torch.bool, device=decoder_input.device).triu(1)
        hidden = self.decoder(self.embed(decoder_input), memory, tgt_mask=causal,
                              tgt_key_padding_mask=decoder_input.eq(PAD), memory_key_padding_mask=~memory_mask)
        return self.head(hidden)

    def forward(self, tokens, groups, decoder_input, targets):
        pooled, mask = self.pool(tokens, groups)
        memory, indices, vq_loss = self.quantize(pooled, mask)
        logits = self.decode(decoder_input, memory, mask)
        active = targets.ne(PAD)
        flat_logits, flat_targets = logits[active].float(), targets[active]
        ce = F.cross_entropy(flat_logits, flat_targets)
        if self.config['vq']['reconstruction_loss'] == 'emd':
            target = F.one_hot(flat_targets, VOCAB_SIZE).float()
            reconstruction = (flat_logits.softmax(-1).cumsum(-1) - target.cumsum(-1)).square().sum(-1).mean()
        elif self.config['vq']['reconstruction_loss'] == 'cross_entropy':
            reconstruction = ce
        else:
            raise ValueError('Unknown reconstruction loss')
        origin_weight = self.config['vq'].get('origin_loss_weight', 0.0)
        if origin_weight:
            if origin_weight < 0 or self.config['vq']['reconstruction_loss'] != 'cross_entropy':
                raise ValueError('Origin auxiliary loss requires CE and a positive weight')
            reconstruction = reconstruction + origin_weight * origin_ordinal_loss(logits, targets) / active.sum()
        return {'loss': reconstruction + vq_loss, 'reconstruction': reconstruction, 'vq_loss': vq_loss,
                'ce': ce, 'correct': (flat_logits.argmax(-1) == flat_targets).sum(),
                'token_count': active.sum(), 'indices': indices[mask], 'pooled': pooled.detach()[mask]}

    @torch.no_grad()
    def encode(self, tokens, groups):
        pooled, mask = self.pool(tokens, groups)
        quantized, indices, _ = self.quantize(pooled, mask)
        return quantized, indices, mask

    @torch.no_grad()
    def reconstruct(self, memory, max_length=None, constrained=True, force_add=False):
        """One sketch–extrusion pair, with grammar enforcement in the VQ decoder.

        `force_add` additionally restricts the first operation's boolean token to
        `add`; pass it only for the first pair of a multi-pair program."""
        if memory.ndim == 2:
            memory = memory[None]
        if memory.shape[0] != 1:
            raise ValueError('Reconstruction takes one pair at a time')
        memory_mask = torch.ones(memory.shape[:2], device=memory.device, dtype=torch.bool)
        max_length = max_length or self.config['data']['max_pair_tokens'] + 1
        extrusions = 3 if self.config['data']['extrusion_partition'] == 'three' else 5
        # Slots cap curve primitives; the token budget is enforced per curve by
        # actual arity below, so long line-only sketches are not cut at 12 tokens/curve.
        curve_budget = self.config['data']['max_primitives'] - extrusions
        if curve_budget < 1 or max_length < CURVES[1] + 25:
            raise ValueError('Reconstruction budget cannot represent one curve and extrusion')
        loop_grouping = self.config['data'].get('sketch_partition', 'curve') == 'loop'
        grammar = Grammar(max_curves=(max_length - 23) // 4 if loop_grouping else curve_budget,
                          max_loops=self.config['data']['max_primitives'] - extrusions if loop_grouping else 10000,
                          force_add=force_add)
        result = [BOS]
        for _ in range(max_length):
            ids = torch.tensor([result], device=memory.device)
            logits = self.decode(ids, memory, memory_mask)[0, -1].float()
            if constrained:
                allowed = sorted(grammar.allowed())
                # A curve needs its coordinate fields, curve_end, then at
                # least loop/face/sketch ends + extrusion + EOS (23 tokens).
                # Budget by actual curve arity, not by latent slots.
                remaining = max_length - len(result) + 1
                allowed = [t for t in allowed if t not in CURVES or remaining >= CURVES[t] + 25]
                if not allowed:
                    raise ValueError('No grammar continuation fits the remaining token budget')
                forbidden = grammar.numeric_forbidden() & set(allowed)
                if len(forbidden) == len(allowed):
                    forbidden = set()  # every legal value is numerically bad; fall back unconstrained
                mask = torch.full_like(logits, -torch.inf)
                mask[[a for a in allowed if a not in forbidden]] = 0
                logits = logits + mask
            else:
                logits[PAD] = logits[BOS] = -torch.inf
            token = int(logits.argmax())
            if constrained:
                grammar.advance(token)
            if token == EOS:
                return result[1:]
            result.append(token)
        raise ValueError('CAD reconstruction reached maximum length before EOS')


def reset_dead_codes(codebook, usage, vectors, optimizer=None, noise=0.01):
    """Re-seed codes that were never selected from recent encoder outputs; returns how many were reset."""
    dead = (usage == 0).nonzero().flatten()
    if len(dead) == 0 or len(vectors) == 0:
        return 0
    choices = torch.randint(len(vectors), (len(dead),), device=vectors.device)
    with torch.no_grad():
        weight = codebook.weight
        weight[dead] = vectors[choices].to(weight.dtype) + torch.randn(len(dead), weight.shape[1], device=weight.device) * noise
    if optimizer is not None:
        for key in ('exp_avg', 'exp_avg_sq'):
            if key in optimizer.state.get(codebook.weight, {}):
                optimizer.state[codebook.weight][key][dead] = 0
    return int(len(dead))


def scheduled_learning_rate(settings, epoch):
    base = settings['learning_rate']
    if settings.get('lr_schedule') != 'cosine':
        return base
    floor = settings.get('min_learning_rate', 0.0)
    return floor + 0.5 * (base - floor) * (1 + math.cos(math.pi * epoch / settings['epochs']))


def load_vq(path, device):
    checkpoint = load_checkpoint(path)
    model = PrimitiveVQVAE(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval(), checkpoint


def train_vq(config, resume=None):
    require_container()
    seed_all(config.get('seed', 42))
    device = device_for(config)
    settings = config['vq']
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'last.pt').exists() and resume is None:
        raise FileExistsError('Existing VQ run; use --resume or choose a new output directory')
    with sqlite3.connect(config['data']['database']) as db:
        prepared = json.loads(db.execute('SELECT value FROM metadata WHERE key="data_config"').fetchone()[0])
    for key in ('max_pair_tokens', 'max_primitives', 'extrusion_partition'):
        if prepared[key] != config['data'][key]:
            raise ValueError(f'Prepared data configuration differs: {key}')
    if prepared.get('sketch_partition', 'curve') != config['data'].get('sketch_partition', 'curve'):
        raise ValueError('Prepared data configuration differs: sketch_partition')
    database_hash = file_hash(config['data']['database'])
    train = PairDataset(config['data']['database'], 'train')
    val = train if settings.get('overfit_validation', False) else PairDataset(config['data']['database'], 'val')
    train_loader = DataLoader(train, batch_size=settings['batch_size'], shuffle=True,
                              num_workers=settings['num_workers'], collate_fn=collate_pairs)
    val_loader = DataLoader(val, batch_size=settings['batch_size'], num_workers=0, collate_fn=collate_pairs)
    model = PrimitiveVQVAE(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    start, best, stale = 0, float('inf'), 0
    if resume:
        checkpoint = load_checkpoint(resume)
        schedule_keys = ('patience', 'epochs', 'epoch_checkpoints', 'epoch_checkpoint_every')
        strip = lambda c: {**c, 'vq': {k: v for k, v in c['vq'].items() if k not in schedule_keys}}
        if strip(checkpoint['config']) != strip(config) or checkpoint['database_sha256'] != database_hash:
            raise ValueError('Resume configuration/data mismatch')
        if checkpoint.get('loss_normalization') != LOSS_NORMALIZATION:
            raise ValueError('VQ checkpoint uses older loss normalization; use a fresh training output to keep scores comparable')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start, best, stale = checkpoint['epoch'] + 1, checkpoint['best_loss'], checkpoint['stale']
        torch.set_rng_state(checkpoint['rng'])
        if device.type == 'cuda' and checkpoint.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    else:
        with torch.no_grad():
            batch = to_device(next(iter(train_loader)), device)
            pooled, mask = model.pool(batch['tokens'], batch['groups'])
            values = pooled[mask]
            choices = torch.randint(len(values), (settings['codebook_size'],), device=device)
            model.codebook.weight.copy_(values[choices] + torch.randn_like(model.codebook.weight) * 0.01)
    use_amp = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    accumulation = settings.get('gradient_accumulation_steps', 1)
    if not isinstance(accumulation, int) or accumulation < 1:
        raise ValueError('VQ gradient_accumulation_steps must be a positive integer')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start, settings['epochs']):
        epoch_start = time.monotonic()
        if settings.get('lr_schedule'):
            for group in optimizer.param_groups:
                group['lr'] = scheduled_learning_rate(settings, epoch)
        reset_interval = settings.get('dead_code_reset_steps', 0)
        window_usage = torch.zeros(settings['codebook_size'], dtype=torch.long, device=device)
        windows = resets = 0
        model.train()
        train_reconstruction = train_vq_total = 0.0
        train_tokens = train_primitives = step = 0
        for window in accumulation_windows(train_loader, accumulation):
            window_tokens = sum(int(batch['targets'].ne(PAD).sum()) for batch in window)
            window_primitives = sum(int((batch['groups'].max(dim=1).values + 1).sum()) for batch in window)
            optimizer.zero_grad(set_to_none=True)
            for batch in window:
                batch = to_device(batch, device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    result = model(**batch)
                if not torch.isfinite(result['loss']):
                    raise FloatingPointError('Non-finite VQ training loss')
                accumulated_loss(result, window_tokens, window_primitives).backward()
                tokens, primitives = int(result['token_count']), result['indices'].numel()
                train_reconstruction += float(result['reconstruction'].detach()) * tokens
                train_vq_total += float(result['vq_loss'].detach()) * primitives
                train_tokens += tokens
                train_primitives += primitives
                if reset_interval:
                    window_usage += torch.bincount(result['indices'], minlength=len(window_usage))
                    recent = result['pooled'].float()
                if step % 100 == 0:
                    print(json.dumps({'epoch': epoch + 1, 'batch': step, 'loss': float(result['loss'].detach())}), flush=True)
                step += 1
            nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            windows += 1
            if reset_interval and windows % reset_interval == 0:
                resets += reset_dead_codes(model.codebook, window_usage, recent, optimizer)
                window_usage.zero_()
        model.eval()
        val_reconstruction, val_vq_total, count, correct, token_count, primitive_count = 0.0, 0.0, 0, 0, 0, 0
        usage = torch.zeros(settings['codebook_size'], dtype=torch.long, device=device)
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch, device)
                result = model(**batch)
                if not torch.isfinite(result['loss']):
                    raise FloatingPointError('Non-finite VQ validation loss')
                tokens, primitives = int(result['token_count']), result['indices'].numel()
                val_reconstruction += float(result['reconstruction']) * tokens
                val_vq_total += float(result['vq_loss']) * primitives
                count += len(batch['tokens'])
                correct += int(result['correct'])
                token_count += int(result['token_count'])
                primitive_count += primitives
                usage += torch.bincount(result['indices'], minlength=len(usage))
        score = val_reconstruction / token_count + val_vq_total / primitive_count
        improved = score < best
        best, stale = (score, 0) if improved else (best, stale + 1)
        probabilities = usage.float() / usage.sum()
        positive = probabilities[probabilities > 0]
        metrics = {'epoch': epoch + 1, 'train_loss': train_reconstruction / train_tokens + train_vq_total / train_primitives,
                   'validation_loss': score, 'loss_normalization': LOSS_NORMALIZATION,
                   'validation_token_accuracy': correct / token_count, 'used_codes': int((usage > 0).sum()),
                   'codebook_perplexity': float(torch.exp(-(positive * positive.log()).sum()))}
        if reset_interval:
            metrics.update(dead_code_resets=resets)
        if settings.get('lr_schedule'):
            metrics.update(learning_rate=optimizer.param_groups[0]['lr'])
        metrics.update(epoch_seconds=time.monotonic() - epoch_start, train_pairs=len(train), validation_pairs=len(val),
                       microbatch_size=settings['batch_size'], effective_batch_size=settings['batch_size'] * accumulation,
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                       peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0)
        print(json.dumps(metrics), flush=True)
        with (output / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(metrics) + '\n')
        checkpoint = {'config': config, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'epoch': epoch, 'best_loss': best, 'stale': stale, 'rng': torch.get_rng_state(),
                      'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                      'database_sha256': database_hash, 'loss_normalization': LOSS_NORMALIZATION}
        save_checkpoint(output / 'last.pt', checkpoint)
        if improved:
            save_checkpoint(output / 'best.pt', checkpoint)
        if settings.get('epoch_checkpoints') and ((epoch + 1) % settings.get('epoch_checkpoint_every', 1) == 0 or epoch + 1 == settings['epochs']):
            save_checkpoint(output / f'epoch-{epoch + 1:03d}.pt', checkpoint)
        if stale >= settings['patience']:
            break
    write_json(output / 'config.json', config)
