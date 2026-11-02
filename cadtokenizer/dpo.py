"""Direct preference optimization (DPO) on kernel-labelled candidates from the RFT sampler.

Preferred completions are kernel-valid, structure-matched candidates. Rejected completions are
candidates the kernel rejects, or valid candidates with retraced edges or out-of-range loop counts,
for the same training prompt. Completions are native IDs plus the end and EOS tokens.
"""
import glob
import json
import random
from pathlib import Path

from .bench_metrics import count_f1
from .instruction_data import instruction_tokens
from .rft import accept, structure


def primitive_fidelity(reference, prediction):
    """Mean count-based F1 over line, arc and circle counts (1.0 when both have none)."""
    def counts(sequence):
        result = {'line': 0, 'arc': 0, 'circle': 0}
        for token in sequence.split():
            name = token.split(',')[0]
            if name in result:
                result[name] += 1
        return result
    a, b = counts(reference), counts(prediction)
    return sum(count_f1(a[k], b[k], absent=1.0) for k in a) / len(a)


def preference_pairs(source, tokenizer, boundaries, targets, loop_limit, per_prompt, seed, chosen_min, rejected_max):
    """List of {prompt, chosen, rejected} token lists, at most `per_prompt` pairs per prompt."""
    end, eos = boundaries['end'], tokenizer.eos_token_id
    rng = random.Random(seed)
    pairs, stats = [], {'prompts': 0, 'with_pair': 0}
    for path in sorted(glob.glob(str(Path(source) / 'shards' / 'candidates-*.jsonl'))):
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            stats['prompts'] += 1
            chosen, rejected = {}, {}
            reference = targets[record['sample_id']]
            for candidate in record['candidates']:
                if candidate['status'] != 'syntax_valid':
                    continue
                sequence, ids = candidate['sequence'], candidate['native_ids']
                if candidate['kernel'] != 'valid' or structure(sequence)[1] > 0 or structure(sequence)[0] > loop_limit:
                    rejected.setdefault(sequence, ids)
                    continue
                fidelity = primitive_fidelity(reference, sequence)
                if accept(sequence, ids, end, True, loop_limit) and fidelity >= chosen_min:
                    chosen.setdefault(sequence, ids)
                elif fidelity < rejected_max:
                    rejected.setdefault(sequence, ids)
            if not chosen or not rejected:
                continue
            stats['with_pair'] += 1
            chosen_list, rejected_list = list(chosen.values()), list(rejected.values())
            rng.shuffle(chosen_list)
            rng.shuffle(rejected_list)
            prompt = instruction_tokens(tokenizer, boundaries, record['instruction'])
            for index in range(min(per_prompt, len(chosen_list), len(rejected_list))):
                pairs.append({'prompt': prompt, 'chosen': chosen_list[index] + [eos], 'rejected': rejected_list[index] + [eos]})
    return pairs, stats


def completion_logps(model, prompts, completions, pad_id, device):
    """Summed log-probability of each completion given its prompt. Right-padded batch."""
    import torch
    sequences = [p + c for p, c in zip(prompts, completions)]
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    targets = torch.zeros_like(ids, dtype=torch.bool)
    for row, (prompt, completion) in enumerate(zip(prompts, completions)):
        ids[row, :len(prompt) + len(completion)] = torch.tensor(prompt + completion)
        mask[row, :len(prompt) + len(completion)] = 1
        targets[row, len(prompt):len(prompt) + len(completion)] = True
    ids, mask, targets = ids.to(device), mask.to(device), targets.to(device)
    import torch.nn.functional as F
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1]
    selected = targets[:, 1:]
    rows = selected.nonzero(as_tuple=True)[0]
    logp = -F.cross_entropy(logits[selected].float(), ids[:, 1:][selected], reduction='none')
    return torch.zeros(len(sequences), device=device, dtype=logp.dtype).index_add(0, rows, logp)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    import torch.nn.functional as F
    margin = beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))
    return -F.logsigmoid(margin).mean(), (margin > 0).float().mean(), margin.mean()
