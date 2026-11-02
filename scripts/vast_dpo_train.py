"""DPO on RFT candidates. Launch with torchrun, one process per GPU (see configs/dpo-r1.json)."""
import argparse
import json
import random
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from cadtokenizer.common import load_checkpoint, read_config, require_container, write_json
from cadtokenizer.dpo import completion_logps, dpo_loss, preference_pairs
from cadtokenizer.rft import training_generation_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    require_container()
    config = read_config(args.config)
    dpo = config['dpo']
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device('cuda', rank % torch.cuda.device_count())
    torch.cuda.set_device(device)
    output = Path(dpo['output'])
    pairs_path = output / 'pairs.jsonl'
    alignment = load_checkpoint(Path(config['alignment']['output']) / 'alignment.pt')
    tokenizer = AutoTokenizer.from_pretrained(Path(config['alignment']['output']) / 'llm_tokenizer')
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if not pairs_path.exists():
            targets = {key: sequence for key, _, sequence in training_generation_rows(config['data']['database'])}
            pairs, stats = preference_pairs(dpo['source'], tokenizer, alignment['boundaries'], targets, dpo['loop_limit'],
                                            dpo['per_prompt'], dpo['seed'], dpo['chosen_min'], dpo['rejected_max'])
            with pairs_path.open('w') as handle:
                for pair in pairs:
                    handle.write(json.dumps(pair) + '\n')
            write_json(output / 'pairs_report.json', stats | {'pairs': len(pairs)})
        shutil.copyfile('artifacts/sft-vastai-paper-coverage-tok2-v1/run.json', output / 'run.json')
    dist.barrier()
    pairs = [json.loads(line) for line in pairs_path.read_text().splitlines()]
    context = config['sft']['context_length']
    pairs = [p for p in pairs if len(p['prompt']) + max(len(p['chosen']), len(p['rejected'])) <= context]
    per_step = world * dpo['micro']
    pairs = pairs[:len(pairs) // per_step * per_step]
    shard = pairs[rank::world]

    base = AutoModelForCausalLM.from_pretrained(config['alignment']['model_name'], revision=config['alignment']['revision'],
                                                torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(device)
    model = PeftModel.from_pretrained(base, dpo['adapter'], is_trainable=True).to(device)
    model.config.use_cache = False
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    model.eval()
    reference = []
    with torch.no_grad():
        for start in range(0, len(shard), dpo['micro']):
            batch = shard[start:start + dpo['micro']]
            prompts = [p['prompt'] for p in batch] * 2
            completions = [p['chosen'] for p in batch] + [p['rejected'] for p in batch]
            logps = completion_logps(model, prompts, completions, pad, device)
            reference.append(logps.view(2, -1).cpu())
    reference = torch.cat(reference, dim=1)

    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=dpo['lr'], weight_decay=0.0)
    order = list(range(len(shard)))
    random.Random(dpo['seed'] + rank).shuffle(order)
    steps = len(order) // dpo['micro'] * dpo['epochs']
    log = output / 'dpo_log.jsonl'
    for step in range(steps):
        index = [order[(step * dpo['micro'] + k) % len(order)] for k in range(dpo['micro'])]
        batch = [shard[i] for i in index]
        prompts = [p['prompt'] for p in batch] * 2
        completions = [p['chosen'] for p in batch] + [p['rejected'] for p in batch]
        logps = completion_logps(model, prompts, completions, pad, device).view(2, -1)
        ref = reference[:, index].to(device)
        loss, accuracy, margin = dpo_loss(logps[0], logps[1], ref[0], ref[1], dpo['beta'])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for parameter in trainable:
            dist.all_reduce(parameter.grad)
            parameter.grad /= world
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if rank == 0 and step % 10 == 0:
            with log.open('a') as handle:
                handle.write(json.dumps({'step': step, 'loss': loss.item(), 'accuracy': accuracy.item(), 'margin': margin.item()}) + '\n')
            print(json.dumps({'step': step, 'steps': steps, 'loss': round(loss.item(), 4), 'accuracy': accuracy.item()}), flush=True)
    dist.barrier()
    if rank == 0:
        model.save_pretrained(output / 'final_adapter')
        print(json.dumps({'dpo_complete': str(output)}), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
