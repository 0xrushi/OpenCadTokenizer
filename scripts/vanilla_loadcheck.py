"""Check that the step-3,056 adapter loads correctly: loss on training batches should be near the logged 0.07 to 0.08."""
import sys
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vanilla_train import Encoded, build, collate  # noqa: E402
from cadtokenizer.common import read_config, require_container  # noqa: E402


def batch_loss(model, batch, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        return model(**batch).loss.item()


def main():
    require_container()
    config = read_config('configs/vanilla-text.json')
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], revision=config['revision'])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    train, _, _ = build(config, tokenizer)
    device = torch.device('cuda:0')
    base = AutoModelForCausalLM.from_pretrained(config['model_name'], revision=config['revision'], torch_dtype=torch.bfloat16,
                                                attn_implementation='sdpa').to(device)
    model = PeftModel.from_pretrained(base, 'artifacts/sft-vanilla-text/checkpoints/checkpoint-3056', is_trainable=False).eval()
    weights = load_file('artifacts/sft-vanilla-text/checkpoints/checkpoint-3056/adapter_model.safetensors')
    loaded = dict(model.named_parameters())
    matched = sum(1 for name in weights if any(name.split('.lora_')[0] in k and 'lora' in k for k in loaded))
    print({'adapter_tensors': len(weights), 'lora_params_in_model': sum('lora' in k for k in loaded)}, flush=True)
    losses = []
    for start in range(0, 64, 4):
        losses.append(batch_loss(model, collate(train[start:start + 4], pad_id), device))
    print({'loss_mean_first_64_train_rows': round(sum(losses) / len(losses), 4), 'matched_names': matched}, flush=True)


if __name__ == '__main__':
    main()
