"""Replay vanilla SFT from checkpoint-3056 with per-step finiteness checks, to locate the NaN near step 3,400.

On the first non-finite forward pass, rank 0 writes the batch and per-sample lengths, then stops.
"""
import json
import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vanilla_train import Encoded, build, collate  # noqa: E402
from cadtokenizer.common import read_config, require_container  # noqa: E402


class Checked(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        logits = outputs.logits
        loss = outputs.loss
        if not (torch.isfinite(logits).all() and torch.isfinite(loss)):
            if int(os.environ.get('RANK', 0)) == 0:
                labels = inputs['labels']
                report = {'step': int(self.state.global_step),
                          'loss': float(loss.item()) if torch.isfinite(loss) else str(loss.item()),
                          'nonfinite_logits': int((~torch.isfinite(logits)).sum().item()),
                          'batch_shape': list(inputs['input_ids'].shape),
                          'sample_lengths': inputs['attention_mask'].sum(dim=1).tolist(),
                          'target_tokens_per_sample': (labels != -100).sum(dim=1).tolist(),
                          'logits_abs_max_finite': float(logits[torch.isfinite(logits)].abs().max().item())}
                Path('artifacts/vanilla-diagnose').mkdir(parents=True, exist_ok=True)
                torch.save({k: v.cpu() for k, v in inputs.items()}, 'artifacts/vanilla-diagnose/bad_batch.pt')
                Path('artifacts/vanilla-diagnose/report.json').write_text(json.dumps(report, indent=2))
                print(json.dumps(report), flush=True)
            raise RuntimeError('non-finite forward pass; batch saved')
        return (loss, outputs) if return_outputs else loss


def main():
    require_container()
    config = read_config('configs/vanilla-text.json')
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], revision=config['revision'])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    train, valid, _ = build(config, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(config['model_name'], revision=config['revision'], torch_dtype=torch.bfloat16,
                                                 attn_implementation='sdpa')
    model.config.use_cache = False
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, 'artifacts/sft-vanilla-text/checkpoints/checkpoint-3056', is_trainable=True)
    arguments = TrainingArguments(
        output_dir='artifacts/vanilla-diagnose/checkpoints', per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'], gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'], weight_decay=config['weight_decay'], lr_scheduler_type='cosine',
        warmup_steps=round(config['warmup_ratio'] * config['max_steps']), max_steps=config['max_steps'], num_train_epochs=20,
        logging_steps=10, bf16=True, optim='adamw_torch', remove_unused_columns=False, report_to=[],
        dataloader_num_workers=2, seed=config['seed'], gradient_checkpointing=config['gradient_checkpointing'],
        gradient_checkpointing_kwargs={'use_reentrant': False}, save_strategy='no', eval_strategy='no')
    print(json.dumps({'trainable_tensors': sum(p.requires_grad for p in model.parameters())}), flush=True)
    trainer = Checked(model=model, args=arguments, train_dataset=Encoded(train), data_collator=lambda b: collate(b, pad_id))
    trainer.train(resume_from_checkpoint='artifacts/sft-vanilla-text/checkpoints/checkpoint-3056')


if __name__ == '__main__':
    main()
