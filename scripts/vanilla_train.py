"""Vanilla-LLaMA control SFT: native Llama tokenizer, CAD written as text. Launch with torchrun (4 GPUs)."""
import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments

from cadtokenizer.common import read_config, require_container, write_json
from cadtokenizer.vanilla import encode_example, example_rows, selected_training_keys


class Encoded(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate(batch, pad_id):
    width = max(len(item['input_ids']) for item in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, item in enumerate(batch):
        length = len(item['input_ids'])
        ids[row, :length] = torch.tensor(item['input_ids'])
        labels[row, :length] = torch.tensor(item['labels'])
        mask[row, :length] = 1
    return {'input_ids': ids, 'labels': labels, 'attention_mask': mask}


def build(config, tokenizer):
    keys = selected_training_keys(config['database'], config['seed'], config['editing_per_generation'])
    train, dropped = [], 0
    for _, task, instruction, source, target in example_rows(config['database'], 'train', keys):
        item = encode_example(tokenizer, task, instruction, source, target, config['context_length'])
        if item is None:
            dropped += 1
        else:
            train.append(item)
    valid = []
    for _, task, instruction, source, target in example_rows(config['database'], 'val'):
        item = encode_example(tokenizer, task, instruction, source, target, config['context_length'])
        if item is not None:
            valid.append(item)
    return train, valid, dropped


class Checked(Trainer):
    """Trainer that stops at the first non-finite forward pass and saves that batch for diagnosis."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        if not (torch.isfinite(loss).item() and torch.isfinite(outputs.logits).all().item()):
            if trainer_rank() == 0:
                directory = Path(self.args.output_dir).parent
                directory.mkdir(parents=True, exist_ok=True)
                torch.save({k: v.cpu() for k, v in inputs.items()}, directory / 'bad_batch.pt')
                report = {'step': int(self.state.global_step), 'loss': str(loss.item()),
                          'sample_lengths': inputs['attention_mask'].sum(dim=1).tolist(),
                          'target_tokens': (inputs['labels'] != -100).sum(dim=1).tolist()}
                (directory / 'nonfinite_report.json').write_text(json.dumps(report, indent=2))
                print(json.dumps(report), flush=True)
            raise RuntimeError('non-finite forward pass; batch saved')
        return (loss, outputs) if return_outputs else loss


def trainer_rank():
    import os
    return int(os.environ.get('RANK', 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    require_container()
    config = read_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], revision=config['revision'])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    train, valid, dropped = build(config, tokenizer)
    output = Path(config['output'])
    if trainer_rank() == 0:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / 'data_report.json', {'train_rows': len(train), 'train_dropped_over_context': dropped, 'valid_rows': len(valid),
                                                'context_length': config['context_length']})
    model = AutoModelForCausalLM.from_pretrained(config['model_name'], revision=config['revision'], torch_dtype=torch.bfloat16,
                                                 attn_implementation='sdpa')
    model.config.use_cache = False
    lora = LoraConfig(r=config['lora_rank'], lora_alpha=config['lora_alpha'], target_modules=config['lora_targets'],
                      lora_dropout=config['lora_dropout'], bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, lora)
    import inspect
    wanted = dict(
        output_dir=str(output / 'checkpoints'), per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'], gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'], weight_decay=config['weight_decay'], lr_scheduler_type='cosine',
        warmup_steps=round(config['warmup_ratio'] * config['max_steps']), max_steps=config['max_steps'], num_train_epochs=20,
        eval_strategy='steps', eval_steps=config['eval_steps'], save_strategy='steps', save_steps=config['eval_steps'],
        save_total_limit=6, load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
        logging_steps=10, bf16=True, optim='adamw_torch', remove_unused_columns=False, report_to=['tensorboard'],
        dataloader_num_workers=2, seed=config['seed'], gradient_checkpointing=config['gradient_checkpointing'],
        gradient_checkpointing_kwargs={'use_reentrant': False})
    supported = inspect.signature(TrainingArguments).parameters
    dropped = sorted(key for key in wanted if key not in supported)
    if dropped and trainer_rank() == 0:
        print(json.dumps({'dropped_training_arguments': dropped}), flush=True)
    arguments = TrainingArguments(**{key: value for key, value in wanted.items() if key in supported})
    trainer = Checked(model=model, args=arguments, train_dataset=Encoded(train), eval_dataset=Encoded(valid),
                      data_collator=lambda batch: collate(batch, pad_id),
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=config['early_stopping_patience'])])
    trainer.train(resume_from_checkpoint=args.resume)
    if trainer_rank() == 0:
        trainer.save_model(str(output / 'final_adapter'))
        print(json.dumps({'vanilla_complete': str(output)}), flush=True)


if __name__ == '__main__':
    main()
