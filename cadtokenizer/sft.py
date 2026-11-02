from __future__ import annotations

import json
import inspect
import sqlite3
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .common import file_hash, load_checkpoint, require_container, seed_all, write_json
from .instruction_data import prepare_sft_tokens


class InstructionDataset(Dataset):
    def __init__(self, path, split):
        self.path, self.connection = str(path), None
        with sqlite3.connect(path) as db:
            self.ids = [x[0] for x in db.execute('SELECT id FROM samples WHERE split=? ORDER BY id', (split,))]
        if not self.ids:
            raise ValueError(f'No SFT samples in {split}')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if self.connection is None:
            self.connection = sqlite3.connect(self.path)
        text, prefix = self.connection.execute('SELECT input_ids,prompt_length FROM samples WHERE id=?', (self.ids[index],)).fetchone()
        ids = json.loads(text)
        return {'input_ids': ids, 'labels': [-100] * prefix + ids[prefix:]}


class InstructionCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, items):
        width = max(len(x['input_ids']) for x in items)
        ids = torch.full((len(items), width), self.pad_id, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        attention = torch.zeros_like(ids)
        for i, item in enumerate(items):
            n = len(item['input_ids'])
            ids[i, :n] = torch.tensor(item['input_ids'])
            labels[i, :n] = torch.tensor(item['labels'])
            attention[i, :n] = 1
        return {'input_ids': ids, 'attention_mask': attention, 'labels': labels}


def train_sft(config, resume=None):
    require_container()
    from accelerate import PartialState
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from .sft_safety import SafeSFTTrainer, ValidationGuard
    seed_all(config.get('seed', 42))
    state = PartialState(cpu=config['device'] == 'cpu')
    device = state.device
    if config['device'] == 'cuda' and device.type != 'cuda':
        raise RuntimeError('CUDA requested but unavailable in the Unsloth container')
    settings = config['sft']
    output = Path(settings['output'])
    guard = ValidationGuard(settings, output)
    if settings.get('save_total_limit', 4) < 4:
        raise ValueError('SFT save_total_limit must be at least 4 to retain the best four checkpoints')
    alignment_path = Path(config['alignment']['output']) / 'alignment.pt'
    alignment = load_checkpoint(alignment_path)
    vq_hash = file_hash(Path(config['vq']['output']) / 'best.pt')
    if alignment['vq_sha256'] != vq_hash:
        raise ValueError('Alignment/VQ checkpoint mismatch')
    signature = {'config': config, 'alignment_sha256': file_hash(alignment_path), 'vq_sha256': vq_hash,
                 'database_sha256': file_hash(config['data']['database']),
                 'encoded_sha256': file_hash(settings['encoded_database'])}
    metadata_path = output / 'run.json'
    if metadata_path.exists():
        if not resume:
            raise FileExistsError('Existing SFT run; use --resume or choose a new output directory')
        if json.loads(metadata_path.read_text()) != signature:
            raise ValueError('SFT resume metadata mismatch')
    tokenizer = AutoTokenizer.from_pretrained(Path(config['alignment']['output']) / 'llm_tokenizer')
    if state.is_main_process:
        prepare_sft_tokens(config, tokenizer, alignment)
        write_json(metadata_path, signature)
    state.wait_for_everyone()
    tokens = output / 'tokens.sqlite'
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(alignment['model_name'], revision=alignment['model_revision'],
                                                torch_dtype=dtype, attn_implementation='sdpa')
    model.config.use_cache = False
    lora = LoraConfig(r=settings['lora_rank'], lora_alpha=settings['lora_alpha'],
                      target_modules=settings['lora_targets'], lora_dropout=settings['lora_dropout'],
                      bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, lora)
    if settings['gradient_checkpointing']:
        model.enable_input_require_grads()
    if state.is_main_process:
        model.print_trainable_parameters()
    train_dataset = InstructionDataset(tokens, 'train')
    eval_dataset = InstructionDataset(tokens, 'val')
    strategy = 'steps' if settings.get('eval_steps') else 'epoch'
    step_kwargs = {'eval_steps': settings['eval_steps'], 'save_steps': settings['eval_steps']} if settings.get('eval_steps') else {}
    training_kwargs = dict(
        output_dir=str(output / 'checkpoints'),
        num_train_epochs=settings['epochs'], max_steps=settings['max_steps'],
        learning_rate=settings['learning_rate'], weight_decay=settings['weight_decay'],
        per_device_train_batch_size=settings['batch_size'], per_device_eval_batch_size=settings['batch_size'],
        gradient_accumulation_steps=settings['gradient_accumulation_steps'],
        lr_scheduler_type='cosine', optim='adamw_torch',
        eval_strategy=strategy, save_strategy=strategy, load_best_model_at_end=True,
        **step_kwargs,
        metric_for_best_model='eval_loss', greater_is_better=False,
        save_total_limit=settings.get('save_total_limit', 4), logging_steps=settings['logging_steps'],
        logging_nan_inf_filter=False,
        logging_dir=str(output / 'tensorboard'), logging_strategy='steps', logging_first_step=True,
        bf16=dtype == torch.bfloat16, fp16=False, use_cpu=device.type == 'cpu',
        gradient_checkpointing=settings['gradient_checkpointing'],
        gradient_checkpointing_kwargs={'use_reentrant': False},
        ddp_find_unused_parameters=False, dataloader_num_workers=settings.get('dataloader_num_workers', 0),
        remove_unused_columns=False, report_to=settings.get('report_to', ['tensorboard']), seed=config.get('seed', 42),
        data_seed=config.get('seed', 42), save_safetensors=True,
    )
    if 'warmup_ratio' in inspect.signature(TrainingArguments).parameters:
        training_kwargs['warmup_ratio'] = settings['warmup_ratio']
    else:
        samples_per_rank = (len(train_dataset) + state.num_processes - 1) // state.num_processes
        steps_per_epoch = max(1, (samples_per_rank + settings['batch_size'] - 1) // settings['batch_size'])
        steps_per_epoch = max(1, (steps_per_epoch + settings['gradient_accumulation_steps'] - 1) //
                              settings['gradient_accumulation_steps'])
        total_steps = settings['max_steps'] if settings['max_steps'] > 0 else steps_per_epoch * settings['epochs']
        training_kwargs['warmup_steps'] = round(settings['warmup_ratio'] * total_steps)
    supported_arguments = inspect.signature(TrainingArguments).parameters
    if settings.get('train_sampling_strategy'):
        if 'train_sampling_strategy' not in supported_arguments:
            raise ValueError('This transformers version has no train_sampling_strategy; remove it from the SFT settings')
        training_kwargs['train_sampling_strategy'] = settings['train_sampling_strategy']
    training_kwargs = {name: value for name, value in training_kwargs.items()
                       if name in supported_arguments}
    arguments = TrainingArguments(**training_kwargs)
    callbacks = [guard]
    if settings.get('gpu_monitor'):
        from .sft_monitor import ThroughputMonitor
        callbacks.append(ThroughputMonitor(output))
    trainer = SafeSFTTrainer(model=model, args=arguments, train_dataset=train_dataset,
                             eval_dataset=eval_dataset, callbacks=callbacks,
                             data_collator=InstructionCollator(tokenizer.pad_token_id))
    # Transformers 5 moved logging_dir from TrainingArguments onto the callback.
    from transformers.integrations import TensorBoardCallback
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, TensorBoardCallback):
            callback.logging_dir = str(output / 'tensorboard')
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output / 'final_adapter'))
    if state.is_main_process:
        tokenizer.save_pretrained(output / 'final_adapter')
        write_json(output / 'validation_metrics.json', trainer.evaluate())
    else:
        # All DDP ranks must participate in the final evaluation collectives.
        trainer.evaluate()
    state.wait_for_everyone()
