"""SFT validation guards and metric-ranked, resumable checkpoint retention."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback

from .common import write_json


def require_finite(value, description, output):
    """Agree across DDP ranks before raising, so peers do not keep training."""
    invalid = (~torch.isfinite(value.detach())).any().to(dtype=torch.int32)
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if distributed:
        torch.distributed.all_reduce(invalid, op=torch.distributed.ReduceOp.MAX)
    if invalid.item():
        if not distributed or torch.distributed.get_rank() == 0:
            write_json(Path(output) / 'stop_reason.json', {'reason': f'Non-finite {description}'})
        raise FloatingPointError(f'Non-finite {description}; stopped before optimizer update')


class ValidationGuard(TrainerCallback):
    def __init__(self, settings, output):
        self.output = Path(output)
        self.patience = settings.get('early_stopping_patience', 5)
        self.threshold = settings.get('early_stopping_threshold', 0.0)
        self.divergence_ratio = settings.get('divergence_ratio', 2.0)
        if self.patience < 1 or self.threshold < 0:
            raise ValueError('Early stopping requires positive patience and nonnegative threshold')
        if self.divergence_ratio is not None and self.divergence_ratio <= 1:
            raise ValueError('divergence_ratio must exceed 1, or be null to disable')
        self.best = math.inf
        self.stale = 0
        self.last_step = None

    def observe(self, loss):
        reason = None
        if not math.isfinite(loss):
            reason = 'Non-finite validation loss'
        elif (self.divergence_ratio is not None and math.isfinite(self.best)
              and loss > self.best * self.divergence_ratio):
            reason = f'Validation loss exceeds {self.divergence_ratio} times the best loss'
        elif loss < self.best - self.threshold:
            self.best, self.stale = loss, 0
        else:
            self.stale += 1
            if self.stale >= self.patience:
                reason = f'No validation improvement for {self.patience} consecutive epochs'
        return reason

    def on_train_begin(self, args, state, control, **kwargs):
        # Reconstruct patience from checkpoint history rather than resetting on resume.
        self.best, self.stale, self.last_step = math.inf, 0, None
        for entry in state.log_history:
            if 'eval_loss' in entry and entry.get('step') != self.last_step:
                self.observe(float(entry['eval_loss']))
                self.last_step = entry.get('step')
        if self.stale >= self.patience:
            control.should_training_stop = True

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if self.last_step == state.global_step:
            return control
        self.last_step = state.global_step
        loss = float(metrics['eval_loss'])
        reason = self.observe(loss)
        if reason:
            control.should_training_stop = True
            # Never save non-finite validation results as resumable checkpoints.
            control.should_save = math.isfinite(loss)
            if state.is_world_process_zero:
                write_json(self.output / 'stop_reason.json', {
                    'reason': reason, 'step': state.global_step, 'epoch': state.epoch,
                    'validation_loss': loss if math.isfinite(loss) else None,
                    'best_validation_loss': self.best if math.isfinite(self.best) else None,
                })
        return control

    def on_pre_optimizer_step(self, args, state, control, model, **kwargs):
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if gradients:
            finite = torch.stack([torch.isfinite(g).all() for g in gradients]).all()
            require_finite(torch.where(finite, 0.0, float('nan')), 'gradients', self.output)


class SafeSFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        require_finite(result[0] if return_outputs else result, 'loss', Path(self.args.output_dir).parent)
        return result

    def _save_checkpoint(self, *args, **kwargs):
        # Newer Transformers calls a module-level rotation function; disable
        # that recent-only policy until saving has completed on every rank.
        limit = self.args.save_total_limit
        self.args.save_total_limit = None
        try:
            super()._save_checkpoint(*args, **kwargs)
        finally:
            self.args.save_total_limit = limit
        if self.args.should_save:
            self._rotate_checkpoints()

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None, **kwargs):
        if self.args.save_total_limit is None:
            return
        # HF's default rotation retains recent checkpoints, not the best K.
        root = Path(output_dir or self.args.output_dir)
        ranked = []
        for path in root.glob('checkpoint-*'):
            if path.is_symlink() or not path.is_dir() or not path.name[11:].isdigit():
                continue
            metadata = path / 'trainer_state.json'
            if not metadata.exists():
                continue
            checkpoint_state = json.loads(metadata.read_text())
            step = checkpoint_state['global_step']
            losses = [float(row['eval_loss']) for row in checkpoint_state['log_history']
                      if row.get('step') == step and 'eval_loss' in row]
            if losses and math.isfinite(losses[-1]):
                ranked.append((losses[-1], step, path))
        ranked.sort(key=lambda row: (row[0], -row[1]))
        count = self.args.save_total_limit or 4
        best = ranked[:count]
        keep = {path for _, _, path in best}
        # Keep the newest checkpoint as well so resume never rewinds training.
        if ranked:
            keep.add(max(ranked, key=lambda row: row[1])[2])
        if self.state.best_model_checkpoint:
            keep.add(Path(self.state.best_model_checkpoint))
        write_json(root / 'best_checkpoints.json', [
            {'path': str(path), 'step': step, 'validation_loss': loss}
            for loss, step, path in best
        ])
        for _, _, path in ranked:
            if path not in keep:
                shutil.rmtree(path)
