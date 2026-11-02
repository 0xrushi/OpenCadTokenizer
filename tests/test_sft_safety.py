import os
import unittest
from unittest.mock import patch

from cadtokenizer.common import model_runtime_available
if not model_runtime_available():
    raise unittest.SkipTest('SFT tests require the authorized Unsloth or Vast.ai runtime')

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Trainer, TrainerControl

from cadtokenizer.sft_safety import SafeSFTTrainer, ValidationGuard, require_finite


class SFTSafetyTests(unittest.TestCase):
    def test_save_disables_builtin_rotation_then_runs_ranked_rotation(self):
        trainer = object.__new__(SafeSFTTrainer)
        trainer.args = SimpleNamespace(save_total_limit=4, should_save=True)
        def save(*args, **kwargs):
            self.assertIsNone(trainer.args.save_total_limit)
        with patch.object(Trainer, '_save_checkpoint', side_effect=save), \
                patch.object(SafeSFTTrainer, '_rotate_checkpoints') as rotate:
            trainer._save_checkpoint(None, None)
        self.assertEqual(trainer.args.save_total_limit, 4)
        rotate.assert_called_once()

    def test_patience_stops_at_fifth_epoch_and_resets_on_improvement(self):
        guard = ValidationGuard({}, '/tmp/unused')
        self.assertIsNone(guard.observe(2.0))
        for _ in range(4):
            self.assertIsNone(guard.observe(2.1))
        self.assertIsNone(guard.observe(1.9))
        for _ in range(4):
            self.assertIsNone(guard.observe(1.9))
        self.assertIn('5 consecutive', guard.observe(1.9))

    def test_divergence_and_disabled_divergence(self):
        guard = ValidationGuard({}, '/tmp/unused')
        guard.observe(1.0)
        self.assertIn('2.0 times', guard.observe(2.01))
        guard = ValidationGuard({'divergence_ratio': None}, '/tmp/unused')
        guard.observe(1.0)
        self.assertIsNone(guard.observe(10.0))

    def test_resume_restores_patience(self):
        guard = ValidationGuard({}, '/tmp/unused')
        state = SimpleNamespace(log_history=[{'eval_loss': x, 'step': i}
                                            for i, x in enumerate([1., 1.1, 1.1, 1.1, 1.1], 1)])
        guard.on_train_begin(None, state, TrainerControl())
        self.assertEqual(guard.stale, 4)
        self.assertIn('5 consecutive', guard.observe(1.1))

    def test_stop_records_reason_and_nonfinite_validation_is_not_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = ValidationGuard({}, directory)
            state = SimpleNamespace(global_step=10, epoch=1, is_world_process_zero=True)
            control = guard.on_evaluate(None, state, TrainerControl(), {'eval_loss': float('nan')})
            self.assertTrue(control.should_training_stop)
            self.assertFalse(control.should_save)
            self.assertIn('Non-finite', json.loads((Path(directory) / 'stop_reason.json').read_text())['reason'])

    def test_nonfinite_gradients_abort_before_update(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 1)
            guard = ValidationGuard({}, directory)
            model(torch.ones(1, 2)).sum().backward()
            guard.on_pre_optimizer_step(None, None, None, model)
            model.weight.grad[0, 0] = float('inf')
            original = model.weight.detach().clone()
            with self.assertRaises(FloatingPointError):
                guard.on_pre_optimizer_step(None, None, None, model)
            self.assertTrue(torch.equal(original, model.weight))
            for value in (float('nan'), float('inf')):
                with self.assertRaises(FloatingPointError):
                    require_finite(torch.tensor(value), 'loss', directory)

    def test_rotation_keeps_best_four_plus_latest_and_ignores_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step, loss in enumerate([.8, .2, .7, .1, .3, .9, 1.0], 1):
                checkpoint = root / f'checkpoint-{step}'
                checkpoint.mkdir()
                (checkpoint / 'trainer_state.json').write_text(json.dumps({
                    'global_step': step, 'log_history': [{'step': step, 'eval_loss': loss}]}))
            (root / 'checkpoint-8').mkdir()
            trainer = object.__new__(SafeSFTTrainer)
            trainer.args = SimpleNamespace(output_dir=str(root), save_total_limit=4)
            trainer.state = SimpleNamespace(best_model_checkpoint=str(root / 'checkpoint-4'))
            trainer._rotate_checkpoints()
            self.assertEqual({p.name for p in root.glob('checkpoint-*')},
                             {'checkpoint-2', 'checkpoint-3', 'checkpoint-4', 'checkpoint-5',
                              'checkpoint-7', 'checkpoint-8'})
            best = json.loads((root / 'best_checkpoints.json').read_text())
            self.assertEqual([row['step'] for row in best], [4, 2, 5, 3])
