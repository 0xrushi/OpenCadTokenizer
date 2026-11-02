import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cadtokenizer.common import model_runtime_available
if not model_runtime_available():
    raise unittest.SkipTest('ML tests require the authorized Unsloth or Vast.ai runtime')

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from cadtokenizer.sft_monitor import ThroughputMonitor, gpu_samples
from cadtokenizer.sft_profile import collate, load_lengths, make_batches, recommend, run_updates


def tokens_database(path, lengths):
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE samples (id INTEGER PRIMARY KEY, example_id TEXT, split TEXT, task TEXT, input_ids TEXT, prompt_length INTEGER)')
        for i, n in enumerate(lengths):
            db.execute('INSERT INTO samples VALUES(?,?,?,?,?,?)', (i, str(i), 'train', 'generation', json.dumps(list(range(3, 3 + n))), 2))
        db.execute('INSERT INTO samples VALUES(?,?,?,?,?,?)', (999, 'v', 'val', 'generation', json.dumps([3, 4, 5]), 1))


class ProfileBatchTests(unittest.TestCase):
    def test_lengths_exclude_validation_and_batches_have_the_intended_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'tokens.sqlite'
            lengths = [5, 9, 7, 40, 6, 8, 100, 12, 11, 10]
            tokens_database(path, lengths)
            measured = load_lengths(path)
            self.assertEqual(sorted(n for _, n in measured), sorted(lengths))
            batches = make_batches(path, measured, 3, 4, seed=1)
            by_id = dict(measured)
            self.assertEqual(sorted(by_id[i] for i in batches['longest'][0]), [12, 40, 100])
            self.assertTrue(all(len(b) == 3 for group in batches.values() for b in group))
            ordered = [i for i, _ in sorted(measured, key=lambda x: (x[1], x[0]))]
            for group in batches['grouped']:
                start = ordered.index(group[0])
                self.assertEqual(group, ordered[start:start + 3])

    def test_collate_masks_prompt_and_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'tokens.sqlite'
            tokens_database(path, [4, 6])
            batch, real = collate(path, [0, 1], 0, torch.device('cpu'))
            self.assertEqual(real, 10)
            self.assertEqual(tuple(batch['input_ids'].shape), (2, 6))
            self.assertTrue((batch['labels'][:, :2] == -100).all())
            self.assertTrue((batch['labels'][0, 4:] == -100).all())


class ProfileRunTests(unittest.TestCase):
    def test_updates_report_finite_throughput_and_padding(self):
        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                             num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'tokens.sqlite'
            tokens_database(path, [5, 9, 7, 20, 6, 8])
            batches = [collate(path, [i, i + 1], 0, torch.device('cpu')) for i in range(0, 4)]
            result = run_updates(model, optimizer, batches, torch.device('cpu'), warmup=1, measured=3)
        self.assertGreater(result['samples_per_second'], 0)
        self.assertGreater(result['real_tokens_per_second'], 0)
        self.assertTrue(0 <= result['padding_fraction'] < 1)

    def test_recommendation_ignores_ineligible_rows(self):
        def row(mb, ckpt, eligible, grouped, rand):
            stats = lambda v: {'samples_per_second': v, 'padding_fraction': 0.1, 'peak_allocated_bytes': 2**30}
            return {'microbatch': mb, 'gradient_checkpointing': ckpt, 'eligible': eligible,
                    'longest': stats(0), 'grouped': stats(grouped), 'random': stats(rand)}
        rows = [row(16, False, True, 100, 80), row(64, False, False, 900, 900), row(32, True, True, 120, 70)]
        best = recommend(rows)
        self.assertEqual(best['best_with_group_by_length']['microbatch'], 32)
        self.assertEqual(best['best_with_random_sampling']['microbatch'], 16)
        self.assertIsNone(recommend([row(8, False, False, 1, 1)])['best_with_group_by_length'])


class MonitorTests(unittest.TestCase):
    def test_missing_nvidia_smi_yields_no_samples(self):
        with patch('cadtokenizer.sft_monitor.subprocess.run', side_effect=FileNotFoundError):
            self.assertEqual(gpu_samples(), [])

    def test_monitor_writes_throughput_and_gpu_scalars_to_a_system_run(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        with tempfile.TemporaryDirectory() as temporary:
            monitor = ThroughputMonitor(temporary)
            args = SimpleNamespace(per_device_train_batch_size=4, gradient_accumulation_steps=2, world_size=2)
            state = SimpleNamespace(is_world_process_zero=True, global_step=0)
            with patch('cadtokenizer.sft_monitor.gpu_samples', return_value=[(0, 87, 40960, 81920), (1, 91, 30720, 81920)]):
                monitor.on_train_begin(args, state, None)
                monitor.mark = (monitor.mark[0] - 2.0, 0)       # pretend two seconds elapsed
                state.global_step = 10
                monitor.on_log(args, state, None, logs={'loss': 1.0})
                monitor.on_log(args, state, None, logs={'learning_rate': 1e-4})   # non-training log: ignored
                monitor.on_train_end(args, state, None)
            events = EventAccumulator(str(Path(temporary) / 'tensorboard' / 'system'))
            events.Reload()
            tags = set(events.Tags()['scalars'])
            self.assertTrue({'throughput/samples_per_second', 'gpu0/utilization_percent', 'gpu1/memory_used_gib'} <= tags)
            rate = events.Scalars('throughput/samples_per_second')[0].value
            self.assertAlmostEqual(rate, 4 * 2 * 2 * 10 / 2.0, delta=0.5)
            self.assertAlmostEqual(events.Scalars('gpu1/memory_fraction')[0].value, 0.375, places=3)

    def test_non_zero_ranks_write_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            monitor = ThroughputMonitor(temporary)
            args = SimpleNamespace(per_device_train_batch_size=1, gradient_accumulation_steps=1, world_size=2)
            state = SimpleNamespace(is_world_process_zero=False, global_step=0)
            monitor.on_train_begin(args, state, None)
            monitor.on_log(args, state, None, logs={'loss': 1.0})
            self.assertFalse((Path(temporary) / 'tensorboard').exists())


if __name__ == '__main__':
    unittest.main()
