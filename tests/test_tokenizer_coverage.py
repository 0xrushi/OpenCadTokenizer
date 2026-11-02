import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from cadtokenizer.common import model_runtime_available
from cadtokenizer.representation import skexgen_to_sem
from tests.fixtures import config_for, make_sources, skexgen_item


def geometry_database(path, rows):
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE geometry(id,sequence,component,tokenizer_candidate,tokenizer_split,split)')
        for i, (split, tokenizer_split, candidate) in enumerate(rows):
            db.execute('INSERT INTO geometry VALUES(?,?,?,?,?,?)',
                       (str(i), skexgen_to_sem(skexgen_item(i)), f'c{i}', candidate, tokenizer_split, split))


def fake_evaluate(checkpoint, cases, settings, output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    good = {'status': 'valid', 'volume': 1.0}
    rows = [{'id': c['id'], 'features': c['features'], 'known_valid_reference': True, 'paired_valid': True,
             'chamfer': 0.5 if 'worse' in checkpoint else 0.1,
             'geometry': {'reference': good, 'reconstruction': good}} for c in cases]
    (root / 'per_case.json').write_text(json.dumps(rows))


class CoverageEvaluationTests(unittest.TestCase):
    def settings(self, root, split):
        return {'output': str(root / 'out'), 'database': str(root / 'db.sqlite'), 'split': split, 'case_count': 3,
                'seed': 1, 'small_slots': 12, 'point_count': 10, 'kernel_workers': 1, 'baseline': 'a',
                'checkpoints': {'a': 'a.pt', 'b': 'worse.pt'}}

    def test_test_split_panel_excludes_train_and_validation_rows(self):
        from cadtokenizer.tokenizer_coverage_eval import run_coverage_eval
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry_database(root / 'db.sqlite', [('train', 'train', 1)] * 3 + [('val', 'val', 1)] * 3 +
                              [('test', None, 1)] * 4 + [('test', None, 0)] * 2)
            with patch('cadtokenizer.tokenizer_coverage_eval.require_container'), \
                    patch('cadtokenizer.tokenizer_coverage_eval.evaluate_checkpoint', fake_evaluate):
                run_coverage_eval(self.settings(root, 'test'))
            ids = json.loads((root / 'out/manifest.json').read_text())['case_ids']
            self.assertEqual(len(ids), 3)
            self.assertTrue(set(ids) <= {'6', '7', '8', '9'})
            result = json.loads((root / 'out/results.json').read_text())
            self.assertEqual(result['models']['a']['full']['valid'], 3)
            self.assertFalse(result['paired_vs_baseline']['b']['full']['improves_validity_and_geometry'])

    def test_validation_panel_ignores_test_rows_and_changed_panel_is_rejected(self):
        from cadtokenizer.tokenizer_coverage_eval import run_coverage_eval
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry_database(root / 'db.sqlite', [('train', 'train', 1)] * 2 + [('val', 'val', 1)] * 4 + [('test', None, 1)] * 4)
            settings = self.settings(root, 'val')
            with patch('cadtokenizer.tokenizer_coverage_eval.require_container'), \
                    patch('cadtokenizer.tokenizer_coverage_eval.evaluate_checkpoint', fake_evaluate):
                run_coverage_eval(settings)
                ids = json.loads((root / 'out/manifest.json').read_text())['case_ids']
                self.assertTrue(set(ids) <= {'2', '3', '4', '5'})
                with self.assertRaisesRegex(ValueError, 'provenance changed'):
                    run_coverage_eval({**settings, 'seed': 2})

    def test_component_overlap_with_training_is_rejected(self):
        from cadtokenizer.tokenizer_coverage_eval import run_coverage_eval
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry_database(root / 'db.sqlite', [('train', 'train', 1)] + [('test', None, 1)] * 3)
            with sqlite3.connect(root / 'db.sqlite') as db:
                db.execute("UPDATE geometry SET component='c1' WHERE id='0'")
            with patch('cadtokenizer.tokenizer_coverage_eval.require_container'), \
                    patch('cadtokenizer.tokenizer_coverage_eval.evaluate_checkpoint', fake_evaluate):
                with self.assertRaisesRegex(ValueError, 'overlaps training'):
                    run_coverage_eval(self.settings(root, 'test'))


@unittest.skipUnless(model_runtime_available(), 'ML tests require the authorized Unsloth or Vast.ai runtime')
class ResumeScheduleTests(unittest.TestCase):
    def test_epoch_checkpoints_and_schedule_only_resume(self):
        import torch
        from cadtokenizer.prepare import prepare
        from cadtokenizer.vq import train_vq
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = config_for(root)
            config['data']['validation_fraction'] = 0.2
            config['vq'].update(epochs=2, patience=5, batch_size=4, num_workers=0, epoch_checkpoints=True)
            make_sources(config)
            prepare(config)
            torch.manual_seed(3)
            train_vq(config)
            output = Path(config['vq']['output'])
            for name in ('epoch-001.pt', 'epoch-002.pt', 'last.pt', 'best.pt'):
                self.assertTrue((output / name).exists(), name)
            longer = copy.deepcopy(config)
            longer['vq'].update(epochs=3, patience=50)
            train_vq(longer, str(output / 'epoch-002.pt'))
            self.assertTrue((output / 'epoch-003.pt').exists())
            self.assertEqual(torch.load(output / 'last.pt', weights_only=False)['epoch'], 2)
            changed = copy.deepcopy(longer)
            changed['vq']['learning_rate'] = config['vq']['learning_rate'] * 2
            changed['vq']['epochs'] = 4
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                train_vq(changed, str(output / 'epoch-003.pt'))

    def test_checkpoints_are_not_saved_per_epoch_by_default(self):
        import torch
        from cadtokenizer.prepare import prepare
        from cadtokenizer.vq import train_vq
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = config_for(root)
            config['data']['validation_fraction'] = 0.2
            config['vq'].update(epochs=1, batch_size=4, num_workers=0)
            make_sources(config)
            prepare(config)
            torch.manual_seed(3)
            train_vq(config)
            self.assertFalse(list(Path(config['vq']['output']).glob('epoch-*.pt')))


if __name__ == '__main__':
    unittest.main()
