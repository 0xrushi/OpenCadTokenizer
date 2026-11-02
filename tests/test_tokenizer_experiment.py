import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from cadtokenizer.representation import analyze, primitive_ids, skexgen_to_sem, split_pairs, tokenize
from cadtokenizer.tokenizer_experiment import diverse_subset, features, make_database, select_cases
from tests.fixtures import config_for, skexgen_item


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.text = skexgen_to_sem(skexgen_item(10))

    def test_selection_is_stable_under_source_reordering(self):
        rows = [{'id': str(i)} for i in range(20)]
        self.assertEqual(select_cases(rows, 8, 42), select_cases(list(reversed(rows)), 8, 42))
        with self.assertRaises(ValueError):
            select_cases(rows, 21, 42)

    def test_database_regroups_without_changing_geometry_or_split(self):
        # Three line curves in one loop must pool to one sketch primitive.
        original = self.text.replace(self.text.split('<curve_end>')[0], 'line 0 0 <curve_end> line 10 0 <curve_end> line 10 10 ')
        self.assertEqual(features(original)['max_curve_primitives'], 6)
        self.assertEqual(features(original)['max_loop_primitives'], 4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = config_for(root)['data']
            data['sketch_partition'] = 'loop'
            rows = [{'id': 'training-id', 'sequence': original, 'component': 'train-component'}]
            val = [{'id': 'val-id', 'sequence': self.text, 'component': 'val-component'}]
            path = root / 'loop.sqlite'
            make_database(path, rows, val, data)
            with sqlite3.connect(path) as db:
                self.assertEqual(db.execute('SELECT split FROM geometry WHERE id="val-id"').fetchone()[0], 'val')
                tokens, groups = db.execute('SELECT tokens,groups FROM pairs WHERE geometry_id="training-id"').fetchone()
                self.assertEqual(max(json.loads(groups)) + 1, 4)
                self.assertEqual(json.loads(tokens), tokenize(original))
            with self.assertRaises(FileExistsError):
                make_database(path, rows, val, data)

    def test_overfit_subset_is_training_only_and_has_diverse_features(self):
        rows = []
        for index in range(8):
            f = dict(pairs=1, arcs=0, holes=0, loops=1, circles=0, max_curve_primitives=4)
            if index == 4:
                f.update(arcs=1, holes=1, loops=2)
            if index == 5:
                f.update(pairs=2, circles=1, max_curve_primitives=10)
            rows.append({'id': str(index), 'features': f})
        chosen = diverse_subset(rows, 4)
        self.assertEqual(len(chosen), 4)
        self.assertTrue({'4', '5'} <= {r['id'] for r in chosen})
