import json
from pathlib import Path
import tempfile
import unittest

from cadtokenizer.bench_judging import read_human_scores, read_vlm_scores
from cadtokenizer.common import file_hash


class JudgingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = {'samples': [{'id': 'a'}]}
        self.headers = {'manifest_sha256': 'manifest', 'predictions_sha256': 'predictions'}

    def tearDown(self):
        self.temp.cleanup()

    def test_vlm_missing_is_distinct_from_zero(self):
        path = self.root / 'scores.json'
        path.write_text(json.dumps({**self.headers, 'judge_model': 'local-judge-v1', 'records': [{'sample_id': 'a', 'score': None}]}))
        scores, _ = read_vlm_scores(path, 'manifest', 'predictions', self.manifest)
        self.assertEqual(scores, {})
        path.write_text(json.dumps({**self.headers, 'judge_model': 'local-judge-v1', 'records': [{'sample_id': 'a', 'score': 0}]}))
        scores, _ = read_vlm_scores(path, 'manifest', 'predictions', self.manifest)
        self.assertEqual(scores, {'a': 0.0})

    def test_scores_cannot_be_reused_for_another_prediction_file(self):
        path = self.root / 'scores.json'
        path.write_text(json.dumps({**self.headers, 'judge_model': 'judge', 'records': []}))
        with self.assertRaises(ValueError):
            read_vlm_scores(path, 'manifest', 'different-run', self.manifest)
        with self.assertRaisesRegex(ValueError, 'protocol'):
            read_vlm_scores(path, 'manifest', 'predictions', self.manifest, settings_hash='changed-camera')

    def test_blind_labels_are_decoded_and_rank_permutation_is_checked(self):
        key = self.root / 'key.json'
        key.write_text(json.dumps({**self.headers, 'labels': {'a': {'A': 'baseline1', 'B': 'ours', 'C': 'baseline2'}}}))
        path = self.root / 'human.json'
        data = {**self.headers, 'key_file': 'key.json', 'key_sha256': file_hash(key),
                'records': [{'sample_id': 'a', 'judge_id': 'judge1', 'ranking': ['A', 'B', 'C']}]}
        path.write_text(json.dumps(data))
        ranks, _ = read_human_scores(path, 'manifest', 'predictions', self.manifest)
        self.assertEqual(ranks, {'a': [2]})
        data['records'][0]['ranking'] = ['A', 'A', 'B']
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            read_human_scores(path, 'manifest', 'predictions', self.manifest)
