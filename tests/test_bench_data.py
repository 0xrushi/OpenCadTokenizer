import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from cadtokenizer.bench_data import freeze_manifest, load_manifest, read_predictions
from cadtokenizer.bench_predict import predict_samples, partition_samples
from cadtokenizer.bench_shards import merge_shards
from cadtokenizer.bench_data import content_hash
from cadtokenizer.common import file_hash
from cadtokenizer.prepare import prepare
from cadtokenizer.representation import skexgen_to_sem
from tests.fixtures import config_for, make_sources, skexgen_item


class BenchmarkDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = config_for(self.root)
        make_sources(self.config)
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(self.config)
        self.settings = {'seed': 42, 'output': str(self.root / 'bench'), 'test_counts': {'generation': 1, 'editing': 1}}

    def tearDown(self):
        self.temp.cleanup()

    def frozen(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return freeze_manifest(self.config, self.settings)

    def test_partitions_cover_every_case_once(self):
        samples = list(range(17))
        shards = [partition_samples(samples, i, 4) for i in range(4)]
        self.assertEqual(sorted(item for shard in shards for item in shard), samples)
        with self.assertRaises(ValueError):
            partition_samples(samples, 4, 4)

    def make_shards(self):
        manifest = self.frozen()
        root = Path(self.settings['output'])
        (root / 'shards').mkdir()
        signature = {'manifest_sha256': file_hash(root / 'manifest.json'), 'model': 'fixture'}
        metadata = {'schema': 'cad-predictions-v1', 'signature': signature,
                    'signature_sha256': content_hash(signature)}
        for index in range(2):
            path = root / 'shards' / f'predictions-{index}.jsonl'
            rows = [{'sample_id': sample['id'], 'status': 'inference_error',
                     'generated_sequence': None, 'manifest_sha256': signature['manifest_sha256'],
                     'run_signature_sha256': metadata['signature_sha256']}
                    for sample in partition_samples(manifest['samples'], index, 2)]
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            path.with_suffix('.meta.json').write_text(json.dumps(metadata))
        return root, manifest

    def test_shard_merge_is_complete_and_idempotent(self):
        root, manifest = self.make_shards()
        with contextlib.redirect_stdout(io.StringIO()):
            merge_shards(self.settings, 2)
            merge_shards(self.settings, 2)
        self.assertEqual(len(read_predictions(root / 'predictions.jsonl', manifest)), 2)

    def test_shard_merge_rejects_missing_cases(self):
        root, _ = self.make_shards()
        (root / 'shards/predictions-0.jsonl').write_text('')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            merge_shards(self.settings, 2)

    def test_shard_merge_rejects_mismatched_model_signature(self):
        root, _ = self.make_shards()
        path = root / 'shards/predictions-0.meta.json'
        metadata = json.loads(path.read_text())
        metadata['signature']['model'] = 'another-model'
        metadata['signature_sha256'] = content_hash(metadata['signature'])
        path.write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            merge_shards(self.settings, 2)

    def test_manifest_is_fixed_and_not_claimed_as_the_paper_test_set(self):
        manifest = self.frozen()
        self.assertEqual(manifest['counts'], {'generation': 1, 'editing': 1})
        self.assertFalse(manifest['paper_test_set_verified'])
        with self.assertRaises(FileExistsError):
            freeze_manifest(self.config, self.settings)

    def test_train_overlap_blocks_benchmark_creation(self):
        sequence = skexgen_to_sem(skexgen_item(2))
        from cadtokenizer.bench_data import canonical_sequence
        with contextlib.closing(sqlite3.connect(self.config['data']['database'])) as db, db:
            db.execute('UPDATE geometry SET split="train" WHERE sequence=?', (canonical_sequence(sequence),))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.frozen()

    def test_explicit_overlap_exclusion_refills_and_records_audit(self):
        self.settings['exclude_training_overlaps'] = True
        path = Path(self.config['data']['root']) / 'cad_editor/dataset/test.json'
        rows = json.loads(path.read_text())
        rows.append({**rows[0], 'original_sequence': skexgen_to_sem(skexgen_item(45)),
                     'edited_sequence': skexgen_to_sem(skexgen_item(46))})
        path.write_text(json.dumps(rows))
        from cadtokenizer.bench_data import canonical_sequence
        with contextlib.closing(sqlite3.connect(self.config['data']['database'])) as db, db:
            target = canonical_sequence(rows[0]['edited_sequence'])
            db.execute('UPDATE geometry SET split="train" WHERE sequence=?', (target,))
        manifest = self.frozen()
        self.assertEqual(manifest['counts'], {'generation': 1, 'editing': 1})
        self.assertIn('cad_editor:test:0', manifest['overlap_audit']['excluded_training_overlap_samples'])
        self.assertNotIn('cad_editor:test:0', {s['id'] for s in manifest['samples']})
        self.assertIn('cad_editor:test:1', {s['id'] for s in manifest['samples']})

    def test_sources_over_tokenizer_limit_remain_in_manifest(self):
        path = Path(self.config['data']['root']) / 'cad_editor/dataset/test.json'
        rows = json.loads(path.read_text())
        full = rows[0]['original_sequence']
        curve = full.split('<loop_end>')[0]
        suffix = full.split('<face_end>')[1]
        large = (curve + ' <loop_end> ') * 20 + '<face_end>' + suffix
        rows[0]['original_sequence'] = large
        path.write_text(json.dumps(rows))
        manifest = self.frozen()
        sample = next(s for s in manifest['samples'] if s['task'] == 'editing')
        self.assertEqual(sample['source_sequence'].count('<curve_end>'), 20)

    def test_empty_upstream_instruction_remains_an_explicit_failure(self):
        path = Path(self.config['data']['root']) / 'cad_editor/dataset/test.json'
        rows = json.loads(path.read_text())
        rows[0]['instruction'] = ''
        path.write_text(json.dumps(rows))
        manifest = self.frozen()
        sample = next(s for s in manifest['samples'] if s['task'] == 'editing')
        self.assertIn('instruction_error', sample)
        class NoCalls:
            def predict(self, *args):
                raise AssertionError('Malformed instruction must not be sent to inference')
        output = self.root / 'bad-instruction.jsonl'
        context = {'manifest_sha256': 'manifest', 'run_signature_sha256': 'run', 'max_new_tokens': 50}
        with contextlib.redirect_stdout(io.StringIO()):
            predict_samples([sample], NoCalls(), output, context)
        result = json.loads(output.read_text())
        self.assertEqual(result['status'], 'inference_error')
        self.assertIn('instruction', result['error'])

    def test_inference_never_receives_target_geometry_and_preserves_failures(self):
        manifest = self.frozen()
        samples = manifest['samples']
        calls = []
        class FakeGenerator:
            def predict(self, instruction, source, budget):
                calls.append((instruction, source, budget))
                if source is not None:
                    raise ValueError('source exceeds context')
                return {'generated_sequence': 'fixed prediction'}
        output = self.root / 'predictions.jsonl'
        context = {'manifest_sha256': 'manifest', 'run_signature_sha256': 'run', 'max_new_tokens': 50}
        with contextlib.redirect_stdout(io.StringIO()):
            predict_samples(samples, FakeGenerator(), output, context)
        rows = read_predictions(output, manifest)
        self.assertEqual(len(rows), len(samples))
        self.assertEqual(sum(row['status'] == 'inference_error' for row in rows.values()), 1)
        expected = [(s['instruction'], s['source_sequence'], 50) for s in samples]
        self.assertEqual(calls, expected)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(predict_samples(samples, FakeGenerator(), output, context, existing=rows), 0)

    def test_resume_recovers_only_incomplete_last_json(self):
        manifest = self.frozen()
        output = self.root / 'predictions.jsonl'
        good = {'sample_id': manifest['samples'][0]['id'], 'status': 'inference_error'}
        output.write_text(json.dumps(good) + '\n{"sample_id":')
        with self.assertRaises(ValueError):
            read_predictions(output, manifest)
        rows = read_predictions(output, manifest, recover_tail=True)
        self.assertEqual(len(rows), 1)
        self.assertTrue(output.read_text().endswith('\n'))

    def test_duplicate_or_unknown_prediction_ids_fail(self):
        manifest = self.frozen()
        output = self.root / 'predictions.jsonl'
        good = {'sample_id': manifest['samples'][0]['id'], 'status': 'inference_error'}
        output.write_text((json.dumps(good) + '\n') * 2)
        with self.assertRaises(ValueError):
            read_predictions(output, manifest)
        output.write_text(json.dumps({**good, 'sample_id': 'unknown'}) + '\n')
        with self.assertRaises(ValueError):
            read_predictions(output, manifest)
