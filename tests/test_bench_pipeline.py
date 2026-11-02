import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cadtokenizer.bench_metrics import distance_matrix_statistics
from cadtokenizer.bench_runtime import KernelRunner
from cadtokenizer.bench_score import comparison_markdown, score_cases
from cadtokenizer.bench_smoke import rectangle_sem


class PipelineTests(unittest.TestCase):
    def test_scoring_keeps_failed_cases_and_reference_pools(self):
        original, edited = rectangle_sem(), rectangle_sem(upper=45)
        manifest = {'samples': [
            {'id': 'g1', 'task': 'generation', 'instruction': 'one', 'source_sequence': None, 'target_sequence': original},
            {'id': 'g2', 'task': 'generation', 'instruction': 'two', 'source_sequence': None, 'target_sequence': original},
            {'id': 'e1', 'task': 'editing', 'instruction': 'three', 'source_sequence': original, 'target_sequence': edited},
            {'id': 'e2', 'task': 'editing', 'instruction': 'four', 'source_sequence': original, 'target_sequence': edited},
        ]}
        predictions = {
            'g1': {'status': 'syntax_valid', 'generated_sequence': original},
            'g2': {'status': 'inference_error', 'generated_sequence': None},
            'e1': {'status': 'syntax_valid', 'generated_sequence': original},
            'e2': {'status': 'syntax_valid', 'generated_sequence': 'broken'},
        }
        class FakeKernel:
            def reconstruct(self, sequence, sample_id, role):
                return {'status': 'syntax_invalid' if sequence == 'broken' else 'valid'}
        pools = []
        def fake_geometry(predicted, reference, settings):
            pools.append((set(predicted), set(reference)))
            common = sorted(set(predicted) & set(reference))
            matched = {'CD': 0.0, 'COV': 1.0, 'MMD': 0.0, 'JSD': 0.0, 'paired_count': len(common)}
            return {'matched_valid': matched, 'all_valid_references': {'COV': len(common)/len(reference)},
                    'paired_cd_by_id': {key: 0 for key in common}}
        with contextlib.redirect_stdout(io.StringIO()):
            tasks, cases = score_cases(manifest, predictions, {'render': False, 'normalization': 'max_abs_per_shape'},
                                       FakeKernel(), lambda _: [[1, 0, 0]], geometry_metric=fake_geometry)
        self.assertEqual(pools, [({'g1'}, {'g1', 'g2'}), ({'e1'}, {'e1', 'e2'})])
        self.assertEqual(tasks['generation']['table_metrics']['IR'], 50)
        self.assertEqual(tasks['editing']['table_metrics']['IR'], 50)
        self.assertEqual(tasks['editing']['copied_source_rate'], 0.5)
        self.assertEqual(len(cases), 4)
        with self.assertRaises(ValueError):
            score_cases(manifest, {}, {}, None, None)

    def test_missing_reference_is_reported_without_dropping_prediction(self):
        sequence = rectangle_sem()
        manifest = {'samples': [{'id': 'g', 'task': 'generation', 'instruction': 'create', 'target_sequence': None, 'source_sequence': None}]}
        class FakeKernel:
            def reconstruct(self, sequence, sample_id, role):
                return {'status': 'missing_sequence' if sequence is None else 'valid'}
        def fake_metrics(predicted, reference, settings):
            empty = {'CD': None, 'COV': None, 'JSD': None, 'MMD': None, 'paired_count': 0}
            return {'matched_valid': empty, 'all_valid_references': empty}
        with contextlib.redirect_stdout(io.StringIO()):
            tasks, _ = score_cases(manifest, {'g': {'status': 'syntax_valid', 'generated_sequence': sequence}},
                                   {'render': False, 'normalization': 'max_abs_per_shape'}, FakeKernel(),
                                   lambda _: [[1, 0, 0]], geometry_metric=fake_metrics)
        self.assertEqual(tasks['generation']['invalid_references'], 1)
        self.assertEqual(tasks['generation']['table_metrics']['IR'], 0)
        self.assertIsNone(tasks['generation']['table_metrics']['CD'])


class KernelControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        settings = json.loads(Path('configs/benchmark.json').read_text())
        settings['output'] = self.temp.name
        with patch('cadtokenizer.bench_runtime.require_container'), patch('cadtokenizer.bench_runtime.importlib.metadata.version', return_value='test'):
            self.runner = KernelRunner(settings)

    def tearDown(self):
        self.temp.cleanup()

    def test_timeout_is_cached_as_failure(self):
        with patch('cadtokenizer.bench_runtime.subprocess.run', side_effect=subprocess.TimeoutExpired('worker', 60)) as mocked:
            result = self.runner.reconstruct(rectangle_sem(), 'sample', 'prediction')
            self.assertEqual(result['status'], 'timeout')
            self.assertEqual(self.runner.reconstruct(rectangle_sem(), 'sample', 'prediction')['status'], 'timeout')
            self.assertEqual(mocked.call_count, 1)

    def test_infrastructure_errors_never_turn_into_invalidity(self):
        def fail(command, **kwargs):
            request = json.loads(Path(command[command.index('--request')+1]).read_text())
            output = Path(command[command.index('--result')+1])
            output.write_text(json.dumps({'status': 'infrastructure_error', 'error': 'missing API', 'request_sha256': request['request_sha256']}))
            return SimpleNamespace(stdout='', stderr='', returncode=2)
        with patch('cadtokenizer.bench_runtime.subprocess.run', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'infrastructure'):
                self.runner.reconstruct(rectangle_sem(), 'sample', 'prediction')
            with self.assertRaisesRegex(RuntimeError, 'infrastructure'):
                self.runner.reconstruct(rectangle_sem(), 'sample', 'prediction')
