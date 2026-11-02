import copy
import unittest

from cadtokenizer.tokenizer_continuation import lower_optimizer_rate


class ContinuationTests(unittest.TestCase):
    def test_preserves_moments_cursor_rng_and_all_parameter_groups(self):
        checkpoint = {'optimizer': {'state': {'moment': [1, 2]}, 'param_groups': [
            {'lr': 3e-4, 'initial_lr': 3e-4}, {'lr': 3e-4}]},
            'step': 35000, 'offset': 17, 'rng': [42]}
        expected = copy.deepcopy(checkpoint)
        self.assertEqual(lower_optimizer_rate(checkpoint, 3e-5), [3e-4, 3e-4])
        expected['optimizer']['param_groups'] = [{'lr': 3e-5, 'initial_lr': 3e-5}, {'lr': 3e-5}]
        self.assertEqual(checkpoint, expected)

    def test_rejects_unexpected_source_rate_without_mutation(self):
        checkpoint = {'optimizer': {'param_groups': [{'lr': 3e-4}, {'lr': 1e-4}]}}
        expected = copy.deepcopy(checkpoint)
        with self.assertRaises(ValueError):
            lower_optimizer_rate(checkpoint, 3e-5)
        self.assertEqual(checkpoint, expected)
