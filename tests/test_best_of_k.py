import unittest

from cadtokenizer.bench_best_of_k import select_first_valid


class SelectFirstValidTest(unittest.TestCase):
    def candidate(self, sequence, status='syntax_valid'):
        return {'status': status, 'generated_sequence': sequence}

    def test_first_valid_is_selected_and_later_candidates_are_not_checked(self):
        checked = []

        def is_valid(sequence):
            checked.append(sequence)
            return sequence == 'good'

        candidates = [self.candidate('bad'), self.candidate('good'), self.candidate('good')]
        index, count = select_first_valid(candidates, is_valid)
        self.assertEqual((index, count), (1, 2))
        self.assertEqual(checked, ['bad', 'good'])

    def test_decode_failures_are_skipped_without_kernel_check(self):
        checked = []
        candidates = [self.candidate(None, 'inference_error'), self.candidate('good')]
        index, count = select_first_valid(candidates, lambda s: checked.append(s) or True)
        self.assertEqual((index, count), (1, 2))
        self.assertEqual(checked, ['good'])

    def test_no_valid_candidate_reports_all_checked(self):
        index, count = select_first_valid([self.candidate('a'), self.candidate('b')], lambda s: False)
        self.assertIsNone(index)
        self.assertEqual(count, 2)


if __name__ == '__main__':
    unittest.main()
