import unittest
from cadtokenizer.matching import unique_assignment


class MatchingTests(unittest.TestCase):
    def test_resolves_collisions_with_distinct_native_ids(self):
        result = unique_assignment([[10, 11, 12], [10, 12, 11], [11, 12, 10]],
                                   [[5, 4, 3], [6, 4, 1], [9, 5, 1]])
        self.assertEqual(result, [12, 10, 11])
        self.assertEqual(len(result), len(set(result)))

    def test_insufficient_candidates_fail(self):
        with self.assertRaises(ValueError):
            unique_assignment([[10], [10]], [[1], [2]])
