import unittest

from cadtokenizer.rft import accept, structure

SQUARE = 'line,4,15 <curve_end> line,58,15 <curve_end> line,58,47 <curve_end> line,4,47 <curve_end> <loop_end> <face_end> <sketch_end>'
RETRACED = 'line,4,15 <curve_end> line,58,15 <curve_end> line,58,17 <curve_end> line,58,15 <curve_end> line,57,17 <curve_end> <loop_end> <face_end> <sketch_end>'


class RftFilterTest(unittest.TestCase):
    def test_structure_counts_loops_and_retraced_edges(self):
        self.assertEqual(structure(SQUARE), (1, 0))
        self.assertEqual(structure(RETRACED)[1] > 0, True)

    def test_accept_requires_kernel_validity_end_token_no_retrace_and_loop_range(self):
        end = 99
        self.assertTrue(accept(SQUARE, [1, 2, end], end, True, 20))
        self.assertFalse(accept(SQUARE, [1, 2, end], end, False, 20))
        self.assertFalse(accept(SQUARE, [1, 2], end, True, 20))
        self.assertFalse(accept(RETRACED, [1, 2, end], end, True, 20))
        self.assertFalse(accept(SQUARE, [1, 2, end], end, True, 0))


if __name__ == '__main__':
    unittest.main()
