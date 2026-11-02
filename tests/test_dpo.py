import unittest

from cadtokenizer.dpo import dpo_loss

try:
    import torch
except ImportError:  # host without torch: only the pure logic is exercised below
    torch = None


class DpoTest(unittest.TestCase):
    @unittest.skipIf(torch is None, 'torch not installed on host')
    def test_loss_is_log2_at_reference_and_falls_when_chosen_gains(self):
        zeros = torch.zeros(4)
        loss, accuracy, _ = dpo_loss(zeros, zeros, zeros, zeros, beta=0.1)
        self.assertAlmostEqual(loss.item(), 0.6931, places=3)
        higher = torch.full((4,), 5.0)
        better, _, _ = dpo_loss(higher, zeros, zeros, zeros, beta=0.1)
        self.assertLess(better.item(), loss.item())


if __name__ == '__main__':
    unittest.main()


class PrimitiveFidelityTest(unittest.TestCase):
    def test_identical_counts_score_one_and_missing_primitives_lower_the_score(self):
        from cadtokenizer.dpo import primitive_fidelity
        reference = 'line,1,1 <curve_end> line,2,2 <curve_end> arc,3,3 <curve_end> <loop_end>'
        self.assertEqual(primitive_fidelity(reference, reference), 1.0)
        self.assertLess(primitive_fidelity(reference, 'line,1,1 <curve_end> <loop_end>'), 1.0)
