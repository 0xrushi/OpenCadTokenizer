import unittest

import torch
from torch import nn

from cadtokenizer.vq import reset_dead_codes, scheduled_learning_rate


class DeadCodeResetTests(unittest.TestCase):
    def test_only_unused_codes_are_replaced(self):
        codebook = nn.Embedding(6, 4)
        before = codebook.weight.detach().clone()
        usage = torch.tensor([3, 0, 1, 0, 0, 2])
        vectors = torch.full((5, 4), 7.0)
        count = reset_dead_codes(codebook, usage, vectors, noise=0.0)
        self.assertEqual(count, 3)
        for index in (0, 2, 5):
            self.assertTrue(torch.equal(codebook.weight[index], before[index]))
        for index in (1, 3, 4):
            self.assertTrue(torch.allclose(codebook.weight[index], torch.full((4,), 7.0)))

    def test_optimizer_moments_are_cleared_for_reset_rows(self):
        codebook = nn.Embedding(4, 3)
        optimizer = torch.optim.AdamW(codebook.parameters(), lr=0.1)
        codebook(torch.tensor([0, 1, 2, 3])).sum().backward()
        optimizer.step()
        reset_dead_codes(codebook, torch.tensor([1, 0, 1, 1]), torch.ones(2, 3), optimizer)
        state = optimizer.state[codebook.weight]
        self.assertTrue(torch.count_nonzero(state['exp_avg'][1]) == 0)
        self.assertTrue(torch.count_nonzero(state['exp_avg'][0]) > 0)

    def test_nothing_to_reset(self):
        codebook = nn.Embedding(3, 2)
        self.assertEqual(reset_dead_codes(codebook, torch.ones(3, dtype=torch.long), torch.zeros(1, 2)), 0)
        self.assertEqual(reset_dead_codes(codebook, torch.zeros(3, dtype=torch.long), torch.zeros(0, 2)), 0)


class ScheduleTests(unittest.TestCase):
    def test_constant_by_default(self):
        self.assertEqual(scheduled_learning_rate({'learning_rate': 3e-4, 'epochs': 10}, 7), 3e-4)

    def test_cosine_decays_to_floor(self):
        settings = {'learning_rate': 3e-4, 'min_learning_rate': 1e-5, 'epochs': 10, 'lr_schedule': 'cosine'}
        self.assertAlmostEqual(scheduled_learning_rate(settings, 0), 3e-4)
        self.assertAlmostEqual(scheduled_learning_rate(settings, 5), 1.55e-4)
        self.assertGreater(scheduled_learning_rate(settings, 9), 1e-5)
        self.assertLess(scheduled_learning_rate(settings, 9), 2e-5)


if __name__ == '__main__':
    unittest.main()
