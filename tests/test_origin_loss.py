import copy
from pathlib import Path
import unittest

from cadtokenizer.common import model_runtime_available
if not model_runtime_available():
    raise unittest.SkipTest('Torch checks require the authorized image')

import torch
from cadtokenizer.vq import origin_ordinal_loss, collate_pairs, accumulated_loss, PrimitiveVQVAE
from cadtokenizer.representation import analyze, skexgen_to_sem
from tests.fixtures import config_for, skexgen_item


class OriginLossTests(unittest.TestCase):
    def setUp(self):
        self.text = skexgen_to_sem(skexgen_item(5))
        self.items = analyze(self.text)[1] + analyze(self.text.replace(
            'circle', 'line 0 0 <curve_end> circle', 1))[1]
        self.batch = collate_pairs(self.items)

    def test_only_origin_positions_receive_auxiliary_gradient(self):
        targets = self.batch['targets']
        logits = torch.randn(*targets.shape, 79, requires_grad=True)
        origin_ordinal_loss(logits, targets).backward()
        expected = torch.zeros_like(targets, dtype=torch.bool)
        for row in range(len(targets)):
            boundary = next(i for i,t in enumerate(targets[row]) if int(t) in (8,9,10))
            expected[row,boundary+16:boundary+18] = True
        self.assertTrue(torch.equal(logits.grad.abs().sum(-1)>0, expected))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_replication_resets_both_arms_rng_without_changing_parent_state(self):
        from cadtokenizer.tokenizer_origin_ablation import warm_start
        torch.manual_seed(7)
        parent = {'model':{'weights':torch.ones(2)},'optimizer':{'moments':[1,2]},
                  'rng':torch.get_rng_state(),'cuda_rng':[], 'epoch':61,'offset':34112,'step':40000}
        original_rng = parent['rng'].clone()
        left = warm_start(parent,{},'control',43)
        torch.rand(50)
        right = warm_start(parent,{},'origin',43)
        self.assertTrue(torch.equal(left['rng'],right['rng']))
        self.assertTrue(torch.equal(parent['rng'],original_rng))
        self.assertFalse(torch.equal(left['rng'],original_rng))
        for key in ('model','optimizer','epoch','offset','step'):
            self.assertIs(left[key],parent[key])

    def test_geometry_seed_does_not_change_training_seed(self):
        from cadtokenizer.tokenizer_learning_curve import evaluation_settings
        settings = {'seed':43,'evaluation_seed':42,'microbatch_size':64}
        result = evaluation_settings(settings)
        self.assertEqual(result['seed'],42)
        self.assertEqual(settings['seed'],43)
        self.assertEqual(result['microbatch_size'],64)

    def test_nearby_bins_cost_less_than_distant_bins(self):
        targets = self.batch['targets']
        logits = torch.full((*targets.shape,79), -40.)
        nearby, distant = logits.clone(), logits.clone()
        for row in range(len(targets)):
            boundary = next(i for i,t in enumerate(targets[row]) if int(t) in (8,9,10))
            for position in (boundary+16,boundary+17):
                truth = int(targets[row,position])-13
                near = truth+1 if truth<63 else truth-1
                far = 63 if truth<32 else 0
                nearby[row,position,near+13] = 40
                distant[row,position,far+13] = 40
        self.assertLess(float(origin_ordinal_loss(nearby,targets)),float(origin_ordinal_loss(distant,targets)))

    def test_variable_length_accumulation_preserves_auxiliary_normalization(self):
        torch.manual_seed(7)
        cfg = config_for(Path('/tmp/origin-loss-test'))
        cfg['vq'].update(reconstruction_loss='cross_entropy',origin_loss_weight=32.)
        full = PrimitiveVQVAE(cfg).eval()
        micro = copy.deepcopy(full)
        full(**self.batch)['loss'].backward()
        tokens = int(self.batch['targets'].ne(0).sum())
        primitives = sum(max(item['groups'])+1 for item in self.items)
        for item in self.items:
            accumulated_loss(micro(**collate_pairs([item])),tokens,primitives).backward()
        for (name,a),(_,b) in zip(full.named_parameters(),micro.named_parameters()):
            if a.grad is not None:
                self.assertTrue(torch.allclose(a.grad,b.grad,atol=2e-5,rtol=2e-4),name)
