import os
import copy
import unittest

from cadtokenizer.common import model_runtime_available
if not model_runtime_available():
    raise unittest.SkipTest('ML tests require the authorized Unsloth or Vast.ai runtime')

import torch
from unittest.mock import patch
from cadtokenizer.vq import PrimitiveVQVAE, accumulated_loss, accumulation_windows, collate_pairs
from cadtokenizer.representation import analyze, skexgen_to_sem, split_pairs
from cadtokenizer.inference import NativeCadGrammar, decode_native_ids
from types import SimpleNamespace
from unittest.mock import Mock
from cadtokenizer.sft import InstructionCollator
from tests.fixtures import config_for, skexgen_item
from pathlib import Path


class TorchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = config_for(Path('/tmp/unused-cad-test'))
        self.model = PrimitiveVQVAE(self.config)
        _, pairs = analyze(skexgen_to_sem(skexgen_item(10)))
        self.batch = collate_pairs(pairs)

    def test_gradients_reach_encoder_decoder_and_codebook(self):
        result = self.model(**self.batch)
        result['loss'].backward()
        for module in (self.model.encoder, self.model.decoder, self.model.codebook):
            grads = [p.grad for p in module.parameters() if p.requires_grad]
            self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads))
            self.assertTrue(all(g is None or torch.isfinite(g).all() for g in grads))

    def test_loop_grouping_decoder_does_not_limit_curves_to_number_of_slots(self):
        self.model.config['data'].update(sketch_partition='loop', max_primitives=4)
        text = ('line 0 0 <curve_end> line 10 0 <curve_end> line 10 10 <curve_end> '
                '<loop_end> <face_end> <sketch_end> add 0 63 31 31 31 1 0 0 0 1 0 0 0 1 3 30 30 <extrude_end>')
        _, pairs = analyze(text, sketch_partition='loop', max_primitives=4)
        desired = pairs[0]['tokens'] + [78]
        def fixed_logits(ids, memory, mask):
            scores = torch.full((1, ids.shape[1], 79), -100.0)
            scores[0, -1, desired[ids.shape[1]-1]] = 100.0
            return scores
        with patch.object(self.model, 'decode', side_effect=fixed_logits):
            actual = self.model.reconstruct(torch.zeros(4, 16), force_add=True)
        self.assertEqual(actual, pairs[0]['tokens'])

    def test_curve_decoder_budgets_long_line_sketches_by_arity(self):
        # 60 lines need 263 tokens; the old 12-tokens/curve budget capped at 41 curves.
        self.model.config['data'].update(max_primitives=128, max_pair_tokens=512)
        points = [(x, 0) for x in range(30)] + [(30 - x, 20) for x in range(30)]
        curves = ' '.join(f'line {x} {y} <curve_end>' for x, y in points)
        text = curves + ' <loop_end> <face_end> <sketch_end> add 0 63 31 31 31 1 0 0 0 1 0 0 0 1 3 30 30 <extrude_end>'
        _, pairs = analyze(text, max_primitives=128)
        desired = pairs[0]['tokens'] + [78]
        def fixed_logits(ids, memory, mask):
            scores = torch.full((1, ids.shape[1], 79), -100.0)
            scores[0, -1, desired[ids.shape[1]-1]] = 100.0
            return scores
        with patch.object(self.model, 'decode', side_effect=fixed_logits):
            actual = self.model.reconstruct(torch.zeros(63, 16), force_add=True)
        self.assertEqual(actual, pairs[0]['tokens'])
        # A budget too short for another line forces the sketch to close instead.
        with patch.object(self.model, 'decode', side_effect=fixed_logits):
            short = self.model.reconstruct(torch.zeros(63, 16), max_length=60, force_add=True)
        self.assertLessEqual(len(short), 60)

    def test_accumulation_matches_full_batch_with_short_final_microbatch(self):
        full = self.model.eval()
        accumulated = copy.deepcopy(full)
        items = [analyze(skexgen_to_sem(skexgen_item(i)))[1][0] for i in range(5)]
        full(**collate_pairs(items))['loss'].backward()
        # The last accumulation window contains 2 + 2 + 1 examples.
        for start in range(0, 5, 2):
            microbatch = items[start:start+2]
            (accumulated(**collate_pairs(microbatch))['loss'] * len(microbatch) / 5).backward()
        for (name, left), (_, right) in zip(full.named_parameters(), accumulated.named_parameters()):
            if left.grad is not None:
                self.assertTrue(torch.allclose(left.grad, right.grad, atol=2e-5, rtol=2e-4), name)

    def test_production_accumulation_matches_full_batch_for_variable_lengths(self):
        full = self.model.eval()
        accumulated = copy.deepcopy(full)
        text = skexgen_to_sem(skexgen_item(5))
        longer = text.replace('circle', 'line 0 0 <curve_end> line 10 0 <curve_end> circle', 1)
        items = analyze(text)[1] + analyze(longer)[1]
        batch = collate_pairs(items)
        full(**batch)['loss'].backward()
        tokens = int(batch['targets'].ne(0).sum())
        primitives = sum(max(p['groups'])+1 for p in items)
        for item in items:
            result = accumulated(**collate_pairs([item]))
            accumulated_loss(result, tokens, primitives).backward()
        for (name, left), (_, right) in zip(full.named_parameters(), accumulated.named_parameters()):
            if left.grad is not None:
                self.assertTrue(torch.allclose(left.grad, right.grad, atol=2e-5, rtol=2e-4), name)
        self.assertEqual(list(accumulation_windows(range(5), 2)), [[0, 1], [2, 3], [4]])

    def test_pool_padding_cannot_override_negative_hidden_states(self):
        hidden = torch.full((1, 3, 16), -2.0)
        hidden[0, 2] = 100.0
        tokens = torch.tensor([[1, 13, 0]])
        groups = torch.tensor([[0, 0, -1]])
        with patch.object(self.model, 'embed', return_value=hidden), patch.object(self.model.encoder, 'forward', return_value=hidden):
            pooled, mask = self.model.pool(tokens, groups)
        self.assertTrue(torch.equal(pooled, torch.full_like(pooled, -2)))
        self.assertTrue(mask.all())

    def test_free_running_fsa_returns_complete_sem(self):
        self.model.eval()
        with torch.no_grad():
            quantized, _, mask = self.model.encode(self.batch['tokens'], self.batch['groups'])
            generated = self.model.reconstruct(quantized[0][mask[0]])
        self.assertEqual(len(split_pairs(generated)), 1)

    def test_native_framing_terminates_inside_budget(self):
        processor = NativeCadGrammar([10, 11], {'start': 20, 'pair_end': 21, 'end': 22}, 1, 4, 8)
        ids = torch.tensor([[20]])
        for _ in range(8):
            scores = processor(ids, torch.zeros(1, 32))
            choice = scores.argmax(-1, keepdim=True)
            ids = torch.cat([ids, choice], dim=1)
            if int(choice) == 22:
                break
        self.assertEqual(int(ids[0, -1]), 22)

    def test_sft_padding_does_not_mask_real_eos(self):
        result = InstructionCollator(2)([{'input_ids': [1, 8, 2], 'labels': [-100, 8, 2]},
                                          {'input_ids': [1, 2], 'labels': [-100, 2]}])
        self.assertEqual(result['labels'].tolist(), [[-100, 8, 2], [-100, 2, -100]])
        self.assertEqual(result['attention_mask'].tolist(), [[1, 1, 1], [1, 1, 0]])

    def test_exact_native_decoder_uses_original_codebook_in_native_id_order(self):
        native = torch.tensor([200, 300, 100, 500, 900, 800, 600, 700])
        alignment = {'native_ids': native, 'boundaries': {'pair_end': 1001, 'end': 1002},
                     'reconstructed_code_vectors': torch.zeros_like(self.model.codebook.weight)}
        reconstruction = self.batch['tokens'][0].tolist()
        decoder = SimpleNamespace(codebook=self.model.codebook, reconstruct=Mock(return_value=reconstruction))
        decode_native_ids(decoder, alignment, [100, 200, 1001, 1002], decoder='codebook')
        memory = decoder.reconstruct.call_args.args[0]
        self.assertTrue(torch.equal(memory, self.model.codebook.weight[[2, 0]]))
        decoder.reconstruct.reset_mock()
        decode_native_ids(decoder, alignment, [100, 200, 1001, 1002], decoder='alignment')
        self.assertTrue(torch.equal(decoder.reconstruct.call_args.args[0], torch.zeros_like(memory)))

    def test_native_decoder_rejects_malformed_frames(self):
        alignment = {'native_ids': torch.arange(8)+100, 'boundaries': {'pair_end': 20, 'end': 21}}
        for tokens in ([100, 21], [20, 21], [999, 20, 21], [100, 20]):
            with self.assertRaises(ValueError):
                decode_native_ids(self.model, alignment, tokens, decoder='codebook')
