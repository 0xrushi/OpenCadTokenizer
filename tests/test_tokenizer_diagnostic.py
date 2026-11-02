import unittest
from pathlib import Path

from cadtokenizer.common import model_runtime_available

if not model_runtime_available():
    raise unittest.SkipTest('Tokenizer diagnostic tests require an authorized model runtime')

import torch
from cadtokenizer.vq import PrimitiveVQVAE
from cadtokenizer.representation import analyze, skexgen_to_sem
from scripts.tokenizer_diagnostic import diagnose_case
from tests.fixtures import config_for, skexgen_item


class TokenizerDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        config = config_for(Path('/tmp/unused-diagnostic-test'))
        self.model = PrimitiveVQVAE(config).eval()
        sequence, pairs = analyze(skexgen_to_sem(skexgen_item(10)))
        self.case = {'id': 'fixture', 'source_sample_id': 'fixture:test:1', 'sequence': sequence, 'pairs': pairs}
        self.alignment = {'native_ids': torch.tensor([23, 11, 52, 8, 900, 4, 31, 61]),
                          'reconstructed_code_vectors': self.model.codebook.weight.detach().clone()}

    def test_identity_alignment_preserves_decode_and_teacher_metrics(self):
        result = diagnose_case(self.model, self.alignment, self.case, torch.device('cpu'))
        self.assertEqual(result['alignment_squared_error'], 0.0)
        self.assertEqual(result['branches']['direct'], result['branches']['aligned'])
        teacher = result['branches']['direct']['teacher']
        self.assertEqual(teacher['tokens'], sum(len(pair['tokens'])+1 for pair in self.case['pairs']))
        self.assertGreater(teacher['numeric_tokens'], 0)

    def test_duplicate_native_ids_are_rejected(self):
        self.alignment['native_ids'][0] = self.alignment['native_ids'][1]
        with self.assertRaisesRegex(ValueError, 'injective'):
            diagnose_case(self.model, self.alignment, self.case, torch.device('cpu'))
