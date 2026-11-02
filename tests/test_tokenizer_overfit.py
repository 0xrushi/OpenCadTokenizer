import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from cadtokenizer.common import model_runtime_available
from cadtokenizer.tokenizer_overfit import select_training_candidates
from tests.fixtures import config_for, skexgen_item
from cadtokenizer.representation import analyze, skexgen_to_sem


class OverfitSelectionTests(unittest.TestCase):
    def test_selection_rejects_nontraining_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {'id': 'x', 'sequence': 'shape', 'component': 'c'}
            (root/'source.json').write_text(json.dumps({'train': [row], 'overfit': [row]}))
            with sqlite3.connect(root/'cad.sqlite') as db:
                db.execute('CREATE TABLE geometry(id,sequence,split,tokenizer_split,component)')
                db.execute('INSERT INTO geometry VALUES(?,?,?,?,?)', ('x','shape','val','val','c'))
            with self.assertRaisesRegex(ValueError, 'training geometry'):
                select_training_candidates(root/'source.json', root/'cad.sqlite')


@unittest.skipUnless(model_runtime_available(), 'ML tests require the authorized container')
class OverfitGradientTests(unittest.TestCase):
    def test_continuous_control_bypasses_codebook_but_trains_encoder_and_decoder(self):
        import torch
        from cadtokenizer.tokenizer_overfit import training_parts
        from cadtokenizer.vq import PrimitiveVQVAE, collate_pairs
        model=PrimitiveVQVAE(config_for(Path('/tmp/unused-overfit-test'))).eval()
        _, pairs=analyze(skexgen_to_sem(skexgen_item(5)))
        result=training_parts(model,collate_pairs(pairs),'continuous')
        (result['ce_sum']/result['token_count']).backward()
        self.assertIsNone(model.codebook.weight.grad)
        for module in (model.encoder,model.decoder):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))
        self.assertEqual(float(result['vq_loss']),0)

    def test_token_and_primitive_weighted_accumulation_matches_full_objective(self):
        import torch
        from cadtokenizer.tokenizer_overfit import training_parts
        from cadtokenizer.vq import PrimitiveVQVAE, collate_pairs
        cfg=config_for(Path('/tmp/unused-overfit-test'))
        a=PrimitiveVQVAE(cfg).eval()
        b=copy.deepcopy(a)
        # Different curve counts and token lengths make example weighting wrong.
        text=skexgen_to_sem(skexgen_item(5))
        longer=text.replace('circle', 'line 0 0 <curve_end> line 10 0 <curve_end> circle',1)
        items=analyze(text)[1]+analyze(longer)[1]
        complete=training_parts(a,collate_pairs(items),'quantized')
        (complete['ce_sum']/complete['token_count']+complete['vq_loss']).backward()
        primitives=sum(max(p['groups'])+1 for p in items)
        for item in items:
            part=training_parts(b,collate_pairs([item]),'quantized')
            (part['ce_sum']/complete['token_count']+part['vq_loss']*part['primitive_count']/primitives).backward()
        for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
            if p.grad is not None:
                self.assertTrue(torch.allclose(p.grad,q.grad,atol=2e-5,rtol=2e-4),name)
