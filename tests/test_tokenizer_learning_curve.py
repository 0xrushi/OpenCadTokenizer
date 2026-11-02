import unittest
import json
import sqlite3
import tempfile
from pathlib import Path
from cadtokenizer.common import model_runtime_available
from cadtokenizer.tokenizer_learning_curve import paired_comparison, verify_cases


def row(key, valid, cd=None):
    return {'id': key, 'paired_valid': valid, 'known_valid_reference': True, 'chamfer': cd}


class LearningCurveComparisonTests(unittest.TestCase):
    def test_requested_emd_is_saved_without_changing_regularization(self):
        from cadtokenizer.tokenizer_learning_curve import freeze
        from cadtokenizer.representation import skexgen_to_sem
        from tests.fixtures import config_for, skexgen_item
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            cfg=config_for(root)
            cfg['vq'].update(output=str(root/'baseline'),dropout=.1,weight_decay=.01)
            (root/'baseline').mkdir()
            (root/'baseline/best.pt').write_bytes(b'fingerprint fixture, no model load')
            cases=[{'id':str(i),'sequence':skexgen_to_sem(skexgen_item(5+i)), 'component':str(i)} for i in range(2)]
            with sqlite3.connect(cfg['data']['database']) as db:
                db.execute('CREATE TABLE geometry(id,sequence,split,tokenizer_split,component)')
                for split,case in zip(('train','val'),cases):
                    db.execute('INSERT INTO geometry VALUES(?,?,?,?,?)',(case['id'],case['sequence'],split,split,case['component']))
            source=root/'source.json'
            source.write_text(json.dumps({'train':cases[:1],'validation':cases[1:],'selection':'test'}))
            settings={'output':str(root/'run'),'source_manifest':str(source),'seed':42,
                      'train_geometries':1,'validation_geometries':1,'microbatch_size':2,
                      'effective_batch_size':4,'reconstruction_loss':'emd'}
            manifest=freeze(cfg,settings,{'runtime':'fixture'})
            for variant in manifest['variants'].values():
                saved=json.loads(Path(variant['config_path']).read_text())
                self.assertEqual(saved['vq']['reconstruction_loss'],'emd')
                self.assertEqual(saved['vq']['dropout'],.1)
                self.assertEqual(saved['vq']['weight_decay'],.01)

    def test_requires_improved_validity_and_geometry_on_same_ids(self):
        baseline=[row('a',True,.1),row('b',True,.2),row('c',False)]
        candidate=[row('a',True,.05),row('b',True,.1),row('c',True,5.)]
        result=paired_comparison(baseline,candidate)
        self.assertTrue(result['improves_validity_and_geometry'])
        self.assertEqual(result['common_valid_cases'],2)
        self.assertEqual(result['baseline_invalid_to_candidate_valid'],1)
        self.assertAlmostEqual(result['candidate_mean_chamfer_common'],.075)
        candidate[0]['chamfer']=1.
        self.assertFalse(paired_comparison(baseline,candidate)['improves_validity_and_geometry'])

    def test_different_cases_or_reference_validity_are_rejected(self):
        with self.assertRaises(ValueError):
            paired_comparison([row('a',True,.1)],[row('b',True,.1)])
        a=row('a',True,.1);b=row('a',True,.1);b['known_valid_reference']=False
        with self.assertRaises(ValueError):
            paired_comparison([a],[b])

    def test_component_overlap_is_rejected_before_reading_database(self):
        manifest={'train':[{'id':'a','component':'shared'}], 'validation':[{'id':'b','component':'shared'}]}
        with self.assertRaisesRegex(ValueError,'component overlap'):
            verify_cases(manifest,'/unused',1,1)


@unittest.skipUnless(model_runtime_available(), 'ML tests require authorized container')
class LearningCurveRngTests(unittest.TestCase):
    def test_evaluation_does_not_change_training_rng(self):
        import random
        import numpy as np
        import torch
        from cadtokenizer.tokenizer_learning_curve import preserve_rng
        random.seed(7);np.random.seed(7);torch.manual_seed(7)
        states=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        with preserve_rng():
            random.seed(42);np.random.seed(42);torch.manual_seed(42)
            random.random();np.random.rand(3);torch.rand(3)
        self.assertEqual(random.getstate(),states[0])
        self.assertTrue(np.array_equal(np.random.get_state()[1],states[1][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(),states[2]))

    def test_epoch_order_is_reproducible_without_consuming_global_rng(self):
        import torch
        from cadtokenizer.tokenizer_learning_curve import epoch_order
        torch.manual_seed(7);state=torch.get_rng_state()
        self.assertEqual(epoch_order(31,42,0),epoch_order(31,42,0))
        self.assertNotEqual(epoch_order(31,42,0),epoch_order(31,42,1))
        self.assertTrue(torch.equal(torch.get_rng_state(),state))
        self.assertEqual(sorted(epoch_order(31,42,0)),list(range(31)))
