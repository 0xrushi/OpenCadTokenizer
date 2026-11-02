import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from cadtokenizer.common import model_runtime_available
from cadtokenizer.representation import skexgen_to_sem
from cadtokenizer.tokenizer_full import full_source
from tests.fixtures import config_for, skexgen_item


class FullSelectionTests(unittest.TestCase):
    def test_active_job_lock_prevents_duplicate_launch(self):
        import fcntl
        from unittest.mock import patch
        from cadtokenizer.tokenizer_full import run_full
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with (root/'run.lock').open('a+') as guard:
                fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch('cadtokenizer.tokenizer_full.require_container'), patch('cadtokenizer.tokenizer_full._run_full') as work:
                    with self.assertRaisesRegex(RuntimeError,'already holds this run lock'):
                        run_full({}, {'output':str(root)})
                    work.assert_not_called()

    def test_preserves_anchor_and_only_uses_tokenizer_training_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cfg=config_for(root)
            cases=[{'id':str(i),'sequence':skexgen_to_sem(skexgen_item(i)),'component':str(i)} for i in range(8)]
            with sqlite3.connect(cfg['data']['database']) as db:
                db.execute('CREATE TABLE geometry(id,sequence,component,tokenizer_candidate,tokenizer_split,split)')
                for i,row in enumerate(cases):
                    split='train' if i<3 else 'val'
                    # One training geometry is not a tokenizer candidate.
                    candidate=0 if i==2 else 1
                    db.execute('INSERT INTO geometry VALUES(?,?,?,?,?,?)',(row['id'],row['sequence'],row['component'],candidate,split,split))
            anchor=root/'anchor.json';anchor.write_text(json.dumps({'validation':[cases[5]]}))
            settings={'output':str(root/'out'),'anchor_manifest':str(anchor),'seed':42,
                      'validation_geometries':3,'train_geometries':2}
            selected=full_source(cfg,settings)
            self.assertEqual([r['id'] for r in selected['train']],['0','1'])
            self.assertEqual(selected['validation'][0]['id'],'5')
            self.assertEqual(len(selected['validation']),3)
            self.assertEqual(full_source(cfg,settings),selected)


@unittest.skipUnless(model_runtime_available(),'ML tests require authorized container')
class FixedBootstrapTests(unittest.TestCase):
    def test_bootstrap_is_independent_of_training_microbatch(self):
        import torch
        from cadtokenizer.tokenizer_experiment import make_database
        from cadtokenizer.tokenizer_learning_curve import initialization_pairs,initialize_codebook
        from cadtokenizer.vq import PairDataset,PrimitiveVQVAE
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cfg=config_for(root)
            cfg['data']['sketch_partition']='curve';cfg['vq']['dropout']=.1
            rows=[{'id':str(i),'sequence':skexgen_to_sem(skexgen_item(i)),'component':str(i)} for i in range(4)]
            path=root/'bootstrap.sqlite';make_database(path,rows,[],cfg['data'])
            dataset=PairDataset(path,'train');pairs=[dataset[i] for i in range(len(dataset))]
            settings={'bootstrap_database':str(path),'bootstrap_batch_size':2,'seed':42}
            # Profile settings need not contain a selected microbatch yet.
            first=initialization_pairs(settings,pairs,cfg)
            other=initialization_pairs({**settings,'microbatch_size':4},pairs,cfg)
            self.assertEqual(first,other)
            torch.manual_seed(7);a=PrimitiveVQVAE(cfg)
            torch.manual_seed(7);b=PrimitiveVQVAE(cfg)
            torch.manual_seed(99);initialize_codebook(a,first,torch.device('cpu'))
            torch.manual_seed(99);initialize_codebook(b,other,torch.device('cpu'))
            self.assertTrue(torch.equal(a.codebook.weight,b.codebook.weight))
