import sqlite3
import tempfile
from pathlib import Path
import unittest

from cadtokenizer.common import model_runtime_available
from cadtokenizer.tokenizer_regression import select_cases


class RegressionSelectionTests(unittest.TestCase):
    def test_duplicate_geometry_requests_are_scored_once_with_roles_preserved(self):
        from cadtokenizer.tokenizer_regression import unique_geometry_jobs
        jobs=[('a','reference','shape'),('a','prediction','shape'),
              ('a','prediction','shape'),('a','prediction','other'),('b','prediction','shape')]
        self.assertEqual(unique_geometry_jobs(jobs),[jobs[0],jobs[1],jobs[3],jobs[4]])

    def test_selection_uses_original_validity_only_and_rejects_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'data.sqlite'
            rows=[{'id':str(i),'sequence':f'shape {i}','component':str(i)} for i in range(4)]
            source={'train':[],'validation':rows}
            # Candidate reconstruction validity must not influence selection.
            baseline=[{'id':str(i),'known_valid_reference':i!=0,'paired_valid':False} for i in range(4)]
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE geometry(id,sequence,split,tokenizer_split,component)')
                db.executemany('INSERT INTO geometry VALUES(?,?,?,?,?)',[(r['id'],r['sequence'],'val','val',r['component']) for r in rows])
            chosen=select_cases(source,baseline,path,3,73)
            self.assertEqual({r['id'] for r in chosen},{'1','2','3'})
            with sqlite3.connect(path) as db:db.execute('UPDATE geometry SET split="train" WHERE id="1"')
            with self.assertRaisesRegex(ValueError,'validation geometry'):
                select_cases(source,baseline,path,3,73)


@unittest.skipUnless(model_runtime_available(),'ML tests require authorized container')
class QuantizerReferenceTests(unittest.TestCase):
    def test_reference_uses_high_precision_distance_and_keeps_paths_separate(self):
        import torch
        from cadtokenizer.vq import PrimitiveVQVAE
        from cadtokenizer.tokenizer_regression import quantizer_paths
        from tests.fixtures import config_for
        model=PrimitiveVQVAE(config_for(Path('/tmp/unused-regression-test'))).eval()
        with torch.no_grad():
            model.codebook.weight.fill_(1000.)
            model.codebook.weight[:,0]+=torch.arange(8)*.25
            pooled=torch.full((1,1,16),1000.)
            pooled[0,0,0]=1000.74
            paths,codes,stats=quantizer_paths(model,pooled,torch.ones((1,1),dtype=torch.bool))
        self.assertEqual(int(codes['quantized_reference'][0,0]),3)
        self.assertEqual(set(paths),{'continuous','quantized','quantized_amp','quantized_reference'})
        self.assertTrue(torch.equal(paths['continuous'],pooled))
        self.assertEqual(stats['vectors'],1)
        self.assertGreaterEqual(stats['nearest_margin_sum'],0.)
