import unittest
from cadtokenizer.common import model_runtime_available
from cadtokenizer.batch_decode import IncrementalConstraint
from cadtokenizer.lossless_sem import LosslessSemCodec,LosslessCadConstraint
from cadtokenizer.representation import skexgen_to_sem,CURVES
from tests.fixtures import skexgen_item


class IncrementalTests(unittest.TestCase):
    def test_all_prefixes_match_reference_including_pair_limit(self):
        codec=LosslessSemCodec(tuple(range(32,108)),{'start':108,'pair_end':109,'end':110})
        first=skexgen_to_sem(skexgen_item(5))
        for text,limit in [(first,1),(first+' '+first.replace('add','cut'),2)]:
            state=IncrementalConstraint(codec,limit);reference=LosslessCadConstraint(codec,max_pairs=limit)
            prefix=[]
            for token in codec.encode(text)[1:]:
                self.assertEqual(state.allowed(),reference.allowed(prefix))
                state.advance(token);prefix.append(token)
            self.assertEqual(state.allowed(),set())
            with self.assertRaises(ValueError):state.advance(110)


@unittest.skipUnless(model_runtime_available(),'GPU decoding checks require the cached image')
class BatchMaskTests(unittest.TestCase):
    def test_left_padding_finished_rows_and_per_row_masks(self):
        import torch
        from cadtokenizer.batch_decode import generate_many
        codec=LosslessSemCodec(tuple(range(32,108)),{'start':108,'pair_end':109,'end':110})
        first=skexgen_to_sem(skexgen_item(5))
        texts=[first,first+' '+first.replace('add','cut')]
        continuations=[codec.encode(text)[1:] for text in texts]
        settings={'max_pairs':16,'max_new_tokens':256,'context_length':1024}
        test=self
        class FakeModel:
            def eval(self):return self
            def generate(self,**kwargs):
                ids=kwargs['input_ids'];width=ids.shape[1]
                test.assertEqual(kwargs['attention_mask'].tolist(),[[0,0,1,1],[1,1,1,1]])
                test.assertEqual(ids[0,:2].tolist(),[128001,128001])
                for step in range(max(map(len,continuations))):
                    scores=kwargs['logits_processor'](ids,torch.zeros((2,111),device='cuda'))
                    next_ids=[]
                    for row,seq in enumerate(continuations):
                        if step>=len(seq):allowed={110};next_ids.append(128001)
                        else:
                            allowed=LosslessCadConstraint(codec,max_pairs=16).allowed(seq[:step])
                            if 256-step<36:allowed-={codec.native_ids[t-1] for t in CURVES}
                            next_ids.append(seq[step])
                        actual=set(torch.isfinite(scores[row]).nonzero().flatten().tolist())
                        test.assertEqual(actual,allowed)
                    ids=torch.cat((ids,torch.tensor(next_ids,device='cuda')[:,None]),dim=1)
                test.assertEqual(width,4)
                return ids
        rows=generate_many(FakeModel(),[[1,108],[1,2,3,108]],codec,settings)
        self.assertEqual([r['sequence'] for r in rows],[codec.decode(codec.encode(text)) for text in texts])
        self.assertEqual([r['generated_ids'] for r in rows],continuations)
