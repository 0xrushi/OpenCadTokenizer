import unittest
from cadtokenizer.common import model_runtime_available
from cadtokenizer.numeric_supervision import changed_numbers,framed_changed_numbers,attach_weights
from cadtokenizer.representation import stringify,tokenize
from tests.fixtures import skexgen_item
from cadtokenizer.representation import skexgen_to_sem


class NumericMaskTests(unittest.TestCase):
    def setUp(self):self.source=stringify(tokenize(skexgen_to_sem(skexgen_item(5))))

    def test_changed_coordinate_and_unchanged_program(self):
        tokens=tokenize(self.source);tokens[1]+=1;target=stringify(tokens)
        flags=changed_numbers(self.source,target)
        self.assertEqual([i for i,x in enumerate(flags) if x],[1])
        self.assertFalse(any(changed_numbers(self.source,self.source)))
        framed=framed_changed_numbers(self.source,target)
        self.assertEqual(sum(framed),1);self.assertEqual(framed[-3:],[False]*3)
        sample={'input_ids':[1]*4+list(range(len(framed))),'labels':[-100]*4+list(range(len(framed))),'prompt_length':4}
        weighted=attach_weights(sample,framed,4)
        self.assertEqual(weighted['weights'][:4],[0.]*4)
        self.assertEqual(weighted['weights'][5],4.)
        self.assertEqual(sum(x==4 for x in weighted['weights']),1)

    def test_added_pair_marks_new_numbers_and_deletion_does_not(self):
        second=self.source.replace('add','cut')
        flags=changed_numbers(self.source,self.source+' '+second)
        self.assertEqual(sum(flags),sum(t>=12 for t in tokenize(second)))
        self.assertFalse(any(changed_numbers(self.source+' '+second,self.source)))


@unittest.skipUnless(model_runtime_available(),'Torch loss tests require the cached image')
class WeightedLossTests(unittest.TestCase):
    def test_causal_masking_normalization_and_accumulation(self):
        import torch
        import torch.nn.functional as F
        from cadtokenizer.numeric_supervision import weighted_numerator
        torch.manual_seed(4)
        labels=torch.tensor([[-100,-100,2,3,4],[-100,5,6,7,-100]])
        logits=torch.randn(2,5,11,requires_grad=True)
        active=labels[:,1:]!=-100
        weights=labels.ne(-100).float()
        expected=F.cross_entropy(logits[:,:-1].reshape(-1,11),labels[:,1:].reshape(-1),ignore_index=-100)
        actual=weighted_numerator(logits,labels,weights)/weights[:,1:].sum()
        self.assertTrue(torch.allclose(actual,expected,atol=1e-6))
        weights[0,2]=4;weights[1,3]=4;denominator=weights[:,1:][active].sum()
        full=weighted_numerator(logits,labels,weights)/denominator
        grad=torch.autograd.grad(full,logits)[0]
        other=logits.detach().clone().requires_grad_()
        for i in range(2):(weighted_numerator(other[i:i+1],labels[i:i+1],weights[i:i+1])/denominator).backward()
        self.assertTrue(torch.allclose(grad,other.grad,atol=1e-6))
        self.assertEqual(float(grad[0,0].abs().sum()),0.)
        self.assertEqual(float(grad[:,-1].abs().sum()),0.)
