import unittest

from cadtokenizer.lossless_sem import LosslessSemCodec, LosslessCadConstraint
from cadtokenizer.lossless_prepare import lossless_sample
from cadtokenizer.representation import stringify, tokenize, skexgen_to_sem
from tests.fixtures import skexgen_item


class TextTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def encode(self,text,add_special_tokens=False):
        return [7]*len(text.split())


class LosslessSemTests(unittest.TestCase):
    def setUp(self):
        self.codec = LosslessSemCodec(tuple(range(1000,1076)),{'start':2000,'pair_end':2001,'end':2002})
        self.first = skexgen_to_sem(skexgen_item(5))
        self.second = skexgen_to_sem(skexgen_item(9)).replace('add','cut')

    def test_single_and_multiple_pairs_preserve_all_semantic_ids(self):
        for text in (self.first,self.first+' '+self.second):
            with self.subTest(text=text):
                encoded = self.codec.encode(text)
                self.assertEqual(self.codec.decode(encoded),stringify(tokenize(text)))
                self.assertEqual(len(encoded),len(tokenize(text))+len(self.codec.encode_pairs(text))+2)
                restored = LosslessSemCodec.from_dict(self.codec.to_dict())
                self.assertEqual(restored.encode(text),encoded)

    def test_unknown_ids_missing_boundaries_and_extra_tokens_fail(self):
        good = self.codec.encode(self.first)
        invalid = [good[1:],good[:-1],good[:-2]+[2002],good[:2]+[9999]+good[2:],
                   [2000,2002],good+[2],good[:1]+[2001]+good[1:]]
        for ids in invalid:
            with self.subTest(ids=ids),self.assertRaises(ValueError):
                self.codec.decode(ids)

    def test_mapping_must_not_collide_with_framing(self):
        with self.assertRaises(ValueError):
            LosslessSemCodec(tuple(range(1000,1076)),{'start':1000,'pair_end':2001,'end':2002})

    def test_each_generated_step_obeys_grammar_and_pair_limit(self):
        for text,limit in ((self.first,1),(self.first+' '+self.second,2)):
            constraint = LosslessCadConstraint(self.codec,max_pairs=limit)
            generated=[]
            for token in self.codec.encode(text)[1:]:
                self.assertIn(token,constraint.allowed(generated))
                generated.append(token)
            self.assertEqual(constraint.allowed(generated),set())
        prefix = self.codec.encode(self.first)[1:-1]
        self.assertEqual(LosslessCadConstraint(self.codec,max_pairs=1).allowed(prefix),{2002})

    def test_first_boolean_must_add_and_upper_depth_excludes_lower(self):
        pair = self.codec.encode_pairs(self.first)[0]
        operation = next(i for i,value in enumerate(tokenize(self.first)) if value in (8,9,10))
        constraint = LosslessCadConstraint(self.codec)
        self.assertEqual(constraint.allowed(pair[:operation]),{self.codec.native_ids[8-1]})
        prefix = pair[:operation+2]
        self.assertNotIn(pair[operation+1],constraint.allowed(prefix))

    def test_full_sample_masks_target_and_does_not_leak_target_into_prompt(self):
        source = self.codec.encode_pairs(self.first)
        first = self.codec.encode_pairs(self.first)
        second = self.codec.encode_pairs(self.second)
        a,pa = lossless_sample(TextTokenizer(),self.codec,'Modify the shape',source,first)
        b,pb = lossless_sample(TextTokenizer(),self.codec,'Modify the shape',source,second)
        self.assertEqual(pa,pb)
        self.assertEqual(a[:pa],b[:pb])
        self.assertNotEqual(a[pa:],b[pb:])
        self.assertEqual(self.codec.decode([2000]+a[pa:-1]),stringify(tokenize(self.first)))
        self.assertEqual(self.codec.decode([2000]+b[pb:-1]),stringify(tokenize(self.second)))
