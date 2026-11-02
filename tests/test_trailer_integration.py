import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import unittest.mock
from unittest.mock import patch

from cadtokenizer.common import model_runtime_available
if not model_runtime_available():
    raise unittest.SkipTest('ML tests require the authorized Unsloth or Vast.ai runtime')

import torch
from cadtokenizer.alignment import TRAILER_BINS, allocate_trailer_ids, trained_embedding_mask
from cadtokenizer.common import file_hash, save_checkpoint
from cadtokenizer.inference import NativeCadGrammar, decode_native_ids
from cadtokenizer.representation import analyze, skexgen_to_sem, split_pairs, stringify
from cadtokenizer.scale_origin_codec import metadata_for_pair
from tests.fixtures import config_for, make_sources, skexgen_item

VOCAB = 4096
BOUNDARIES = {'start': 10, 'pair_end': 11, 'end': 12}
CODES = list(range(100, 108))
TRAILER = list(range(200, 200 + TRAILER_BINS))


def allowed(processor, generated, remaining_budget=64):
    prompt = [1, 2, 3]
    scores = torch.zeros((1, VOCAB))
    processor.prompt_length = len(prompt)
    processor.max_new_tokens = len(generated) + remaining_budget
    out = processor(torch.tensor([prompt + generated]), scores)[0]
    return set(torch.nonzero(out == 0).flatten().tolist())


class TrailerAllocationTests(unittest.TestCase):
    def test_trailer_ids_are_disjoint_deterministic_and_high(self):
        blocked = [0, 1, 2, VOCAB - 1, VOCAB - 2]
        ids = allocate_trailer_ids(blocked, CODES, VOCAB)
        self.assertEqual(ids, allocate_trailer_ids(blocked, CODES, VOCAB))
        self.assertEqual(len(set(ids)), TRAILER_BINS)
        self.assertFalse(set(ids) & (set(blocked) | set(CODES)))
        self.assertEqual(ids[0], VOCAB - 3)

    def test_ineligible_ids_are_never_chosen(self):
        eligible = torch.ones(VOCAB, dtype=torch.bool)
        eligible[VOCAB - 100:] = False          # e.g. reserved tokens with untrained embeddings
        ids = allocate_trailer_ids([], CODES, VOCAB, eligible=eligible)
        self.assertTrue(all(i < VOCAB - 100 for i in ids))
        self.assertEqual(ids[0], VOCAB - 101)

    def test_untrained_embedding_rows_are_not_eligible(self):
        torch.manual_seed(0)
        inputs, outputs = torch.randn(300, 8), torch.randn(300, 8)
        inputs[250:] = 0                         # zero input embedding, like Llama-3 reserved tokens
        outputs[100] = 0
        mask = trained_embedding_mask(inputs, outputs)
        self.assertFalse(mask[250:].any())
        self.assertFalse(mask[100])
        self.assertTrue(mask[:100].all() and mask[101:250].all())

    def test_too_small_vocabulary_is_rejected(self):
        with self.assertRaises(ValueError):
            allocate_trailer_ids([], list(range(10)), 40)


class TrailerGrammarTests(unittest.TestCase):
    def processor(self, trailer=True, max_primitives=3):
        return NativeCadGrammar(CODES, BOUNDARIES, 3, max_primitives, 64, trailer_ids=TRAILER if trailer else None)

    def test_pair_is_codes_then_exactly_three_trailer_bins_then_boundary(self):
        g = self.processor()
        self.assertEqual(allowed(g, []), set(CODES))
        self.assertEqual(allowed(g, [100]), set(CODES) | set(TRAILER))
        self.assertEqual(allowed(g, [100, 101, 102]), set(TRAILER))          # primitive cap reached
        self.assertEqual(allowed(g, [100, 200]), set(TRAILER))
        self.assertEqual(allowed(g, [100, 200, 201]), set(TRAILER))
        self.assertEqual(allowed(g, [100, 200, 201, 202]), {BOUNDARIES['pair_end']})
        after = [100, 200, 201, 202, BOUNDARIES['pair_end']]
        self.assertEqual(allowed(g, after), set(CODES) | {BOUNDARIES['end']})

    def test_trailer_budget_reserved_before_new_codes_or_pairs(self):
        g = self.processor()
        # After a code, adding another needs room for 3 bins, a boundary and END.
        self.assertEqual(allowed(g, [100], remaining_budget=5), set(TRAILER))
        self.assertEqual(allowed(g, [100], remaining_budget=6), set(CODES) | set(TRAILER))
        after = [100, 200, 201, 202, BOUNDARIES['pair_end']]
        self.assertEqual(allowed(g, after, remaining_budget=5), {BOUNDARIES['end']})

    def test_without_trailer_behaviour_is_unchanged(self):
        g = self.processor(trailer=False)
        self.assertEqual(allowed(g, [100]), set(CODES) | {BOUNDARIES['pair_end']})
        self.assertEqual(allowed(g, [100, BOUNDARIES['pair_end']]), set(CODES) | {BOUNDARIES['end']})


class TrailerDecodeTests(unittest.TestCase):
    def setUp(self):
        _, pairs = analyze(skexgen_to_sem(skexgen_item(10)))
        self.pair = pairs[0]['tokens']
        self.bins = metadata_for_pair(self.pair)
        self.alignment = {'boundaries': BOUNDARIES, 'native_ids': torch.tensor(CODES), 'trailer_ids': TRAILER,
                          'reconstructed_code_vectors': torch.zeros((len(CODES), 4))}

    def decoded(self, generated, bins_in_reconstruction=None):
        scrubbed = list(self.pair)
        boundary = next(i for i, t in enumerate(scrubbed) if t in (8, 9, 10))
        scrubbed[boundary + 15:boundary + 18] = [13, 13, 13]      # decoder cannot know the literals
        vq = unittest.mock.Mock()
        vq.codebook.weight = torch.zeros((len(CODES), 4))
        vq.reconstruct.return_value = scrubbed
        return decode_native_ids(vq, self.alignment, generated, 'codebook')

    def test_trailer_literals_are_restored_into_the_decoded_pair(self):
        generated = [100, 101] + [TRAILER[v] for v in self.bins] + [BOUNDARIES['pair_end'], BOUNDARIES['end']]
        self.assertEqual(self.decoded(generated), stringify(self.pair))

    def test_malformed_trailers_are_rejected(self):
        t = [TRAILER[v] for v in self.bins]
        pair_end, end = BOUNDARIES['pair_end'], BOUNDARIES['end']
        for generated in ([100] + t[:2] + [pair_end, end],            # too few bins
                          [100] + t + [TRAILER[0], pair_end, end],    # too many
                          [100] + t[:1] + [101] + t[1:] + [pair_end, end],  # code after trailer
                          t + [pair_end, end],                        # trailer without code
                          [100] + [pair_end, end]):                   # trailer missing
            with self.assertRaises(ValueError):
                self.decoded(generated)

    def test_trailer_ids_must_not_overlap_codes(self):
        self.alignment['trailer_ids'] = [CODES[0]] + TRAILER[1:]
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.decoded([100] + [TRAILER[1]] * 3 + [BOUNDARIES['pair_end'], BOUNDARIES['end']])


class TrailerEncodeTests(unittest.TestCase):
    def test_encoded_native_pairs_append_the_exact_trailer(self):
        from cadtokenizer.encode import encode_geometry
        from cadtokenizer.prepare import prepare
        from cadtokenizer.vq import train_vq
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = config_for(root)
            config['data']['validation_fraction'] = 0.2
            config['vq'].update(epochs=1, batch_size=4, num_workers=0)
            make_sources(config)
            prepare(config)
            torch.manual_seed(3)
            train_vq(config)
            vq_hash = file_hash(Path(config['vq']['output']) / 'best.pt')
            native = torch.arange(300, 300 + config['vq']['codebook_size'])
            outputs = {}
            for name, trailer in (('plain', None), ('hybrid', TRAILER)):
                run = copy.deepcopy(config)
                run['alignment']['output'] = str(root / f'alignment-{name}')
                run['sft']['encoded_database'] = str(root / f'encoded-{name}.sqlite')
                save_checkpoint(Path(run['alignment']['output']) / 'alignment.pt',
                                {'vq_sha256': vq_hash, 'native_ids': native, 'trailer_ids': trailer})
                encode_geometry(run)
                with sqlite3.connect(run['sft']['encoded_database']) as db:
                    outputs[name] = {k: (json.loads(c), json.loads(n)) for k, c, n in
                                     db.execute('SELECT id,code_pairs,native_pairs FROM geometry')}
                    outputs[name + '_trailer'] = json.loads(dict(db.execute('SELECT key,value FROM metadata'))['trailer'])
            self.assertFalse(outputs['plain_trailer'])
            self.assertTrue(outputs['hybrid_trailer'])
            with sqlite3.connect(config['data']['database']) as db:
                sequences = dict(db.execute('SELECT id,sequence FROM geometry'))
            checked = 0
            for key, (codes, plain_native) in outputs['plain'].items():
                hybrid_codes, hybrid_native = outputs['hybrid'][key]
                self.assertEqual(codes, hybrid_codes)
                _, pairs = analyze(sequences[key])
                for plain, hybrid, pair in zip(plain_native, hybrid_native, pairs):
                    self.assertEqual(hybrid[:-3], plain)
                    self.assertEqual(hybrid[-3:], [TRAILER[v] for v in metadata_for_pair(pair['tokens'])])
                    checked += 1
            self.assertGreater(checked, 0)


if __name__ == '__main__':
    unittest.main()
