import unittest
from cadtokenizer.scale_origin_codec import HYBRID,LOSSLESS,VQ,decode,lossless_packet,metadata_for_pair,pack,unpack
from cadtokenizer.representation import split_pairs,stringify,tokenize

SEM='line,9,9 <curve_end> line,53,9 <curve_end> line,53,53 <curve_end> line,9,53 <curve_end> <loop_end> <face_end> <sketch_end> add,31,39,31,31,31,1,0,0,0,1,0,0,0,1,18,17,44 <extrude_end>'


class ScaleOriginCodecTests(unittest.TestCase):
    def test_lossless_roundtrip_without_decoder(self):
        self.assertEqual(tokenize(decode(lossless_packet(SEM))),tokenize(SEM))

    def test_three_bins_are_inside_packet(self):
        packet=pack(HYBRID,2048,[[2,800,0]],[[18,17,44]])
        parsed=unpack(packet)
        self.assertEqual(parsed['metadata'],[[18,17,44]])
        self.assertEqual(parsed['token_count'],2+1+3+3)
        self.assertEqual(parsed['serialized_bytes'],len(packet))

    def test_decode_has_no_source_input_or_encoder_state(self):
        # The callback supplies an intentionally different metadata prediction.
        wrong=tokenize(SEM.replace(',18,17,44',',2,3,4'))
        packet=pack(HYBRID,16,[[5,9]],[[18,17,44]])
        calls=[]
        def decoder(codes,first):calls.append((codes,first));return wrong
        self.assertEqual(tokenize(decode(bytes(packet),decoder,16)),tokenize(SEM))
        self.assertEqual(calls,[([5,9],True)])

    def test_plain_vq_does_not_replace_values(self):
        wrong=tokenize(SEM.replace(',18,17,44',',2,3,4'))
        packet=pack(VQ,16,[[5,9]])
        self.assertEqual(tokenize(decode(packet,lambda c,f:wrong)),wrong)

    def test_metadata_change_changes_only_three_fields(self):
        original=tokenize(SEM)
        packet=pack(HYBRID,16,[[1]],[[0,63,1]])
        out=tokenize(decode(packet,lambda c,f:original))
        self.assertEqual(sum(a!=b for a,b in zip(out,original)),3)

    def test_crc_truncation_and_kind_rejected(self):
        packet=pack(HYBRID,16,[[1]],[[0,0,63]])
        for broken in [packet[:-1],packet[:12]+bytes([packet[12]^1])+packet[13:],b'bad']:
            with self.assertRaises(ValueError):unpack(broken)
        with self.assertRaises(ValueError):pack(3,16,[[1]])

    def test_bad_or_unaccounted_metadata_rejected(self):
        for value in [None,[[0,1]],[[0,1,64]],[[0,1,True]]]:
            with self.assertRaises(ValueError):pack(HYBRID,16,[[1]],value)
        with self.assertRaises(ValueError):pack(VQ,16,[[1]],[[0,1,2]])

    def test_wrong_codebook_rejected(self):
        with self.assertRaises(ValueError):decode(pack(HYBRID,16,[[1]],[[0,0,0]]),lambda c,f:tokenize(SEM),32)

    def test_pair_boundaries_and_first_operation_flag(self):
        packet=pack(HYBRID,16,[[1],[2]],[[18,17,44],[18,17,44]])
        flags=[]
        def callback(codes,first):flags.append(first);return tokenize(SEM if first else SEM.replace('add,','cut,'))
        self.assertEqual(len(split_pairs(tokenize(decode(packet,callback)))),2)
        self.assertEqual(flags,[True,False])

    def test_source_metadata_extraction(self):
        self.assertEqual(metadata_for_pair(tokenize(SEM)),[18,17,44])


if __name__=='__main__':unittest.main()
