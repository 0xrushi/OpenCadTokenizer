import unittest

from cadtokenizer.geometry_worker import parse_sem
from scripts.audit_codebook_failures import flags_for, probe_variant, proper_cross, serialize


def sequence(points, operation='add', bounds=(10, 40)):
    curves=[{'type':'line','coordinates':list(p)} for p in points]
    return serialize([{'operation':operation,'faces':[[curves]],
                       'parameters':list(bounds)+[32]*3+[1,0,0,0,1,0,0,0,1]+[32,32,32]}])


class FailureAuditTests(unittest.TestCase):
    def test_roundtrip(self):
        text=sequence([(0,0),(40,0),(40,40),(0,40)])
        self.assertEqual(parse_sem(serialize(parse_sem(text))),parse_sem(text))
        self.assertEqual(flags_for(text),[])

    def test_crossing_and_touch(self):
        self.assertTrue(proper_cross((0,0),(40,40),(0,40),(40,0)))
        self.assertFalse(proper_cross((0,0),(40,0),(40,0),(40,40)))
        flags=flags_for(sequence([(0,0),(40,40),(0,40),(40,0)]))
        self.assertIn('proper_line_loop_self_intersection',{f['kind'] for f in flags})

    def test_bounds_probe_is_private(self):
        text=sequence([(0,0),(40,0),(40,40)],bounds=(40,10))
        changed,edits=probe_variant(text,'bounds')
        self.assertEqual(parse_sem(text)[0]['parameters'][:2],[40,10])
        self.assertEqual(parse_sem(changed)[0]['parameters'][:2],[10,40])
        self.assertEqual(len(edits),1)

    def test_degenerate_line_probe(self):
        text=sequence([(0,0),(0,0),(40,0),(40,40),(0,40)])
        self.assertIn('zero_length_line',{f['kind'] for f in flags_for(text)})
        changed,_=probe_variant(text,'degenerate_curves')
        self.assertEqual(len(parse_sem(changed)[0]['faces'][0][0]),4)
        self.assertEqual(flags_for(changed),[])

    def test_noop_and_initial_boolean(self):
        text=sequence([(0,0),(40,0),(40,40)],operation='cut')
        self.assertEqual(probe_variant(text,'bounds'),(None,[]))
        changed,_=probe_variant(text,'initial_add')
        self.assertEqual(parse_sem(changed)[0]['operation'],'add')
        self.assertEqual(parse_sem(text)[0]['operation'],'cut')
