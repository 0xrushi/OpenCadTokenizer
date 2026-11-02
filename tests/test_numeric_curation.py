"""Adversarial checks for accepting a restricted existing-caption diagnostic."""
import unittest

from cadtokenizer.numeric_curation import parse_program, verify


def sequence(height=8, scale=11, lower=31, operation='add'):
    profile='circle,31,53,31,9,53,31,9,31 <curve_end> <loop_end> <face_end> <sketch_end>'
    params=[lower,lower+height,31,31,31,1,0,0,0,1,0,0,0,1,scale,30,30]
    return profile+' '+operation+','+','.join(map(str,params))+' <extrude_end>'


class NumericCurationTests(unittest.TestCase):
    def test_correct_decrease(self):
        r=verify('Decrease first extrusion height by 6 units.',sequence(),sequence(2))
        self.assertTrue(r['accepted']);self.assertEqual(r['after_height_grid'],2)

    def test_wrong_sign(self):
        self.assertFalse(verify('Decrease first extrusion height by 6 units.',sequence(2),sequence())['accepted'])

    def test_wrong_amount(self):
        self.assertFalse(verify('Decrease first extrusion height by 5 units.',sequence(),sequence(2))['accepted'])

    def test_noop(self):
        self.assertFalse(verify('Increase first extrusion height by 0 units.',sequence(),sequence())['accepted'])

    def test_resize_not_allowed(self):
        self.assertFalse(verify('Decrease first extrusion height by 6 units.',sequence(),sequence(2,scale=12))['accepted'])

    def test_profile_change_not_allowed(self):
        target=sequence(2).replace('circle,31,53','circle,31,52')
        self.assertFalse(verify('Decrease first extrusion height by 6 units.',sequence(),target)['accepted'])

    def test_lower_plane_not_allowed(self):
        self.assertFalse(verify('Decrease first extrusion height by 6 units.',sequence(),sequence(2,lower=37))['accepted'])

    def test_boolean_not_allowed(self):
        self.assertFalse(verify('Decrease first extrusion height by 6 units.',sequence(),sequence(2,operation='cut'))['accepted'])

    def test_other_operation_not_allowed(self):
        self.assertFalse(verify('Decrease first extrusion height by 6 units.',sequence()+sequence(),sequence(2)+sequence(4))['accepted'])

    def test_ordinal_second(self):
        r=verify('Increase height of second cylinder by 2 units.',sequence()+sequence(2),sequence()+sequence(4))
        self.assertTrue(r['accepted']);self.assertEqual(r['operation_index'],1)

    def test_wrong_ordinal(self):
        self.assertFalse(verify('Increase height of first cylinder by 2 units.',sequence()+sequence(2),sequence()+sequence(4))['accepted'])

    def test_absolute_height(self):
        self.assertTrue(verify('Change first extrusion height to 2 units.',sequence(),sequence(2))['accepted'])

    def test_absolute_increase_direction(self):
        self.assertFalse(verify('Increase first extrusion height to 2 units.',sequence(),sequence(2))['accepted'])

    def test_unsupported_clause(self):
        self.assertFalse(verify('Decrease first extrusion height by 6 units and move right.',sequence(),sequence(2))['accepted'])

    def test_ambiguous_change_by(self):
        self.assertFalse(verify('Change first extrusion height by 6 units.',sequence(),sequence(2))['accepted'])

    def test_cylinder_noun_requires_additive(self):
        self.assertFalse(verify('Decrease first cylinder height by 6 units.',sequence(operation='cut'),sequence(2,operation='cut'))['accepted'])

    def test_coordinate_change_not_allowed(self):
        target=sequence(2).replace(',11,30,30',',11,31,30')
        self.assertFalse(verify('Decrease first extrusion height by 6 units.',sequence(),target)['accepted'])

    def test_invalid_bounds_and_syntax(self):
        self.assertFalse(verify('Increase first extrusion height by 60 units.',sequence(),sequence(68))['accepted'])
        with self.assertRaises(ValueError):parse_program(sequence()+'garbage')


if __name__=='__main__':
    unittest.main()
