import unittest

from cadtokenizer.addition_curation import caption_claims, profile, verify


def operation(sketch, boolean='add', height=8, scale=18):
    fields=[31,31+height,31,31,31,1,0,0,0,1,0,0,0,1,scale,30,30]
    return sketch+' '+boolean+','+','.join(map(str,fields))+' <extrude_end>'


RECT=' '.join('line,'+str(x)+','+str(y)+' <curve_end>' for x,y in [(9,9),(53,9),(53,53),(9,53)])+' <loop_end> <face_end> <sketch_end>'
CIRCLE='circle,31,53,31,9,53,31,9,31 <curve_end> <loop_end> <face_end> <sketch_end>'
SOURCE=operation(RECT)


class AdditionCurationTests(unittest.TestCase):
    def check(self,text,tool):
        return verify(text,SOURCE,SOURCE+' '+tool)

    def test_matching_rectangle(self):
        self.assertTrue(self.check('Add a 44x44 rectangular prism.',operation(RECT))['accepted'])

    def test_matching_center(self):
        self.assertTrue(self.check('Add a 44x44 rectangular base centered at (31, 31).',operation(RECT))['accepted'])

    def test_wrong_center(self):
        self.assertFalse(self.check('Add a 44x44 rectangular base centered at (9, 9).',operation(RECT))['accepted'])

    def test_no_corner_fallback(self):
        self.assertFalse(self.check('Add a 44x44 rectangular prism at (9, 9).',operation(RECT))['accepted'])

    def test_wrong_dimension(self):
        self.assertFalse(self.check('Add a 44x43 rectangular prism.',operation(RECT))['accepted'])

    def test_height_correct(self):
        self.assertTrue(self.check('Add a 44x44x8 rectangular prism.',operation(RECT))['accepted'])

    def test_height_wrong(self):
        self.assertFalse(self.check('Add a 44x44x7 rectangular prism.',operation(RECT))['accepted'])

    def test_with_height(self):
        self.assertTrue(self.check('Add a rectangular prism with 44x44 base and height 8 units.',operation(RECT))['accepted'])

    def test_hole_requires_cut(self):
        self.assertFalse(self.check('Add a 44x44 rectangular hole.',operation(RECT))['accepted'])
        self.assertTrue(self.check('Add a 44x44 rectangular hole.',operation(RECT,'cut'))['accepted'])

    def test_cannot_cut_solid(self):
        self.assertFalse(self.check('Add a 44x44 rectangular prism.',operation(RECT,'cut'))['accepted'])

    def test_conflicting_boolean_clause(self):
        self.assertFalse(self.check('Add a 44x44 rectangular prism with a cut operation.',operation(RECT))['accepted'])

    def test_radius_and_diameter(self):
        self.assertTrue(self.check('Add a circular hole with radius 22 units.',operation(CIRCLE,'cut'))['accepted'])
        self.assertTrue(self.check('Add a 44-unit diameter cylindrical hole.',operation(CIRCLE,'cut'))['accepted'])
        self.assertFalse(self.check('Add a circular hole with radius 9 units.',operation(CIRCLE,'cut'))['accepted'])

    def test_circle_boundary_not_center(self):
        self.assertFalse(self.check('Add a 22-unit radius hole at (31,53).',operation(CIRCLE,'cut'))['accepted'])

    def test_circle_bad_diameters(self):
        self.assertIsNone(profile(CIRCLE.replace('31,9','31,10')))

    def test_no_added_loops(self):
        sketch=RECT.replace('<loop_end>','<loop_end> '+CIRCLE.split(' <face_end>')[0])
        self.assertFalse(self.check('Add a 44x44 rectangular prism.',operation(sketch))['accepted'])

    def test_source_cannot_change(self):
        altered=operation(RECT,height=9)+' '+operation(RECT)
        self.assertFalse(verify('Add a 44x44 rectangular prism.',SOURCE,altered)['accepted'])

    def test_cannot_append_two(self):
        self.assertFalse(self.check('Add a 44x44 rectangular prism.',operation(RECT)+' '+operation(RECT))['accepted'])

    def test_unsupported_spatial_claim(self):
        self.assertIsNone(caption_claims('Add a 44x44 rectangular prism on top.'))

    def test_bowtie_rejected(self):
        sketch=RECT.replace('line,53,9 <curve_end> line,53,53','line,53,53 <curve_end> line,53,9')
        self.assertIsNone(profile(sketch))

    def test_zero_dimensions(self):
        self.assertFalse(self.check('Add a 0x44 rectangular prism.',operation(RECT))['accepted'])

    def test_target_not_unique(self):
        self.assertFalse(self.check('Add a 44x44 rectangular prism.',operation(RECT))['unique_reference_target_certified'])


if __name__=='__main__':unittest.main()
