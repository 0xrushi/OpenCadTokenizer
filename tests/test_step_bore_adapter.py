"""Pure request-contract checks; geometry/control checks run in the image study."""
import unittest
from cadtokenizer.step_bore_adapter import public_request,same_bore,continued_cylinders

PROMPT='Revise the vertical through port at (x,y)=(24.5,24), currently diameter 18 mm, to diameter value. Preserve independent cuts.'


class StepBoreAdapterTests(unittest.TestCase):
    def test_units_and_center(self):
        r=public_request(PROMPT,9.9)
        self.assertEqual(r['center_xy_mm'],[24.5,24]);self.assertEqual(r['old_diameter_mm'],18)
        self.assertEqual(r['diameter_mm'],9.9)

    def test_no_implicit_parameters(self):
        with self.assertRaises(ValueError):public_request('Make the hole smaller',9.9)

    def test_invalid_values(self):
        for value in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):public_request(PROMPT,value)

    def test_independent_cut_contract_required(self):
        with self.assertRaises(ValueError):public_request(PROMPT.replace('independent','several'),9.9)

    def test_match_axis_radius_and_center(self):
        r=public_request(PROMPT,9.9)
        face={'type':'CYLINDER','concave':True,'axis':[0,0,-1],'origin_mm':[24.5,24,999],'radius_mm':9}
        self.assertTrue(same_bore(face,r))
        self.assertFalse(same_bore({**face,'radius_mm':8},r))
        self.assertFalse(same_bore({**face,'axis':[1,0,0]},r))
        self.assertFalse(same_bore({**face,'concave':False},r))
        self.assertFalse(same_bore({**face,'origin_mm':[25,24,999]},r))

    def test_torus_requires_public_full_ring_statement(self):
        self.assertFalse(public_request(PROMPT,9.9)['full_toroidal_passage'])
        self.assertTrue(public_request(PROMPT+' The passage is a full circular ring.',9.9)['full_toroidal_passage'])

    def test_passage_continues_across_missing_wall_interval(self):
        face={'index':1,'type':'CYLINDER','concave':True,'axis':[1,0,0],'origin_mm':[-1,24,15.3],'radius_mm':6,'v_bounds':[1,19]}
        other={**face,'index':2,'v_bounds':[33,71]}
        merged=continued_cylinders([face,other])
        self.assertEqual(len(merged),1);self.assertEqual(merged[0]['v_bounds'],[0,70])

    def test_axis_sign_and_origin_do_not_change_cylinder_identity(self):
        face={'index':1,'type':'CYLINDER','concave':True,'axis':[1,0,0],'origin_mm':[-1,24,15.3],'radius_mm':6,'v_bounds':[1,19]}
        other={**face,'index':2,'axis':[-1,0,0],'origin_mm':[100,24,15.3],'v_bounds':[30,68]}
        self.assertEqual(len(continued_cylinders([face,other])),1)
        self.assertEqual(len(continued_cylinders([face,{**other,'radius_mm':5}])),2)


if __name__=='__main__':unittest.main()
