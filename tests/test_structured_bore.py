import json
import unittest
from cadtokenizer.structured_bore import baseline,canonical_instruction,complete_object,parse,same_intent,validate

PUBLIC={'instruction':'Resize the vertical through-bore at (x,y)=(24.5,24) from diameter 18 mm to value mm. Preserve independent cuts.','requested_value_mm':9.9}


class StructuredBoreTests(unittest.TestCase):
    def test_baseline_roundtrip(self):
        value=baseline(PUBLIC);self.assertTrue(same_intent(parse(json.dumps(value)),value))

    def test_extra_keys_and_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):validate({**baseline(PUBLIC),'source_spec':{}})
        text=json.dumps(baseline(PUBLIC)).replace('"units": "mm"','"units": "mm", "units": "cm"')
        with self.assertRaises(ValueError):parse(text)

    def test_no_fence_or_prose_repair(self):
        text=json.dumps(baseline(PUBLIC))
        for value in ['```json\n'+text+'\n```',text+' next question','Answer: '+text]:
            with self.assertRaises(ValueError):parse(value)

    def test_numeric_and_boolean_types(self):
        for key,value in [('diameter_mm',True),('diameter_mm',0),('diameter_mm',float('inf')),('full_toroidal_passage',1),('preserve_other_features',False),('units','cm')]:
            with self.assertRaises(ValueError):validate({**baseline(PUBLIC),key:value})

    def test_nonfinite_json_rejected(self):
        with self.assertRaises(ValueError):parse(json.dumps({**baseline(PUBLIC),'diameter_mm':float('nan')}))

    def test_each_predicted_slot_is_used(self):
        x={**baseline(PUBLIC),'center_xy_mm':[33,27],'old_diameter_mm':20,'full_toroidal_passage':True}
        text=canonical_instruction(x)
        self.assertIn('(33,27)',text);self.assertIn('diameter 20 mm',text);self.assertIn('full circular ring',text)
        self.assertFalse(same_intent(x,baseline(PUBLIC)))

    def test_exact_match_checks_new_value_and_flag(self):
        x=baseline(PUBLIC)
        self.assertFalse(same_intent({**x,'diameter_mm':19.8},x))
        self.assertFalse(same_intent({**x,'full_toroidal_passage':True},x))

    def test_stopping_ignores_quoted_braces(self):
        self.assertFalse(complete_object('{"x":"}"'))
        self.assertTrue(complete_object('{"x":"}","nested":{}}'))
        self.assertFalse(complete_object('Output: {}'))


if __name__=='__main__':unittest.main()
