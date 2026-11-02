import random
import unittest

from cadtokenizer.geometry_worker import parse_sem
from cadtokenizer.representation import (EOS, NUMBERS, ROTATIONS, Grammar, analyze, cross, orient,
                                         primitive_ids, skexgen_to_sem, split_pairs, stringify, tokenize)
from tests.fixtures import skexgen_item


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.text = skexgen_to_sem(skexgen_item(10))

    def test_source_layout_and_offsets(self):
        canonical, pairs = analyze(self.text)
        self.assertTrue(canonical.startswith('circle,10,10,12,12,14,10,12,8 <curve_end>'))
        self.assertIn('add,0,63,31,31,31,1,0,0,0,1,0,0,0,1,3,30,30', canonical)
        self.assertEqual(len(pairs), 1)

    def test_round_trip_and_legacy_spelling(self):
        ids = tokenize(self.text)
        self.assertEqual(tokenize(stringify(ids)), ids)
        self.assertEqual(tokenize(self.text.replace('<extrude_end>', '<extrusion_end>')), ids)

    def test_multiple_pairs_preserve_order(self):
        self.assertEqual(len(split_pairs(tokenize(self.text + ' ' + self.text))), 2)

    def test_end_markers_pool_with_curve(self):
        pair = split_pairs(tokenize(self.text))[0]
        groups = primitive_ids(pair)
        start = next(i for i, t in enumerate(pair) if t in (8, 9, 10))
        self.assertEqual(set(groups[:start]), {0})
        self.assertEqual([groups.count(i) for i in (1, 2, 3)], [6, 9, 4])
        self.assertEqual(max(primitive_ids(pair, 'rotation_rows')) + 1, 6)

    def test_rejects_malformed_and_out_of_range_sequences(self):
        for text in [self.text.replace('circle 10', 'circle 64', 1), self.text.replace('<loop_end>', ''),
                     self.text.replace('<extrude_end>', ''), self.text + ' garbage', 'pad ' + self.text]:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    analyze(text)

    def test_limits_fail_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, 'Primitive count'):
            analyze(self.text, max_primitives=3)
        with self.assertRaisesRegex(ValueError, 'Pair length'):
            analyze(self.text, max_pair_tokens=10)

    def test_loop_pooling_keeps_all_curves_and_trailing_markers_together(self):
        sketch = ('line 0 0 <curve_end> line 10 0 <curve_end> line 10 10 <curve_end> <loop_end> '
                  'line 2 2 <curve_end> line 3 2 <curve_end> line 3 3 <curve_end> <loop_end> <face_end> <sketch_end> ')
        extrusion = self.text.split('add')[1]
        pair = tokenize(sketch + 'add' + extrusion)
        groups = primitive_ids(pair, sketch_partition='loop')
        end_first = pair.index(5)
        boundary = pair.index(8)
        self.assertEqual(set(groups[:end_first + 1]), {0})
        self.assertEqual(set(groups[end_first + 1:boundary]), {1})
        self.assertEqual([groups.count(i) for i in (2, 3, 4)], [6, 9, 4])
        self.assertEqual(max(primitive_ids(pair)) + 1, 9)
        analyze(stringify(pair), max_primitives=5, sketch_partition='loop')
        with self.assertRaisesRegex(ValueError, 'Primitive count'):
            analyze(stringify(pair), max_primitives=5)

    def test_loop_budget_allows_more_curves_inside_existing_loop(self):
        grammar = Grammar(max_curves=20, max_loops=1)
        for token in tokenize('line 0 0 <curve_end> line 10 0 <curve_end> line 10 10 <curve_end>'):
            grammar.advance(token)
        self.assertIn(1, grammar.allowed())
        grammar.advance(5)
        self.assertEqual(grammar.allowed(), {6})
        grammar.advance(6)
        self.assertEqual(grammar.allowed(), {7})

    def test_grammar_completes_and_stays_within_curve_budget(self):
        rng = random.Random(2)
        for _ in range(50):
            grammar = Grammar(max_curves=3)
            generated = []
            while grammar.state != 'done':
                token = rng.choice(sorted(grammar.allowed()))
                grammar.advance(token)
                generated.append(token)
                self.assertLess(len(generated), 100)
            self.assertLessEqual(grammar.curves, 3)
            self.assertEqual(generated[-1], EOS)
            split_pairs(generated[:-1])

    def _drive_to_boolean(self, grammar):
        """One minimal line/loop/face/sketch, landing on the 'boolean' state."""
        grammar.advance(1)
        grammar.advance(13)
        grammar.advance(13)
        grammar.advance(4)
        grammar.advance(5)
        grammar.advance(6)
        grammar.advance(7)

    def test_force_add_restricts_first_boolean_operation(self):
        grammar = Grammar(force_add=True)
        self._drive_to_boolean(grammar)
        self.assertEqual(grammar.allowed(), {8})
        with self.assertRaises(ValueError):
            grammar.advance(9)
        grammar.advance(8)
        self.assertEqual(grammar.state, 'ext_number')

    def test_extrusion_bounds_forbid_non_increasing_upper_value(self):
        grammar = Grammar()
        self._drive_to_boolean(grammar)
        grammar.advance(8)
        grammar.advance(13 + 40)
        forbidden = grammar.numeric_forbidden()
        self.assertEqual(forbidden, {t for t in NUMBERS if t - 13 <= 40})
        self.assertNotIn(13 + 41, forbidden)

    def test_sketch_frame_zero_normal_forbidden_but_not_exhaustive(self):
        grammar = Grammar()
        self._drive_to_boolean(grammar)
        grammar.advance(8)
        grammar.advance(13)       # idx0 lower = 0
        grammar.advance(13 + 10)  # idx1 upper = 10
        grammar.advance(13)       # idx2 tx
        grammar.advance(13)       # idx3 ty
        grammar.advance(13)       # idx4 tz
        grammar.advance(14)       # idx5 x0 = 1
        grammar.advance(13)       # idx6 x1 = 0   -> x = (1, 0, 0)
        grammar.advance(13)       # idx7 x2 = 0
        grammar.advance(13)       # idx8 y0
        grammar.advance(13)       # idx9 y1
        grammar.advance(13)       # idx10 y2       -> y axis unguarded = (0, 0, 0)
        grammar.advance(13)       # idx11 z0 = 0
        grammar.advance(13)       # idx12 z1 = 0
        self.assertEqual(grammar.ext_index, 13)
        self.assertEqual(grammar.numeric_forbidden(), {13})
        grammar.advance(14)  # z = (0, 0, 1): non-degenerate frame
        self.assertEqual(grammar.ext_index, 14)

    def test_sketch_frame_guard_can_exhaust_rotation_domain_when_x_axis_degenerate(self):
        grammar = Grammar()
        self._drive_to_boolean(grammar)
        grammar.advance(8)
        grammar.advance(13)
        grammar.advance(13 + 10)
        grammar.advance(13)
        grammar.advance(13)
        grammar.advance(13)
        grammar.advance(13)
        grammar.advance(13)
        grammar.advance(14)  # x = (0, 0, 1), collinear with any (0, 0, t)
        grammar.advance(13)
        grammar.advance(13)
        grammar.advance(13)
        grammar.advance(13)
        grammar.advance(13)  # z0, z1 = 0, 0
        self.assertEqual(grammar.numeric_forbidden(), set(ROTATIONS))

    def test_zero_length_line_forbidden_only_when_x_already_matches(self):
        grammar = Grammar()
        grammar.advance(1)
        grammar.advance(13 + 5)
        grammar.advance(13 + 7)
        grammar.advance(4)
        self.assertEqual(grammar.state, 'after_curve')
        grammar.advance(1)
        grammar.advance(13 + 5)
        self.assertEqual(grammar.numeric_forbidden(), {13 + 7})

        other = Grammar()
        other.advance(1)
        other.advance(13 + 5)
        other.advance(13 + 7)
        other.advance(4)
        other.advance(1)
        other.advance(13 + 9)
        self.assertEqual(other.numeric_forbidden(), set())

    def test_degenerate_arc_forbidden(self):
        grammar = Grammar()
        grammar.advance(2)
        grammar.advance(13)
        grammar.advance(13)
        grammar.advance(13 + 10)
        grammar.advance(13)
        grammar.advance(4)
        grammar.advance(2)
        grammar.advance(13 + 20)
        self.assertEqual(grammar.numeric_forbidden(), {13})

    def test_loop_closing_edge_zero_length_line_forbids_loop_end(self):
        grammar = Grammar(max_curves=4)
        grammar.advance(1)
        grammar.advance(13 + 5)
        grammar.advance(13 + 7)
        grammar.advance(4)
        self.assertEqual(grammar.state, 'after_curve')
        self.assertIn(5, grammar.allowed())
        # A one-line "loop" always closes against its own single point.
        self.assertEqual(grammar.numeric_forbidden(), {5})
        grammar.advance(1)
        grammar.advance(13 + 9)
        grammar.advance(13 + 9)
        grammar.advance(4)
        self.assertEqual(grammar.state, 'after_curve')
        self.assertEqual(grammar.numeric_forbidden(), set())  # second point differs from the first curve's
        grammar.advance(1)
        grammar.advance(13 + 5)  # matches the first curve's (5, 7)
        grammar.advance(13 + 7)
        grammar.advance(4)
        self.assertEqual(grammar.numeric_forbidden(), {5})
        self.assertEqual(grammar.allowed() - grammar.numeric_forbidden(), {1, 2, 3})

    def test_loop_closing_edge_degenerate_arc_forbids_loop_end(self):
        grammar = Grammar(max_curves=4)
        grammar.advance(1)
        grammar.advance(13 + 20)
        grammar.advance(13)  # first curve: line point (20, 0)
        grammar.advance(4)
        grammar.advance(2)
        grammar.advance(13)
        grammar.advance(13)
        grammar.advance(13 + 10)
        grammar.advance(13)  # last curve: arc, start (0, 0), middle (10, 0): collinear with (20, 0)
        grammar.advance(4)
        self.assertEqual(grammar.numeric_forbidden(), {5})

    def test_numeric_guards_prevent_known_kernel_degeneracies(self):
        rng = random.Random(7)
        for _ in range(30):
            tokens, fallback_used = [], []
            for pair_index in range(rng.randint(1, 3)):
                grammar = Grammar(max_curves=3, force_add=(pair_index == 0))
                exhausted = False
                while grammar.state != 'done':
                    allowed = sorted(grammar.allowed())
                    forbidden = grammar.numeric_forbidden() & set(allowed)
                    if allowed and len(forbidden) == len(allowed):
                        exhausted, forbidden = True, set()
                    token = rng.choice([a for a in allowed if a not in forbidden])
                    grammar.advance(token)
                    tokens.append(token)
                tokens.pop()  # drop the internal EOS marker
                fallback_used.append(exhausted)
            for operation, pair_fallback_used in zip(parse_sem(stringify(tokens)), fallback_used):
                if pair_fallback_used:
                    # Every candidate was numerically bad at some position (tested
                    # directly above: e.g. a degenerate x-axis leaves no safe z). The
                    # guard documents this escape hatch instead of silently editing
                    # decoded geometry, so this pair's degeneracy checks don't apply.
                    continue
                p = operation['parameters']
                self.assertLess(p[0], p[1])
                self.assertTrue(any(p[11:14]))
                self.assertTrue(any(cross(p[5:8], p[11:14])))
                for face in operation['faces']:
                    for loop in face:
                        for ci, curve in enumerate(loop):
                            following = loop[(ci + 1) % len(loop)]
                            if curve['type'] == 'line':
                                self.assertNotEqual(curve['coordinates'][:2], following['coordinates'][:2])
                            if curve['type'] == 'arc':
                                start, middle = curve['coordinates'][:2], curve['coordinates'][2:4]
                                self.assertNotEqual(orient(start, middle, following['coordinates'][:2]), 0)
            self.assertEqual(parse_sem(stringify(tokens))[0]['operation'], 'add')
