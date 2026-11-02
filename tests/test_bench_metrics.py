import math
import json
from pathlib import Path
import unittest

from cadtokenizer.bench_metrics import (all_sample_count_f1, cadfusion_f1, chamfer_squared_reference,
                                        distance_matrix_statistics, jsd_counts, normalize_points, voxel_counts)
from cadtokenizer.bench_score import aggregate_task, comparison_markdown, display_metrics


SEM = ('line,1,1 <curve_end> arc,2,2,3,3 <curve_end> '
       'circle,1,2,3,4,5,6,7,8 <curve_end> <loop_end> <face_end> <sketch_end> '
       'add,0,63,31,31,31,1,0,0,0,1,0,0,0,1,40,31,31 <extrude_end>')


class BenchmarkMetricTests(unittest.TestCase):
    def test_upstream_count_f1_is_geometry_blind(self):
        changed_geometry = SEM.replace('line,1,1', 'line,50,50')
        result = cadfusion_f1([(SEM, changed_geometry)])
        self.assertEqual(result['F1-Skt'], 1)
        self.assertEqual(result['F1-Ext'], 1)

    def test_failure_exclusion_is_visible_and_strict_alternative_penalizes_it(self):
        cases = [(SEM, SEM), (SEM, None)]
        self.assertEqual(cadfusion_f1(cases)['F1-Skt'], 1)
        self.assertEqual(cadfusion_f1(cases)['included_counts']['extrusion'], 1)
        self.assertEqual(all_sample_count_f1(cases)['F1-Skt'], 0.5)
        self.assertEqual(all_sample_count_f1(cases)['F1-Ext'], 0.5)

    def test_missing_curve_types_are_not_invented(self):
        result = cadfusion_f1([('line,1,1 <curve_end> <extrude_end>', 'line,1,1 <curve_end> <extrude_end>')])
        self.assertIsNone(result['F1-Skt'])
        self.assertEqual(result['F1-Ext'], 1)

    def test_chamfer_uses_squared_distance_and_both_directions(self):
        self.assertEqual(chamfer_squared_reference([[0, 0, 0]], [[2, 0, 0]]), 8)
        self.assertEqual(chamfer_squared_reference([[1, 2, 3]], [[1, 2, 3]]), 0)

    def test_matrix_statistics_are_id_aligned_not_diagonal_by_position(self):
        result = distance_matrix_statistics([[9, 0], [0, 9]], ['b', 'a'], ['a', 'b'])
        self.assertEqual(result['CD'], 0)
        self.assertEqual(result['COV'], 1)
        self.assertEqual(result['MMD'], 0)

    def test_cov_and_mmd_have_reference_denominators(self):
        result = distance_matrix_statistics([[0, 10], [0, 20]], ['a', 'b'], ['a', 'b'])
        self.assertEqual(result['COV'], 0.5)
        self.assertEqual(result['MMD'], 5)
        self.assertEqual(result['CD'], 10)

    def test_distribution_empty_sets_are_null_not_success(self):
        result = distance_matrix_statistics([], [], ['a'])
        self.assertIsNone(result['CD'])
        self.assertIsNone(result['COV'])
        self.assertIsNone(jsd_counts({}, {0: 1}))

    def test_bad_distances_are_rejected(self):
        for distance in [-1, math.nan, math.inf]:
            with self.assertRaises(ValueError):
                distance_matrix_statistics([[distance]], ['a'], ['a'])

    def test_normalization_matches_max_absolute_reference(self):
        self.assertEqual(normalize_points([[-2, 1, 0]]), [[-1, 0.5, 0]])
        with self.assertRaises(ValueError):
            normalize_points([[0, 0, 0]])

    def test_jsd_uses_base_two_and_the_full_grid(self):
        a = voxel_counts([[[-1, -1, -1]]], 2)
        b = voxel_counts([[[1, 1, 1]]], 2)
        self.assertEqual(a, {0: 1})
        self.assertEqual(b, {7: 1})
        self.assertEqual(jsd_counts(a, a), 0)
        self.assertEqual(jsd_counts(a, b), 1)

    def test_units_match_paper_display(self):
        table = display_metrics({'F1-Skt': 0.8, 'CD': 0.135, 'COV': 0.524, 'JSD': 0.0254, 'IR': 0.0838, 'HE': 1.72})
        self.assertEqual(table['F1-Skt'], 80)
        self.assertAlmostEqual(table['CD'], 13.5)
        self.assertAlmostEqual(table['COV'], 52.4)
        self.assertAlmostEqual(table['JSD'], 2.54)
        self.assertAlmostEqual(table['IR'], 8.38)
        self.assertEqual(table['HE'], 1.72)
        self.assertIsNone(table['VLM'])

    def test_every_failure_remains_in_invalidity_denominator(self):
        cases = []
        for i, status in enumerate(['valid', 'timeout', 'inference_error', 'syntax_invalid']):
            cases.append({'sample_id': str(i), 'task': 'editing', 'target_sequence': SEM,
                          'prediction_sequence': SEM if i == 0 else None,
                          'prediction_geometry': {'status': status}, 'reference_geometry': {'status': 'valid'},
                          'exact_sequence': i == 0, 'copied_source': False})
        geometry = {'matched_valid': {'CD': 0, 'COV': 1, 'JSD': 0, 'MMD': 0, 'paired_count': 1},
                    'all_valid_references': {'COV': 0.25}}
        result = aggregate_task(cases, geometry, {}, {})
        self.assertEqual(result['attempted'], 4)
        self.assertEqual(result['invalid_predictions'], 3)
        self.assertEqual(result['table_metrics']['IR'], 75)
        self.assertIsNone(result['table_metrics']['VLM'])
        self.assertIsNone(result['table_metrics']['HE'])
        references = json.loads(Path('configs/paper_reference.json').read_text())
        rendered = comparison_markdown({'manifest_sha256': 'test-manifest', 'tasks': {'editing': result},
                                         'comparability_notes': ['Fixture only.']}, references)
        self.assertIn('provisional', rendered)
        self.assertIn('not scored', rendered)
        self.assertIn('75.0000', rendered)
