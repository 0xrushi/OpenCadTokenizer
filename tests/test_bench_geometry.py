import os
from pathlib import Path
import tempfile
import unittest

from cadtokenizer.common import model_runtime_available
if not model_runtime_available():
    raise unittest.SkipTest('CAD kernel checks require the authorized Unsloth or Vast.ai runtime')

try:
    import cadquery
    import numpy as np
except ImportError:
    raise unittest.SkipTest('Run benchmark-setup and benchmark-test for CAD kernel checks')

from cadtokenizer.bench_smoke import rectangle_sem, circle_sem
from cadtokenizer.bench_metrics import chamfer_squared_reference, geometry_scores, normalize_points
from cadtokenizer.geometry_worker import build_solid, sample_mesh, parse_sem


class GeometryTests(unittest.TestCase):
    def settings(self):
        import json
        return json.loads(Path('configs/benchmark.json').read_text())

    def test_rectangle_volume_and_shorter_extrusion(self):
        original = build_solid(rectangle_sem(), self.settings())
        shorter = build_solid(rectangle_sem(upper=45), self.settings())
        self.assertTrue(original.isValid())
        self.assertTrue(shorter.isValid())
        self.assertAlmostEqual(shorter.Volume() / original.Volume(), 45/63, places=5)

    def test_circle_and_boolean_hole(self):
        self.assertTrue(build_solid(circle_sem(), self.settings()).isValid())
        cut = rectangle_sem() + ' ' + rectangle_sem(operation='cut', low=20, high=40)
        original = build_solid(rectangle_sem(), self.settings())
        hollow = build_solid(cut, self.settings())
        self.assertAlmostEqual(hollow.Volume() / original.Volume(), 0.75, places=5)

    def test_invalid_extrusion_and_frame_are_rejected(self):
        with self.assertRaises(ValueError):
            build_solid(rectangle_sem(upper=0), self.settings())
        broken = rectangle_sem().replace('1,0,0,0,1,0,0,0,1,45', '0,0,0,0,1,0,0,0,0,45')
        with self.assertRaises(ValueError):
            build_solid(broken, self.settings())

    def test_surface_sampling_is_seeded_and_area_weighted(self):
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [12, 0, 0], [10, 2, 0]]
        triangles = [[0, 1, 2], [3, 4, 5]]
        a = sample_mesh(vertices, triangles, 10000, 42)
        b = sample_mesh(vertices, triangles, 10000, 42)
        self.assertTrue(np.array_equal(a, b))
        self.assertAlmostEqual(float((a[:, 0] > 5).mean()), 0.8, delta=0.02)

    def test_kdtree_metrics_match_small_cloud_oracle(self):
        a, b = [[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 1, 0]]
        result = geometry_scores({'a': a}, {'a': b}, self.settings())
        self.assertAlmostEqual(result['matched_valid']['CD'], chamfer_squared_reference(a, b))
        self.assertEqual(result['matched_valid']['COV'], 1)
