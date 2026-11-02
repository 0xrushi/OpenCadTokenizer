"""Benchmark definitions. Pure-Python reference functions can be tested on the host.

The CADFusion F1 compatibility calculation deliberately preserves its count-only
matching and conditional averaging. Geometry distances use symmetric squared CD.
See docs/EVALUATION.md for the source audit and protocol differences.
"""
from __future__ import annotations

import collections
import json
import math
import re


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def _text(sequence):
    return (sequence or '').replace('<extrusion_end>', '<extrude_end>')


def count_f1(reference_count, predicted_count, absent=None):
    if reference_count + predicted_count == 0:
        return absent
    return 2.0 * min(reference_count, predicted_count) / (reference_count + predicted_count)


def cadfusion_f1(cases):
    """Port of upstream f1_eval.py's math, without its hardcoded 1000-row loop.

Each case is (ground-truth SEM, predicted SEM or None). Parameters/coordinates are
NOT matched. Empty/non-positive type averages are excluded exactly as upstream.
"""
    buckets = {key: [] for key in ('line', 'arc', 'circle', 'extrusion')}
    for reference, prediction in cases:
        reference, prediction = _text(reference), _text(prediction)
        a, b = reference.count('<extrude_end>'), prediction.count('<extrude_end>')
        if min(a, b) > 0:
            buckets['extrusion'].append(count_f1(a, b))
        ref_pairs = reference.split('<extrude_end>')[:-1]
        pred_pairs = prediction.split('<extrude_end>')[:-1]
        pair_count = min(len(ref_pairs), len(pred_pairs))
        if not pair_count:
            continue
        for kind in ('line', 'arc', 'circle'):
            scores = []
            for ref, pred in zip(ref_pairs, pred_pairs):
                pattern = rf'{kind}.*?<curve_end>'
                a, b = len(re.findall(pattern, ref)), len(re.findall(pattern, pred))
                scores.append(count_f1(a, b) if min(a, b) else -1.0)
            score = sum(scores) / pair_count
            if score > 0:
                buckets[kind].append(score)
    averages = {key: mean(values) for key, values in buckets.items()}
    sketch = mean(averages[k] for k in ('line', 'arc', 'circle')) if all(averages[k] is not None for k in ('line', 'arc', 'circle')) else None
    return {'F1-Skt': sketch, 'F1-Ext': averages['extrusion'], 'type_means': averages,
            'included_counts': {key: len(value) for key, value in buckets.items()}}


def all_sample_count_f1(cases):
    """Additional count-based metric that includes failures as zero when GT has a type."""
    buckets = {key: [] for key in ('line', 'arc', 'circle', 'extrusion')}
    for reference, prediction in cases:
        reference, prediction = _text(reference), _text(prediction)
        for kind in buckets:
            pattern = '<extrude_end>' if kind == 'extrusion' else rf'{kind}.*?<curve_end>'
            a, b = len(re.findall(pattern, reference)), len(re.findall(pattern, prediction))
            score = count_f1(a, b)
            if score is not None:
                buckets[kind].append(score)
    averages = {key: mean(value) for key, value in buckets.items()}
    return {'F1-Skt': mean(averages[k] for k in ('line', 'arc', 'circle') if averages[k] is not None),
            'F1-Ext': averages['extrusion'], 'type_means': averages}


def normalize_points(points, policy='max_abs_per_shape'):
    points = [[float(v) for v in row] for row in points]
    if not points or any(len(row) != 3 or not all(math.isfinite(v) for v in row) for row in points):
        raise ValueError('Point cloud must be a nonempty finite N x 3 array')
    if policy == 'none':
        return points
    if policy != 'max_abs_per_shape':
        raise ValueError('Unknown point normalization policy')
    scale = max(abs(v) for row in points for v in row)
    if scale <= 0:
        raise ValueError('Degenerate zero-scale point cloud')
    return [[v / scale for v in row] for row in points]


def chamfer_squared_reference(a, b):
    """Small-cloud mathematical oracle; production uses scipy KD trees."""
    if not len(a) or not len(b):
        raise ValueError('Chamfer distance requires two nonempty clouds')
    def directed(source, target):
        return sum(min(sum((float(x)-float(y))**2 for x, y in zip(p, q)) for q in target) for p in source) / len(source)
    return directed(a, b) + directed(b, a)


def distance_matrix_statistics(matrix, sample_ids, reference_ids):
    if len(matrix) != len(sample_ids) or any(len(row) != len(reference_ids) for row in matrix):
        raise ValueError('Distance matrix dimensions do not match the ID lists')
    if len(set(sample_ids)) != len(sample_ids) or len(set(reference_ids)) != len(reference_ids):
        raise ValueError('Duplicate point-cloud IDs')
    if not sample_ids or not reference_ids:
        return {'CD': None, 'COV': None, 'MMD': None, 'paired_count': 0}
    if any(not math.isfinite(float(v)) or v < 0 for row in matrix for v in row):
        raise ValueError('Distances must be finite and nonnegative')
    closest = [min(range(len(reference_ids)), key=lambda j: float(row[j])) for row in matrix]
    cov = len(set(closest)) / len(reference_ids)
    mmd = mean(min(float(row[j]) for row in matrix) for j in range(len(reference_ids)))
    ref_index = {key: i for i, key in enumerate(reference_ids)}
    paired = [float(row[ref_index[key]]) for key, row in zip(sample_ids, matrix) if key in ref_index]
    return {'CD': mean(paired), 'COV': cov, 'MMD': mmd, 'paired_count': len(paired)}


def voxel_counts(clouds, resolution=28):
    """Nearest nodes of the un-clipped [-1,1]^3 grid used by CADFusion."""
    if resolution < 2:
        raise ValueError('Grid resolution must be >= 2')
    counts = collections.Counter()
    for cloud in clouds:
        for point in cloud:
            if len(point) != 3 or any(not math.isfinite(float(v)) or abs(v) > 1.00001 for v in point):
                raise ValueError('JSD requires finite points in [-1,1]^3')
            index = [max(0, min(resolution - 1, int(round((float(v) + 1) * (resolution - 1) / 2)))) for v in point]
            counts[(index[0] * resolution + index[1]) * resolution + index[2]] += 1
    return counts


def jsd_counts(a, b):
    total_a, total_b = sum(a.values()), sum(b.values())
    if total_a <= 0 or total_b <= 0:
        return None
    result = 0.0
    for key in set(a) | set(b):
        p, q = a.get(key, 0) / total_a, b.get(key, 0) / total_b
        middle = (p + q) / 2
        if p:
            result += 0.5 * p * math.log2(p / middle)
        if q:
            result += 0.5 * q * math.log2(q / middle)
    return max(0.0, result)


def geometry_scores(predictions, references, settings):
    """ID-aligned paired CD plus conditional and full-reference distributions.

Inputs are maps of sample ID -> normalized point cloud. No failed shape is
represented by a zero point cloud, and no positional/filename pairing is assumed.
"""
    import numpy as np
    from scipy.spatial import cKDTree
    from concurrent.futures import ThreadPoolExecutor
    pred_ids, ref_ids = sorted(predictions), sorted(references)
    if not pred_ids or not ref_ids:
        empty = {'CD': None, 'COV': None, 'MMD': None, 'JSD': None, 'paired_count': 0}
        return {'matched_valid': dict(empty), 'all_valid_references': dict(empty), 'paired_cd_by_id': {}}
    pred = [np.asarray(predictions[key], dtype=np.float64) for key in pred_ids]
    ref = [np.asarray(references[key], dtype=np.float64) for key in ref_ids]
    pred_trees, ref_trees = [cKDTree(x) for x in pred], [cKDTree(x) for x in ref]
    def row(i):
        distances = []
        for j in range(len(ref)):
            left = ref_trees[j].query(pred[i], k=1, workers=1)[0]
            right = pred_trees[i].query(ref[j], k=1, workers=1)[0]
            distances.append(float(np.mean(left**2) + np.mean(right**2)))
        return distances
    with ThreadPoolExecutor(max_workers=settings['distribution_workers']) as executor:
        matrix = []
        for index, values in enumerate(executor.map(row, range(len(pred)))):
            matrix.append(values)
            if index % max(1, len(pred) // 20) == 0:
                print(json.dumps({'distribution_rows': index + 1, 'generated_shapes': len(pred), 'reference_shapes': len(ref)}), flush=True)
    full = distance_matrix_statistics(matrix, pred_ids, ref_ids)
    full['JSD'] = jsd_counts(voxel_counts(pred, settings['grid_resolution']), voxel_counts(ref, settings['grid_resolution']))
    matching_ids = sorted(set(pred_ids) & set(ref_ids))
    pi, ri = {key: i for i, key in enumerate(pred_ids)}, {key: i for i, key in enumerate(ref_ids)}
    matched_matrix = [[matrix[pi[a]][ri[b]] for b in matching_ids] for a in matching_ids]
    matched = distance_matrix_statistics(matched_matrix, matching_ids, matching_ids)
    matched['JSD'] = jsd_counts(voxel_counts([pred[pi[key]] for key in matching_ids], settings['grid_resolution']),
                              voxel_counts([ref[ri[key]] for key in matching_ids], settings['grid_resolution'])) if matching_ids else None
    paired = {key: matrix[pi[key]][ri[key]] for key in matching_ids}
    return {'matched_valid': matched, 'all_valid_references': full, 'paired_cd_by_id': paired}
