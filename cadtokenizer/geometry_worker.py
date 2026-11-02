"""Isolated OpenCascade reconstruction worker, launched inside Unsloth Docker.

Uses CadQuery/OCP to avoid requiring a second Docker image or a host pythonocc
environment. SEM dequantization follows CADFusion's released parser. Kernel repair
and meshing differ from its pythonocc backend; reports label that difference.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import traceback

from .common import require_container, write_json
from .representation import NAMES, split_pairs, tokenize


def parse_sem(sequence):
    """Return faces/loops/curves and the 18-field extrusion, without ML imports."""
    result = []
    for pair in split_pairs(tokenize(sequence)):
        boundary = next(i for i, token in enumerate(pair) if token in (8, 9, 10))
        faces, loops, curves = [], [], []
        i = 0
        while i < boundary:
            token = pair[i]
            if token in (1, 2, 3):
                count = {1: 2, 2: 4, 3: 8}[token]
                curves.append({'type': NAMES[token], 'coordinates': [int(NAMES[v]) for v in pair[i+1:i+1+count]]})
                i += count + 2  # identifier, coordinates, curve_end
            elif token == 5:
                loops.append(curves)
                curves = []
                i += 1
            elif token == 6:
                faces.append(loops)
                loops = []
                i += 1
            elif token == 7:
                i += 1
            else:
                raise ValueError('Unexpected SEM sketch token')
        if curves or loops or not faces:
            raise ValueError('Incomplete sketch topology')
        result.append({'faces': faces, 'operation': NAMES[pair[boundary]],
                       'parameters': [int(NAMES[v]) for v in pair[boundary+1:-1]]})
    return result


def dequantize(value, low, high):
    return float(value) * (high - low) / 63.0 + low


def build_solid(sequence, settings):
    import cadquery as cq
    import numpy as np
    shape = None
    for operation in parse_sem(sequence):
        p = operation['parameters']
        lower, upper = [dequantize(v, -1, 1) for v in p[:2]]
        if not lower < upper:
            raise ValueError('Extrusion lower plane must precede upper plane')
        translation = np.array([dequantize(v, -1, 1) for v in p[2:5]])
        raw_axes = np.array(p[5:14], dtype=float).reshape(3, 3)
        # CADFusion constructs gp_Ax3 from z and x; the supplied y row is ignored.
        # Reproduce that frame construction rather than silently imposing a
        # stricter nine-parameter orthonormality test on generated sequences.
        normal = raw_axes[2]
        if np.linalg.norm(normal) < 1e-10:
            raise ValueError('Zero sketch normal')
        normal = normal / np.linalg.norm(normal)
        x_axis = raw_axes[0] - np.dot(raw_axes[0], normal) * normal
        if np.linalg.norm(x_axis) < 1e-10:
            raise ValueError('Sketch x-axis is parallel to its normal')
        x_axis = x_axis / np.linalg.norm(x_axis)
        axes = np.array([x_axis, np.cross(normal, x_axis), normal])
        scale = dequantize(p[14], 0, 1.4)
        offset = np.array([dequantize(v, -0.9, 0.9) for v in p[15:17]])
        if scale <= 0:
            raise ValueError('Sketch scale must be positive')
        def local_point(coordinates):
            return np.array([dequantize(v, -1, 1) for v in coordinates])
        def world_point(point):
            point = point * scale + offset
            return tuple(translation + point[0] * axes[0] + point[1] * axes[1] + lower * normal)
        solids = []
        for face in operation['faces']:
            wires = []
            for loop in face:
                if len(loop) == 1 and loop[0]['type'] == 'circle':
                    points = np.array([local_point(loop[0]['coordinates'][i:i+2]) for i in range(0, 8, 2)])
                    center = np.array([(points[0, 0] + points[1, 0]) / 2, (points[2, 1] + points[3, 1]) / 2])
                    radius = (np.linalg.norm(points[0]-points[1]) + np.linalg.norm(points[2]-points[3])) / 4 * scale
                    if radius <= 0:
                        raise ValueError('Degenerate circle')
                    wire = cq.Wire.makeCircle(float(radius), world_point(center), tuple(normal))
                else:
                    edges = []
                    for index, curve in enumerate(loop):
                        following = loop[(index + 1) % len(loop)]
                        if curve['type'] == 'circle' or following['type'] == 'circle':
                            raise ValueError('A full circle must form its own loop')
                        start = world_point(local_point(curve['coordinates'][:2]))
                        end = world_point(local_point(following['coordinates'][:2]))
                        if curve['type'] == 'line':
                            edge = cq.Edge.makeLine(start, end)
                        else:
                            middle = world_point(local_point(curve['coordinates'][2:4]))
                            edge = cq.Edge.makeThreePointArc(start, middle, end)
                        edges.append(edge)
                    wire = cq.Wire.assembleEdges(edges)
                if not wire.isValid() or not wire.IsClosed():
                    raise ValueError('Invalid or open sketch wire')
                wires.append(wire)
            face_shape = cq.Face.makeFromWires(wires[0], wires[1:])
            if not face_shape.isValid():
                raise ValueError('Invalid planar face or hole')
            solid = cq.Solid.extrudeLinear(wires[0], wires[1:], tuple(normal * (upper-lower)))
            if not solid.isValid() or not solid.Solids():
                raise ValueError('Extrusion did not produce a valid solid')
            solids.append(solid)
        body = solids[0]
        for other in solids[1:]:
            body = body.fuse(other, tol=settings['boolean_tolerance'])
        op = operation['operation']
        if shape is None:
            if op != 'add':
                raise ValueError('The initial extrusion must add material')
            shape = body
        elif op == 'add':
            shape = shape.fuse(body, tol=settings['boolean_tolerance'])
        elif op == 'cut':
            shape = shape.cut(body, tol=settings['boolean_tolerance'])
        else:
            shape = shape.intersect(body, tol=settings['boolean_tolerance'])
        if not shape.isValid() or not shape.Solids() or shape.Volume() <= settings['minimum_volume']:
            raise ValueError('Boolean operation produced an invalid or empty solid')
    return shape


def sample_mesh(vertices, triangles, count, seed):
    import numpy as np
    vertices, triangles = np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError('Invalid tessellation')
    tris = vertices[triangles]
    areas = np.linalg.norm(np.cross(tris[:, 1]-tris[:, 0], tris[:, 2]-tris[:, 0]), axis=1) / 2
    keep = np.isfinite(areas) & (areas > 0)
    tris, areas = tris[keep], areas[keep]
    if not len(areas) or not np.isfinite(vertices).all():
        raise ValueError('Degenerate mesh')
    rng = np.random.default_rng(seed)
    selected = tris[rng.choice(len(tris), size=count, p=areas/areas.sum())]
    a, b = rng.random((count, 2)).T
    root = np.sqrt(a)
    return selected[:, 0] * (1-root[:, None]) + selected[:, 1] * (root*(1-b))[:, None] + selected[:, 2] * (root*b)[:, None]


def render_mesh(vertices, triangles, path, settings):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if len(triangles) > settings['render_max_triangles']:
        raise ValueError('Render triangle budget exceeded; mesh was not silently truncated')
    limit = settings['render_axis_limit']
    if np.max(np.abs(vertices)) > limit:
        raise ValueError('Shape exceeds fixed render bounds; increase render_axis_limit for the whole benchmark')
    figure = plt.figure(figsize=(5, 5), dpi=120)
    try:
        axes = figure.add_subplot(111, projection='3d')
        axes.set_proj_type('ortho')
        mesh = Poly3DCollection(vertices[triangles], facecolor='#a4b9cc', edgecolor='#51677a', linewidth=0.1)
        axes.add_collection3d(mesh)
        axes.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
        axes.set_box_aspect((1, 1, 1))
        axes.view_init(elev=25, azim=45)
        axes.set_axis_off()
        figure.savefig(path, bbox_inches='tight', pad_inches=0.02, facecolor='white')
    finally:
        plt.close(figure)


def execute_request(request, output):
    import numpy as np
    import cadquery as cq
    settings = request['settings']
    destination = Path(request['directory'])
    destination.mkdir(parents=True, exist_ok=True)
    shape = build_solid(request['sequence'], settings)
    volume = float(shape.Volume())
    if not math.isfinite(volume) or volume <= settings['minimum_volume']:
        raise ValueError('Nonpositive or non-finite solid volume')
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    mesher = BRepMesh_IncrementalMesh(shape.wrapped, settings['mesh_tolerance'], False, settings['mesh_angular_tolerance'], True)
    mesher.Perform()
    if not mesher.IsDone():
        raise ValueError('Absolute-deflection meshing failed')
    # Read the existing absolute-deflection mesh directly. CadQuery's tessellate
    # helper may request relative meshing, which is a different metric protocol.
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopAbs import TopAbs_Orientation
    vertices, triangles = [], []
    for face in shape.Faces():
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face.wrapped, location)
        if triangulation is None:
            raise ValueError('A solid face has no tessellation')
        offset = len(vertices)
        for index in range(1, triangulation.NbNodes() + 1):
            point = triangulation.Node(index).Transformed(location.Transformation())
            vertices.append((point.X(), point.Y(), point.Z()))
        for triangle in triangulation.Triangles():
            a, b, c = (triangle.Value(i) + offset - 1 for i in (1, 2, 3))
            triangles.append((a, c, b) if face.wrapped.Orientation() == TopAbs_Orientation.TopAbs_REVERSED else (a, b, c))
    vertices = np.array(vertices)
    triangles = np.array(triangles, dtype=int)
    points = sample_mesh(vertices, triangles, settings['point_count'], request['sample_seed'])
    np.save(destination / 'points.npy', points)
    cq.exporters.export(shape, str(destination / 'model.step'), exportType='STEP')
    cq.exporters.export(shape, str(destination / 'model.stl'), exportType='STL',
                        tolerance=settings['mesh_tolerance'], angularTolerance=settings['mesh_angular_tolerance'])
    image, render_error = None, None
    if settings['render']:
        try:
            render_mesh(vertices, triangles, destination / 'render.png', settings)
            image = str(destination / 'render.png')
        except ValueError as e:
            render_error = str(e)
    report = {'status': 'valid', 'volume': volume, 'solid_count': len(shape.Solids()),
              'points': str(destination / 'points.npy'), 'step': str(destination / 'model.step'),
              'stl': str(destination / 'model.stl'), 'image': image, 'render_error': render_error,
              'triangles': len(triangles), 'sample_seed': request['sample_seed'],
              'kernel_backend': 'cadquery/OCP', 'request_sha256': request['request_sha256'],
              'versions': {name: importlib.metadata.version(name) for name in ('cadquery', 'numpy')}}
    write_json(output, report)


def main():
    require_container()
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    try:
        execute_request(request, args.result)
    except (ImportError, AttributeError, TypeError) as e:
        write_json(args.result, {'status': 'infrastructure_error', 'error': str(e),
                                'traceback': traceback.format_exc(), 'request_sha256': request['request_sha256']})
        raise SystemExit(2)
    except Exception as e:
        write_json(args.result, {'status': 'invalid', 'error': str(e), 'error_type': type(e).__name__,
                                'request_sha256': request['request_sha256']})


if __name__ == '__main__':
    main()
