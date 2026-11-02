from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from .bench_data import canonical_sequence, code_hash, content_hash
from .common import require_container, write_json

EVAL_PACKAGES = ('cadquery', 'scipy', 'numpy', 'matplotlib', 'Pillow')


def setup_evaluation():
    require_container()
    directory = Path('.runtime/eval-venv')
    marker = directory / 'image-id'
    image_id = os.environ.get('CADTOKENIZER_IMAGE_ID')
    if not image_id:
        raise RuntimeError('Missing pinned Unsloth image ID; use scripts/cad.sh')
    if directory.exists() and marker.exists() and marker.read_text().strip() != image_id:
        raise RuntimeError('Existing evaluation environment belongs to another image. Move .runtime/eval-venv aside before setup.')
    if not directory.exists():
        subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(directory)], check=True)
    python = str(directory / 'bin/python')
    subprocess.run([python, '-m', 'pip', 'install', '-r', 'requirements-eval.txt'], check=True)
    # A venv created by a Python already in a venv does not inherit that parent
    # venv's packages via --system-site-packages. Preserve the container's ML
    # environment without reinstalling torch or creating a host environment.
    site_directory = subprocess.check_output(
        [python, '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip()
    parent_sites = [p for p in sys.path if p.endswith(('site-packages', 'dist-packages')) and Path(p).is_dir()]
    (Path(site_directory) / 'container-runtime.pth').write_text('\n'.join(parent_sites) + '\n')
    marker.write_text(image_id + '\n')
    versions = subprocess.check_output([python, '-m', 'pip', 'freeze'], text=True)
    (directory / 'packages.txt').write_text(versions)
    print('Evaluation dependencies installed inside the cached Unsloth image; no image was pulled or built.', flush=True)


def evaluation_doctor():
    require_container()
    versions, missing = {}, []
    for name in EVAL_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'MISSING'
            missing.append(name)
    if missing:
        raise RuntimeError(f'Missing evaluation packages: {missing}. Run CAD_CPU=1 ./scripts/cad.sh benchmark-setup in the same cached image.')
    import cadquery as cq
    from scipy.spatial import cKDTree
    from OCP.BRepCheck import BRepCheck_Analyzer
    solid = cq.Workplane('XY').box(1, 1, 1).val()
    if not BRepCheck_Analyzer(solid.wrapped).IsValid() or abs(solid.Volume() - 1) > 1e-6:
        raise RuntimeError('OpenCascade unit-box self-check failed')
    report = {'versions': versions, 'image_id': os.environ.get('CADTOKENIZER_IMAGE_ID'),
              'kernel_self_check': 'passed', 'python': sys.executable}
    write_json('.runtime/evaluation_environment.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report


class KernelRunner:
    def __init__(self, settings):
        require_container()
        if settings['kernel_backend'] != 'cadquery':
            raise ValueError('Implemented kernel_backend is cadquery')
        if settings['point_count'] < 1 or settings['kernel_timeout_seconds'] <= 0:
            raise ValueError('Invalid kernel sampling/timeout setting')
        self.settings = settings
        self.root = Path(settings['output']) / 'geometry'
        self.version = {'code_sha256': code_hash(), 'image_id': os.environ.get('CADTOKENIZER_IMAGE_ID'),
                        'packages': {name: importlib.metadata.version(name) for name in EVAL_PACKAGES}}

    def reconstruct(self, sequence, sample_id, role):
        if sequence is None:
            return {'status': 'missing_sequence', 'error': 'No CAD sequence was produced'}
        try:
            sequence = canonical_sequence(sequence)
        except ValueError as e:
            return {'status': 'syntax_invalid', 'error': str(e)}
        settings = {k: self.settings[k] for k in (
            'kernel_backend', 'point_count', 'mesh_tolerance', 'mesh_angular_tolerance',
            'boolean_tolerance', 'minimum_volume', 'render', 'render_axis_limit', 'render_max_triangles')}
        seed = int(content_hash([self.settings['seed'], sample_id, role])[:8], 16)
        fingerprint = content_hash({'sequence': sequence, 'settings': settings, 'sample_seed': seed,
                                    'timeout': self.settings['kernel_timeout_seconds'], 'runtime': self.version})
        directory = self.root / fingerprint
        result_path = directory / 'result.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get('request_sha256') == fingerprint:
                if result.get('status') == 'infrastructure_error':
                    raise RuntimeError(f'Cached geometry infrastructure failure: {result["error"]}')
                if result.get('status') != 'valid' or all(Path(result[key]).is_file() for key in ('points', 'step', 'stl')):
                    return result
        directory.mkdir(parents=True, exist_ok=True)
        request = {'request_sha256': fingerprint, 'sequence': sequence, 'settings': settings,
                   'directory': str(directory), 'sample_seed': seed}
        request_path = directory / 'request.json'
        write_json(request_path, request)
        result_path.unlink(missing_ok=True)
        environment = os.environ.copy()
        environment['OMP_NUM_THREADS'] = '1'
        try:
            completed = subprocess.run([sys.executable, '-m', 'cadtokenizer.geometry_worker',
                                        '--request', str(request_path), '--result', str(result_path)],
                                       timeout=self.settings['kernel_timeout_seconds'], capture_output=True, text=True, env=environment)
        except subprocess.TimeoutExpired:
            result = {'status': 'timeout', 'error': 'OpenCascade/meshing worker exceeded its time budget', 'request_sha256': fingerprint}
            write_json(result_path, result)
            return result
        (directory / 'worker.log').write_text(completed.stdout + completed.stderr)
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get('request_sha256') != fingerprint:
                raise RuntimeError('Geometry worker returned an incompatible cache result')
            if result['status'] == 'infrastructure_error':
                raise RuntimeError(f'Geometry infrastructure failure: {result["error"]}; see {directory / "worker.log"}')
            return result
        if completed.returncode < 0:
            result = {'status': 'kernel_crash', 'error': f'Kernel process terminated with signal {-completed.returncode}', 'request_sha256': fingerprint}
            write_json(result_path, result)
            return result
        raise RuntimeError(f'Geometry worker failed without a result; see {directory / "worker.log"}')
