"""Full-corpus loop-grouped tokenizer run on the frozen full-data protocol."""
from __future__ import annotations

import fcntl
from pathlib import Path

from .common import file_hash, read_config, require_container, write_json


def loop_manifest(settings, root):
    source = read_config(settings['source_manifest'])
    spec = dict(source['variants']['loop'])
    if file_hash(spec['config_path']) != spec['config_sha256'] or file_hash(spec['database']) != spec['database_sha256']:
        raise ValueError('Frozen loop-grouped config or database changed')
    if 'learning_rate' in settings:
        cfg = read_config(spec['config_path'])
        cfg['vq']['learning_rate'] = settings['learning_rate']
        path = root / 'loop' / 'config.json'
        write_json(path, cfg)
        spec.update(config_path=str(path), config_sha256=file_hash(path))
    return {'signature': source['signature'], 'validation': source['validation'], 'variants': {'loop': spec}}


def selection(points):
    qualified = [point['step'] for point in points if point['paired']['improves_validity_and_geometry']]
    best = max(points, key=lambda point: point['summary']['valid_reconstructions_of_valid_references'])
    return {'best_by_validation_solid_count_step': best['step'], 'qualified_at_steps': qualified}


def run_loop_full(settings):
    require_container()
    import torch
    from .tokenizer_learning_curve import train_variant
    root = Path(settings['output'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a+') as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('A loop-grouped full-data job already holds this run lock; inspect state.json instead') from error
        try:
            torch.set_num_threads(2)
            torch.set_float32_matmul_precision('high')
            if not torch.cuda.is_available():
                raise RuntimeError('Local full-data training requires CUDA')
            (root / 'loop').mkdir(exist_ok=True)
            manifest = loop_manifest(settings, root)
            path = root / 'manifest.json'
            if path.exists():
                if read_config(path) != manifest:
                    raise ValueError('Loop-grouped provenance changed; choose a fresh output')
            else:
                write_json(path, manifest)
            protocol = read_config(settings['training_settings'])
            protocol.update(output=str(root), source_manifest=settings['source_manifest'])
            write_json(root / 'training-settings.json', protocol)
            baseline = root / 'baseline'
            if not baseline.exists():
                baseline.symlink_to(Path(settings['baseline']).resolve(), target_is_directory=True)
            write_json(root / 'state.json', {'status': 'running', 'phase': 'training'})
            result = train_variant(manifest, protocol, 'loop', root)
            write_json(root / 'results.json', {'loop': result, **selection(result['points'])})
            write_json(root / 'state.json', {'status': 'complete', 'results': str(root / 'results.json')})
        except BaseException as error:
            write_json(root / 'state.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
            raise
