"""Controlled low-rate continuation, preserving the saved optimizer and data cursor."""
from pathlib import Path
import copy
import fcntl
import json
import os
import shutil

from .common import file_hash, load_checkpoint, read_config, require_container, save_checkpoint, write_json


def lower_optimizer_rate(checkpoint, learning_rate):
    old = [group['lr'] for group in checkpoint['optimizer']['param_groups']]
    if not old or any(abs(rate / learning_rate - 10) > 1e-8 for rate in old):
        raise ValueError('Expected exactly a tenfold learning-rate reduction')
    for group in checkpoint['optimizer']['param_groups']:
        group['lr'] = learning_rate
        if 'initial_lr' in group:
            group['initial_lr'] = learning_rate
    return old


def run_continuation(settings):
    require_container()
    import torch
    from .tokenizer_experiment import evaluate_checkpoint
    from .tokenizer_learning_curve import paired_comparison, train_variant
    from .vq import LOSS_NORMALIZATION
    root = Path(settings['output'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            torch.set_num_threads(2)
            torch.set_float32_matmul_precision('high')
            if not torch.cuda.is_available():
                raise RuntimeError('Local CUDA is required')
            source = Path(settings['checkpoint'])
            original = read_config(settings['manifest'])
            spec = copy.deepcopy(original['variants']['curve'])
            if file_hash(spec['database']) != spec['database_sha256']:
                raise ValueError('Source training database changed')
            cfg = read_config(spec['config_path'])
            cfg['vq']['learning_rate'] = settings['learning_rate']
            cfg['vq']['output'] = str(root / 'curve')
            protocol = read_config(settings['training_settings'])
            protocol.update({k: settings[k] for k in ('output', 'evaluation_updates', 'max_updates')})
            signature = {'schema': 'tokenizer-low-lr-v1', 'settings': settings,
                         'source_sha256': file_hash(source), 'original_signature': original['signature'],
                         'code': {str(p): file_hash(p) for p in sorted(Path('cadtokenizer').glob('*.py'))},
                         'image_id': os.environ.get('CADTOKENIZER_IMAGE_ID'),
                         'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name()}
            manifest = {'signature': signature, 'validation': original['validation'],
                        'variants': {'curve': {**spec, 'config_path': str(root / 'curve/config.json')}}}
            path = root / 'manifest.json'
            if path.exists() and read_config(path) != manifest:
                raise ValueError('Continuation provenance changed; use a fresh output directory')
            write_json(path, manifest)
            write_json(root / 'curve/config.json', cfg)
            write_json(root / 'settings.json', protocol)
            archive = root / 'source'
            archive.mkdir(exist_ok=True)
            for file in Path('cadtokenizer').glob('*.py'):
                shutil.copy2(file, archive / file.name)
            shutil.copy2('scripts/cad.sh', archive / 'cad.sh')
            last = root / 'curve/last.pt'
            if not last.exists():
                checkpoint = load_checkpoint(source)
                if (checkpoint['step'] != 35000 or checkpoint['loss_normalization'] != LOSS_NORMALIZATION
                        or checkpoint['database_sha256'] != spec['database_sha256']
                        or checkpoint['experiment_signature'] != original['signature']):
                    raise ValueError('Unexpected warm-start checkpoint')
                expected = copy.deepcopy(checkpoint['config'])
                expected['vq'].update(learning_rate=settings['learning_rate'], output=str(root / 'curve'))
                if expected != cfg:
                    raise ValueError('Continuation would change more than output and learning rate')
                old_rates = lower_optimizer_rate(checkpoint, settings['learning_rate'])
                checkpoint.update(config=cfg, experiment_signature=signature)
                save_checkpoint(last, checkpoint)
                write_json(root / 'warm_start.json', {'checkpoint': str(source),
                           'sha256': signature['source_sha256'], 'step': checkpoint['step'],
                           'epoch': checkpoint['epoch'], 'offset': checkpoint['offset'],
                           'old_learning_rates': old_rates, 'learning_rate': settings['learning_rate'],
                           'optimizer_moments_rng_and_data_cursor_preserved': True})
                del checkpoint
            write_json(root / 'state.json', {'status': 'running', 'phase': 'baseline-35000'})
            evaluate_checkpoint(source, manifest['validation'], protocol, root / 'baseline')
            result = train_variant(manifest, protocol, 'curve', root)
            # Selection is on validation geometry, never training loss. Preserve
            # the parent unless a continuation improves validity and paired CD.
            qualified = [p for p in result['points'] if p['paired']['improves_validity_and_geometry']]
            selected = max(qualified, key=lambda p: p['paired']['candidate_valid']) if qualified else None
            old_final = read_config(settings['original_final_rows'])
            controls = {str(p['step']): paired_comparison(old_final, read_config(
                root / 'curve' / f"evaluation-{p['step']}" / 'per_case.json')) for p in result['points']}
            selection = {'selected_checkpoint': selected['checkpoint'] if selected else str(source),
                         'qualified_steps': [p['step'] for p in qualified],
                         'criterion': 'More valid solids and no worse paired mean normalized CD than 35000 parent',
                         'production_promoted': False, 'original_50000_control_comparisons': controls}
            write_json(root / 'selection.json', selection)
            write_json(root / 'results.json', result)
            write_json(root / 'state.json', {'status': 'complete', **selection})
            print(json.dumps(selection), flush=True)
        except Exception as exc:
            write_json(root / 'state.json', {'status': 'failed', 'error': repr(exc)})
            raise
