"""Frozen 500-case validation expansion for the saved lossless editing model."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gc
import json
import os
from pathlib import Path
import sqlite3
import statistics
import time

from .common import file_hash, fraction, read_config, require_container, seed_all, write_json
from .representation import stringify, tokenize

ROOT = Path('artifacts/lossless-editing-validation-v1')
PILOT = Path('artifacts/lossless-sft-pilot-v1')


def select_cases():
    """Select without model outputs; exclude the initial panel's components."""
    pilot = read_config(PILOT / 'manifest.json')
    excluded = {c['example_id'] for c in pilot['validation']}
    with sqlite3.connect('data/prepared/cad.sqlite') as db, \
            sqlite3.connect('artifacts/lossless-sem-v2/tokens.sqlite') as cache, \
            sqlite3.connect('artifacts/vastai-download/tokens.sqlite') as old:
        old_val = {r[0] for r in old.execute("SELECT example_id FROM samples WHERE split='val' AND task='editing'")}
        cached = {r[1]: r[0] for r in cache.execute("SELECT id,example_id FROM samples WHERE split='val' AND task='editing' AND context_overflow=0")}
        geometries = {r[0]: r[1:] for r in db.execute("SELECT id,sequence,component,split FROM geometry WHERE split IN ('train','val')")}
        examples = {r[0]: r[1:] for r in db.execute("SELECT id,instruction,source_id,target_id,split FROM examples WHERE split IN ('train','val')")}
        prior_components = set()
        train_components = set()
        for group, components in [(pilot['validation'], prior_components), (pilot['train'], train_components)]:
            for case in group:
                _, source, target, _ = examples[case['example_id']]
                for endpoint in (source, target):
                    if endpoint:
                        components.add(geometries[endpoint][1])
        candidates = []
        for key in sorted(set(cached) & old_val - excluded, key=lambda k: (fraction(k, 142), k)):
            instruction, source, target, split = examples[key]
            if split != 'val' or source is None:
                raise ValueError('Editing validation split/source mismatch')
            endpoints = [geometries[source], geometries[target]]
            if any(g[2] != 'val' for g in endpoints):
                raise ValueError('Validation endpoint split mismatch')
            components = {g[1] for g in endpoints}
            if components & train_components:
                raise ValueError('Pilot training component leakage')
            if components & prior_components:
                continue
            candidates.append({'example_id': key, 'cache_id': cached[key], 'task': 'editing',
                               'instruction': instruction, 'source': endpoints[0][0], 'target': endpoints[1][0],
                               'source_id': source, 'target_id': target, 'components': sorted(components)})
        if len(candidates) < 500:
            raise ValueError('Insufficient fresh validation prompts')
        return candidates[:500], {'eligible_fresh_prompts': len(candidates),
                                  'selection_seed': 142, 'prior_panel_component_overlap': 0,
                                  'pilot_training_component_overlap': 0,
                                  'unique_selected_components': len({v for c in candidates[:500] for v in c['components']})}


def read_journal(path, allowed):
    if not path.exists():
        return {}
    raw = path.read_bytes()
    # Only an incomplete final write can be removed during recovery.
    if raw and not raw.endswith(b'\n'):
        end = raw.rfind(b'\n') + 1
        tail = raw[end:]
        try:
            json.loads(tail)
        except (ValueError, UnicodeDecodeError):
            raw = raw[:end]
        else:
            raw += b'\n'
        path.write_bytes(raw)
    rows = [json.loads(line) for line in raw.splitlines()]
    result = {r['id']: r for r in rows}
    if len(rows) != len(result) or not set(result).issubset(allowed):
        raise ValueError('Duplicate or foreign journal IDs')
    return result


def run_editing_evaluation():
    require_container()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    import numpy as np
    from scipy.spatial import cKDTree
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from .bench_data import code_hash
    from .bench_predict import model_fingerprints
    from .bench_runtime import KernelRunner
    from .bench_metrics import normalize_points
    from .inference import CadGenerator
    from .lossless_compare import summarize, paired_comparison
    from .lossless_sem import LosslessSemCodec
    from .lossless_sft import build_model, generate_one, load_samples

    ROOT.mkdir(parents=True, exist_ok=True)
    def status(phase, **kwargs):
        write_json(ROOT / 'state.json', {'status': 'running', 'phase': phase, **kwargs})
    with (ROOT / 'run.lock').open('a') as lock, sdpa_kernel(SDPBackend.MATH):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            status('audit')
            cases, audit = select_cases()
            settings = read_config('configs/lossless-sft-pilot.json')
            checkpoint = PILOT / 'checkpoints/step-000500'
            if read_config(checkpoint / 'complete.json')['step'] != 500:
                raise ValueError('Expected completed step-500 checkpoint')
            config = read_config('artifacts/vastai-download/run.json')['config']
            config['sft']['output'] = 'artifacts/vastai-download'
            config['device'] = 'cuda'
            config['inference'] = {'cad_decoder': 'codebook'}
            database_hash = file_hash(config['data']['database'])
            pilot_manifest = read_config(PILOT / 'manifest.json')
            if file_hash('artifacts/lossless-sem-v2/codec.json') != pilot_manifest['signature']['codec_sha256'] or file_hash('artifacts/lossless-sem-v2/tokens.sqlite') != pilot_manifest['signature']['cache_sha256']:
                raise ValueError('Pilot codec or prompt cache changed')
            for name in ('lossless_model.py', 'lossless_sem.py', 'lossless_sft.py'):
                path = 'cadtokenizer/' + name
                if file_hash(path) != pilot_manifest['signature']['code_sha256'][path]:
                    raise ValueError('Lossless model or decoding implementation changed')
            if database_hash != pilot_manifest['signature']['source_database_sha256'] or database_hash != read_config('artifacts/vastai-download/run.json')['database_sha256']:
                raise ValueError('Training database fingerprints differ')
            model_dir = Path(settings['model_path'])
            for name, expected in pilot_manifest['signature']['model_sha256'].items():
                if file_hash(model_dir / name) != expected:
                    raise ValueError('Cached base model changed')
            signature = {
                'database_sha256': database_hash, 'pilot_manifest_sha256': file_hash(PILOT / 'manifest.json'),
                'lossless_adapter': {p.name: file_hash(p) for p in sorted((checkpoint / 'adapter').iterdir()) if p.is_file()},
                'baseline_files': model_fingerprints(config), 'baseline_config': config,
                'codec_sha256': file_hash('artifacts/lossless-sem-v2/codec.json'),
                'cache_sha256': file_hash('artifacts/lossless-sem-v2/tokens.sqlite'),
                'baseline_tokens_sha256': file_hash('artifacts/vastai-download/tokens.sqlite'),
                'code_sha256': code_hash(), 'image_id': os.environ.get('CADTOKENIZER_IMAGE_ID'),
                'settings': settings, 'audit': audit, 'max_new_tokens': 256,
                'backend': 'deterministic_math_sdpa',
                'limits': '500 fresh validation editing prompts, excluding initial-panel components. Saved models have different training exposure; baseline selected checkpoints on validation. Repeated components may remain within this panel. No test data, new training or promotion. Exact target matching and copying do not measure full instruction compliance.',
            }
            frozen = {'signature': signature, 'cases': cases}
            manifest_path = ROOT / 'manifest.json'
            if manifest_path.exists() and read_config(manifest_path) != frozen:
                raise ValueError('Evaluation provenance changed; use a fresh output')
            write_json(manifest_path, frozen)
            torch.set_num_threads(2)
            torch.set_float32_matmul_precision('high')
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            allowed = {c['example_id'] for c in cases}
            arms = {}
            for arm in ('lossless', 'codebook'):
                journal = ROOT / f'{arm}_predictions.jsonl'
                rows = read_journal(journal, allowed)
                if len(rows) < 500:
                    seed_all(42)
                    status('load_' + arm, completed=len(rows), total=500)
                    if arm == 'lossless':
                        codec = LosslessSemCodec.from_dict(read_config('artifacts/lossless-sem-v2/codec.json'))
                        model, _, _ = build_model(settings, codec, checkpoint)
                        model.eval()
                        samples = load_samples(settings, cases)
                    else:
                        model = CadGenerator(config)
                    with journal.open('a') as stream:
                        for i, case in enumerate(cases):
                            key = case['example_id']
                            if key in rows:
                                continue
                            row = {'id': key, 'task': 'editing', 'sequence': None, 'error': None}
                            start = time.monotonic()
                            try:
                                # Both APIs receive only instruction/source-derived prompts.
                                if arm == 'lossless':
                                    sample = samples[i]
                                    row['sequence'], row['generated_ids'] = generate_one(model,
                                        sample['input_ids'][:sample['prompt_length']], codec, settings)
                                else:
                                    prediction = model.predict(case['instruction'], case['source'], 256)
                                    row['sequence'] = prediction['generated_sequence']
                                    row['generated_ids'] = prediction['generated_native_ids']
                            except (ValueError, KeyError) as exc:
                                row['error'] = str(exc)
                            row['seconds'] = time.monotonic() - start
                            stream.write(json.dumps(row) + '\n')
                            stream.flush()
                            os.fsync(stream.fileno())
                            rows[key] = row
                            status(arm + '_inference', completed=len(rows), total=500)
                            print(json.dumps({'arm': arm, 'completed': len(rows), 'total': 500}), flush=True)
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                arms[arm] = [rows[c['example_id']] for c in cases]
            cfg = {**read_config('configs/benchmark.json'), 'output': str(ROOT / 'geometry-cache'),
                   'render': False, 'seed': 42, 'point_count': 2000}
            runner = KernelRunner(cfg)
            jobs = []
            for i, case in enumerate(cases):
                key = case['example_id']
                jobs += [(key, 'reference', case['target']), (key, 'prediction', case['source'])]
                jobs += [(key, 'prediction', arms[arm][i]['sequence']) for arm in arms]
            jobs = list(dict.fromkeys(jobs))
            status('geometry', completed=0, total=len(jobs))
            geometry = {}
            with ThreadPoolExecutor(max_workers=8) as pool:
                for i, (job, result) in enumerate(zip(jobs, pool.map(lambda j: runner.reconstruct(j[2], j[0], j[1]), jobs)), 1):
                    geometry[job] = result
                    status('geometry', completed=i, total=len(jobs))
            arms['source_copy'] = [{'id': c['example_id'], 'task': 'editing', 'sequence': c['source']} for c in cases]
            def chamfer(ref, pred):
                if ref['status'] != 'valid' or pred['status'] != 'valid':
                    return None
                a, b = [np.asarray(normalize_points(np.load(g['points']))) for g in (ref, pred)]
                return float(np.mean(cKDTree(b).query(a)[0]**2) + np.mean(cKDTree(a).query(b)[0]**2))
            for arm, rows in arms.items():
                for case, row in zip(cases, rows):
                    ref = geometry[(row['id'], 'reference', case['target'])]
                    pred = geometry[(row['id'], 'prediction', row['sequence'])]
                    row.update(reference_geometry=ref, prediction_geometry=pred, chamfer=chamfer(ref, pred),
                               exact_sem=row['sequence'] == stringify(tokenize(case['target'])),
                               copies_source=row['sequence'] == stringify(tokenize(case['source'])))
            result = {'summaries': {arm: summarize(rows)['editing'] for arm, rows in arms.items()},
                      'paired_lossless_vs_codebook': paired_comparison(arms['lossless'], arms['codebook'])['editing'],
                      'source_copy_comparisons': {},
                      'target_equals_source_cases': sum(stringify(tokenize(c['target'])) == stringify(tokenize(c['source'])) for c in cases),
                      'limits': signature['limits']}
            for arm in ('lossless', 'codebook'):
                paired = [(a, b) for a, b in zip(arms[arm], arms['source_copy']) if a['chamfer'] is not None and b['chamfer'] is not None]
                result['source_copy_comparisons'][arm] = {
                    'shared_valid_cases': len(paired),
                    'model_mean_chamfer': statistics.mean(a['chamfer'] for a, b in paired) if paired else None,
                    'copy_mean_chamfer': statistics.mean(b['chamfer'] for a, b in paired) if paired else None,
                    'model_lower_chamfer_cases': sum(a['chamfer'] < b['chamfer'] for a, b in paired),
                    'unchanged_output_on_changed_target': sum(r['copies_source'] and not arms['source_copy'][i]['exact_sem'] for i, r in enumerate(arms[arm])),
                }
            write_json(ROOT / 'per_case.json', arms)
            write_json(ROOT / 'results.json', result)
            (ROOT / 'findings.md').write_text('# Expanded validation editing evaluation\n\n' + signature['limits'] + '\n\n```json\n' + json.dumps(result, indent=2) + '\n```\n')
            write_json(ROOT / 'state.json', {'status': 'complete', 'results': str(ROOT / 'results.json')})
        except Exception as exc:
            write_json(ROOT / 'state.json', {'status': 'failed', 'error': repr(exc)})
            raise
