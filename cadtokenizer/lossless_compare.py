"""Compare the completed lossless pilot with the preserved downstream VQ model."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import statistics

from .common import file_hash, read_config, require_container, seed_all, write_json
from .representation import stringify, tokenize


def summarize(rows):
    result = {}
    for task in ('all', 'generation', 'editing'):
        selected = [r for r in rows if task == 'all' or r['task'] == task]
        valid = [r for r in selected if r['chamfer'] is not None]
        result[task] = {
            'cases': len(selected),
            'syntax_decoded': sum(r['sequence'] is not None for r in selected),
            'known_valid_references': sum(r['reference_geometry']['status'] == 'valid' for r in selected),
            'valid_predictions': sum(r['prediction_geometry']['status'] == 'valid' for r in selected),
            'valid_of_valid_references': len(valid),
            'mean_chamfer': statistics.mean(r['chamfer'] for r in valid) if valid else None,
            'exact_sem': sum(r['exact_sem'] for r in selected),
            'copies_source': sum(r['copies_source'] for r in selected),
        }
    return result


def paired_comparison(lossless, baseline):
    if [r['id'] for r in lossless] != [r['id'] for r in baseline]:
        raise ValueError('Comparison IDs/order differ')
    result = {}
    for task in ('all', 'generation', 'editing'):
        pairs = [(a, b) for a, b in zip(lossless, baseline)
                 if (task == 'all' or a['task'] == task)
                 and a['chamfer'] is not None and b['chamfer'] is not None]
        result[task] = {
            'shared_valid_cases': len(pairs),
            'lossless_mean_chamfer': statistics.mean(a['chamfer'] for a, b in pairs) if pairs else None,
            'baseline_mean_chamfer': statistics.mean(b['chamfer'] for a, b in pairs) if pairs else None,
            'lossless_lower_chamfer_cases': sum(a['chamfer'] < b['chamfer'] for a, b in pairs),
        }
    return result


def run_comparison():
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
    from .inference import CadGenerator, decode_native_ids

    root = Path('artifacts/lossless-sft-comparison-v1')
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock, sdpa_kernel(SDPBackend.MATH):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(root / 'state.json', {'status': 'running', 'phase': 'audit'})
            pilot = Path('artifacts/lossless-sft-pilot-v1')
            manifest = read_config(pilot / 'manifest.json')
            cases = manifest['validation']
            if len(cases) != 60 or len({c['example_id'] for c in cases}) != 60:
                raise ValueError('Expected the frozen 60-case pilot panel')
            config = read_config('artifacts/vastai-download/run.json')['config']
            config['sft']['output'] = 'artifacts/vastai-download'
            config['device'] = 'cuda'
            config['inference'] = {'cad_decoder': 'alignment'}
            run = read_config('artifacts/vastai-download/run.json')
            if file_hash(config['data']['database']) != run['database_sha256']:
                raise ValueError('Baseline training data fingerprint mismatch')
            # Confirm each exact instruction/source/target and every endpoint is held out.
            with sqlite3.connect(config['data']['database']) as db, sqlite3.connect('artifacts/vastai-download/tokens.sqlite') as tokens:
                for case in cases:
                    task, instruction, source, target, split = db.execute(
                        'SELECT task,instruction,source_id,target_id,split FROM examples WHERE id=?',
                        (case['example_id'],)).fetchone()
                    if split != 'val' or task != case['task'] or instruction != case['instruction']:
                        raise ValueError('Validation prompt mismatch')
                    stored = tokens.execute('SELECT split FROM samples WHERE example_id=?', (case['example_id'],)).fetchone()
                    if stored is None or stored[0] != 'val':
                        raise ValueError('Panel was not held out by baseline SFT')
                    for endpoint, expected in [(target, case['target'])] + ([(source, case['source'])] if source else []):
                        seq, endpoint_split = db.execute('SELECT sequence,split FROM geometry WHERE id=?', (endpoint,)).fetchone()
                        if endpoint_split != 'val' or seq != expected:
                            raise ValueError('Validation geometry mismatch')
            predictions_path = pilot / 'generation/step-000500/predictions.jsonl'
            signature = {
                'pilot_manifest_sha256': file_hash(pilot / 'manifest.json'),
                'lossless_predictions_sha256': file_hash(predictions_path),
                'baseline_files': model_fingerprints(config),
                'baseline_tokens_sha256': file_hash('artifacts/vastai-download/tokens.sqlite'),
                'code_sha256': code_hash(), 'image_id': os.environ.get('CADTOKENIZER_IMAGE_ID'),
                'max_new_tokens': 256, 'seed': 42, 'backend': 'deterministic_math_sdpa',
                'decoders': ['alignment', 'codebook'],
                'limits': 'Same 60 validation prompts, independently trained models with different training exposure. This compares saved systems, not a matched tokenizer ablation. No test data or promotion.',
            }
            path = root / 'manifest.json'
            if path.exists() and read_config(path) != {'signature': signature, 'cases': cases}:
                raise ValueError('Comparison provenance changed; use a fresh output')
            write_json(path, {'signature': signature, 'cases': cases})
            torch.set_num_threads(2)
            torch.set_float32_matmul_precision('high')
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            seed_all(42)
            journal = root / 'baseline_predictions.jsonl'
            existing = {r['id']: r for r in map(json.loads, journal.read_text().splitlines())} if journal.exists() else {}
            if not set(existing).issubset({c['example_id'] for c in cases}):
                raise ValueError('Unexpected baseline journal IDs')
            if len(existing) < len(cases):
                write_json(root / 'state.json', {'status': 'running', 'phase': 'load_baseline'})
                generator = CadGenerator(config)
                with journal.open('a') as stream:
                    for i, case in enumerate(cases, 1):
                        if case['example_id'] in existing:
                            continue
                        # Ground truth never enters the prediction API.
                        row = {'id': case['example_id'], 'task': case['task'],
                               'alignment': None, 'codebook': None, 'errors': {}}
                        try:
                            result = generator.predict(case['instruction'], case['source'], 256)
                            row['alignment'] = result['generated_sequence']
                            row['generated_ids'] = result['generated_native_ids']
                            try:
                                row['codebook'] = decode_native_ids(generator.vq, generator.alignment,
                                    result['generated_native_ids'], 'codebook')
                            except ValueError as exc:
                                row['errors']['codebook'] = str(exc)
                        except (ValueError, KeyError) as exc:
                            row['errors']['prediction'] = str(exc)
                        stream.write(json.dumps(row) + '\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                        existing[row['id']] = row
                        write_json(root / 'state.json', {'status': 'running', 'phase': 'baseline_inference', 'completed': i, 'total': 60})
                        print(json.dumps({'completed': i, 'total': 60}), flush=True)
                del generator
                torch.cuda.empty_cache()
            lossless = {r['id']: r for r in map(json.loads, predictions_path.read_text().splitlines())}
            if set(lossless) != {c['example_id'] for c in cases}:
                raise ValueError('Lossless prediction IDs mismatch')
            settings = {**read_config('configs/benchmark.json'), 'output': str(root / 'geometry-cache'),
                        'render': False, 'seed': 42, 'point_count': 2000}
            runner = KernelRunner(settings)
            write_json(root / 'state.json', {'status': 'running', 'phase': 'geometry'})
            jobs = [(c['example_id'], 'reference', c['target']) for c in cases]
            for case in cases:
                key = case['example_id']
                # Same sample ID and role seeds for all three prediction arms.
                jobs += [(key, 'prediction', seq) for seq in
                         [lossless[key]['sequence'], existing[key]['alignment'], existing[key]['codebook']]]
            jobs = list(dict.fromkeys(jobs))
            geometry = {}
            with ThreadPoolExecutor(max_workers=8) as pool:
                for i, (job, result) in enumerate(zip(jobs, pool.map(lambda j: runner.reconstruct(j[2], j[0], j[1]), jobs)), 1):
                    geometry[job] = result
                    write_json(root / 'state.json', {'status': 'running', 'phase': 'geometry', 'completed': i, 'total': len(jobs)})
            arms = {name: [] for name in ('lossless', 'alignment', 'codebook')}
            for case in cases:
                key = case['example_id']
                ref = geometry[(key, 'reference', case['target'])]
                for name in arms:
                    seq = lossless[key]['sequence'] if name == 'lossless' else existing[key][name]
                    pred = geometry[(key, 'prediction', seq)]
                    row = {'id': key, 'task': case['task'], 'sequence': seq,
                           'reference_geometry': ref, 'prediction_geometry': pred, 'chamfer': None,
                           'exact_sem': seq == stringify(tokenize(case['target'])),
                           'copies_source': case['source'] is not None and seq == stringify(tokenize(case['source']))}
                    if ref['status'] == pred['status'] == 'valid':
                        from .bench_metrics import normalize_points
                        a, b = [np.asarray(normalize_points(np.load(g['points']))) for g in (ref, pred)]
                        row['chamfer'] = float(np.mean(cKDTree(b).query(a)[0]**2) + np.mean(cKDTree(a).query(b)[0]**2))
                    arms[name].append(row)
            result = {'summaries': {name: summarize(rows) for name, rows in arms.items()},
                      'paired': {name: paired_comparison(arms['lossless'], arms[name]) for name in ('alignment', 'codebook')},
                      'limits': signature['limits']}
            write_json(root / 'per_case.json', arms)
            write_json(root / 'results.json', result)
            (root / 'findings.md').write_text('# Same-prompt downstream comparison\n\n' + signature['limits'] + '\n\n```json\n' + json.dumps(result, indent=2) + '\n```\n')
            write_json(root / 'state.json', {'status': 'complete', 'results': str(root / 'results.json')})
        except Exception as exc:
            write_json(root / 'state.json', {'status': 'failed', 'error': repr(exc)})
            raise
