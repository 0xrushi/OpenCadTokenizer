"""Free-running recall of 64 preselected training examples; no optimization."""
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import statistics
import time

from .common import file_hash, read_config, require_container, seed_all, write_json


def run_train_recall():
    require_container()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from .bench_data import code_hash
    from .lossless_sem import LosslessSemCodec
    from .lossless_sft import build_model, load_samples, generate_one
    from .lossless_editing_eval import read_journal
    from scripts.analyze_edit_correctness import align, units, command_types

    root = Path('artifacts/lossless-train-recall-v1')
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock, sdpa_kernel(SDPBackend.MATH):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(root / 'state.json', {'status': 'running', 'phase': 'audit'})
            plan_path = Path('artifacts/lossless-edit-supervision-v1/next_diagnostic_plan.json')
            plan = read_config(plan_path)
            settings = read_config('configs/lossless-sft-pilot.json')
            pilot_path = Path(settings['output']) / 'manifest.json'
            pilot = read_config(pilot_path)
            selected = {r['example_id']: r for r in pilot['train']}
            ids = plan['sample_ids']
            if len(ids) != 64 or len(set(ids)) != 64 or not set(ids).issubset(selected):
                raise ValueError('Recall panel must contain exactly the 64 registered training IDs')
            original = pilot['signature']
            codec_path = Path(settings['codec_root']) / 'codec.json'
            cache = Path(settings['codec_root']) / 'tokens.sqlite'
            database = Path('data/prepared/cad.sqlite')
            if file_hash(database) != original['source_database_sha256'] or file_hash(cache) != original['cache_sha256'] or file_hash(codec_path) != original['codec_sha256']:
                raise ValueError('Training data/cache/codec changed')
            for name, expected in original['model_sha256'].items():
                if file_hash(Path(settings['model_path']) / name) != expected:
                    raise ValueError('Base model changed')
            for name in ('lossless_model.py', 'lossless_sem.py', 'lossless_sft.py'):
                path = 'cadtokenizer/' + name
                if file_hash(path) != original['code_sha256'][path]:
                    raise ValueError('Model or decoder implementation changed')
            checkpoint = Path(plan['checkpoint'])
            if read_config(checkpoint / 'complete.json')['step'] != 500:
                raise ValueError('Expected completed step-500 checkpoint')
            cases = []
            with sqlite3.connect(database) as db:
                for index, key in enumerate(ids):
                    instruction, source, target, split, task = db.execute(
                        'SELECT instruction,source_id,target_id,split,task FROM examples WHERE id=?', (key,)).fetchone()
                    if split != 'train' or task != 'editing':
                        raise ValueError('Expected training editing examples')
                    if not instruction.lower().startswith('add ' if index < 32 else 'remove '):
                        raise ValueError('Registered group mismatch')
                    seqs = []
                    for endpoint in (source, target):
                        seq, endpoint_split = db.execute('SELECT sequence,split FROM geometry WHERE id=?', (endpoint,)).fetchone()
                        if endpoint_split != 'train':
                            raise ValueError('Endpoint is not training data')
                        seqs.append(seq)
                    cases.append({**selected[key], 'instruction': instruction, 'source': seqs[0], 'target': seqs[1],
                                  'group': 'add_numeric' if index < 16 else 'add_no_numeric' if index < 32 else 'remove'})
            signature = {'plan_sha256': file_hash(plan_path), 'pilot_manifest_sha256': file_hash(pilot_path),
                         'adapter': {p.name: file_hash(p) for p in sorted((checkpoint / 'adapter').iterdir()) if p.is_file()},
                         'code_sha256': code_hash(), 'settings': settings,
                         'image_id': os.environ.get('CADTOKENIZER_IMAGE_ID')}
            manifest = {'signature': signature, 'cases': cases}
            path = root / 'manifest.json'
            if path.exists() and read_config(path) != manifest:
                raise ValueError('Recall provenance changed; use a fresh output')
            write_json(path, manifest)
            codec = LosslessSemCodec.from_dict(read_config(codec_path))
            samples = load_samples(settings, cases)
            journal = root / 'predictions.jsonl'
            rows = read_journal(journal, set(ids))
            torch.set_num_threads(2)
            torch.set_float32_matmul_precision('high')
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            seed_all(42)
            if len(rows) < 64:
                write_json(root / 'state.json', {'status': 'running', 'phase': 'load_model'})
                model, _, _ = build_model(settings, codec, checkpoint)
                model.eval()
                with journal.open('a') as stream:
                    for case, sample in zip(cases, samples):
                        key = case['example_id']
                        if key in rows:
                            continue
                        row = {'id': key, 'sequence': None, 'error': None}
                        started = time.monotonic()
                        try:
                            # Target tokens are sliced off before inference.
                            row['sequence'], row['generated_ids'] = generate_one(model,
                                sample['input_ids'][:sample['prompt_length']], codec, settings)
                        except ValueError as exc:
                            row['error'] = str(exc)
                        row['seconds'] = time.monotonic() - started
                        stream.write(json.dumps(row) + '\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                        rows[key] = row
                        progress = {'status': 'running', 'phase': 'inference', 'completed': len(rows), 'total': 64}
                        write_json(root / 'state.json', progress)
                        print(json.dumps(progress), flush=True)
            scored = []
            for case in cases:
                row = {**rows[case['example_id']], 'group': case['group']}
                a, b, p = units(case['source']), units(case['target']), units(row['sequence'])
                source_distance, _, _ = align(a, b)
                distance, _, _ = align(b, p)
                row.update(exact_target=p == b, copies_source=p == a,
                           correct_command_types=command_types(p) == command_types(b),
                           target_command_distance=distance, source_target_command_distance=source_distance,
                           closer_than_copy=distance < source_distance)
                scored.append(row)
            summaries = {}
            for group in ('all', 'add', 'add_numeric', 'add_no_numeric', 'remove'):
                chosen = [r for r in scored if group == 'all' or r['group'] == group or group == 'add' and r['group'].startswith('add_')]
                summaries[group] = {'cases': len(chosen), 'syntax_decoded': sum(r['sequence'] is not None for r in chosen),
                    'exact_targets': sum(r['exact_target'] for r in chosen),
                    'correct_command_types_but_wrong_numbers': sum(r['correct_command_types'] and not r['exact_target'] for r in chosen),
                    'copies_source': sum(r['copies_source'] for r in chosen),
                    'closer_than_copy_by_command_distance': sum(r['closer_than_copy'] for r in chosen),
                    'mean_target_command_distance': statistics.mean(r['target_command_distance'] for r in chosen)}
            result = {'summaries': summaries, 'limits': 'Recall of 64 preselected training prompts, seen once or twice; not new training or held-out evaluation. No kernel scoring. Low recall identifies incomplete training-target reconstruction but does not establish its cause. Numeric groups use a coarse numeral heuristic; validation groups differ in size/composition.'}
            write_json(root / 'per_case.json', scored)
            write_json(root / 'results.json', result)
            (root / 'findings.md').write_text('# Training recall diagnostic\n\n' + result['limits'] + '\n\n```json\n' + json.dumps(result, indent=2) + '\n```\n')
            write_json(root / 'state.json', {'status': 'complete', 'results': str(root / 'results.json')})
        except Exception as exc:
            write_json(root / 'state.json', {'status': 'failed', 'error': repr(exc)})
            raise
