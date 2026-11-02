"""Best-of-k variant of the frozen generation benchmark (not the greedy benchmark itself).

For each generation prompt, sample k candidates, keep the first one the CAD kernel accepts, and
score the selected outputs with the unchanged benchmark scorer. Editing rows are copied from the
greedy run, so only generation changes. Reported separately from the frozen benchmark.
"""
import json
import shutil
from pathlib import Path

from .bench_data import content_hash, file_hash, load_manifest, read_predictions
from .common import require_container, seed_all, write_json


def select_first_valid(candidates, is_valid):
    """Index of the first candidate that decoded and passed the kernel, and how many were checked.

    Returns (None, len(candidates)) when no candidate is valid; callers keep candidate 0 then, so an
    invalid selection still counts as a failure in the scorer.
    """
    for checked, candidate in enumerate(candidates, 1):
        if candidate.get('status') == 'syntax_valid' and is_valid(candidate['generated_sequence']):
            return checked - 1, checked
    return None, len(candidates)


def copied_row(row, signature_hash, manifest_hash):
    """Editing rows are greedy outputs of the same adapter; keep them, relabelled to this run."""
    return {**row, 'manifest_sha256': manifest_hash, 'run_signature_sha256': signature_hash,
            'copied_from_run_signature_sha256': row.get('run_signature_sha256')}


def run_best_of_k(config, settings, k=8, temperature=1.0):
    require_container()
    from .bench_runtime import KernelRunner
    from .inference import CadGenerator
    base = Path(settings['base_directory'])
    manifest_path = base / 'manifest.json'
    manifest = load_manifest(manifest_path)
    greedy = read_predictions(base / 'predictions.jsonl', manifest)
    root = Path(settings['output'])
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_path, root / 'manifest.json')
    base_metadata = json.loads((base / 'predictions.meta.json').read_text())
    signature = {'manifest_sha256': file_hash(manifest_path), 'model_files': base_metadata['signature']['model_files'],
                 'model_config': config, 'max_new_tokens': settings['max_new_tokens'], 'seed': settings['seed'],
                 'decoding': f'best_of_{k}_sampled_t{temperature}_first_kernel_valid_generation_editing_copied_greedy',
                 'base_predictions_sha256': file_hash(base / 'predictions.jsonl')}
    signature_hash = content_hash(signature)
    write_json(root / 'predictions.meta.json', {'schema': 'cad-predictions-v1', 'signature': signature,
                                                'signature_sha256': signature_hash})
    journal = root / 'selected-generation.jsonl'
    if not journal.exists():
        selected = generate_and_select(config, settings, manifest, k, temperature, CadGenerator, KernelRunner)
        with journal.open('w') as handle:
            for row in selected:
                handle.write(json.dumps(row) + '\n')
    selected = {row['sample_id']: row for row in map(json.loads, journal.read_text().splitlines())}
    rows = []
    for sample in manifest['samples']:
        if sample['task'] == 'generation':
            row = dict(selected[sample['id']])
            rows.append({**row, 'manifest_sha256': signature['manifest_sha256'], 'run_signature_sha256': signature_hash})
        else:
            rows.append(copied_row(greedy[sample['id']], signature_hash, signature['manifest_sha256']))
    with (root / 'predictions.jsonl').open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    return root / 'predictions.jsonl'


def generate_and_select(config, settings, manifest, k, temperature, CadGenerator, KernelRunner):
    seed_all(settings['seed'])
    generator = CadGenerator(config)
    runner = KernelRunner(settings)
    selected = []
    for sample in manifest['samples']:
        if sample['task'] != 'generation':
            continue
        candidates = generator.predict_candidates(sample['instruction'], sample.get('source_sequence'),
                                                  settings['max_new_tokens'], k, temperature)
        index, checked = select_first_valid(
            candidates, lambda sequence: runner.reconstruct(sequence, sample['id'], 'prediction')['status'] == 'valid')
        chosen = candidates[index] if index is not None else candidates[0]
        selected.append({'sample_id': sample['id'], 'status': chosen['status'],
                         'generated_sequence': chosen.get('generated_sequence'),
                         'generated_native_ids': chosen.get('generated_native_ids'),
                         'selection': {'k': k, 'checked': checked, 'selected_index': index}})
    return selected
