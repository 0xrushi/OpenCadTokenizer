"""One-shot tokenizer evaluation on the untouched upstream test split."""
from pathlib import Path
import hashlib
import json
import sqlite3

from .common import file_hash, read_config, require_container, write_json
from .tokenizer_experiment import features, evaluate_checkpoint
from .tokenizer_learning_curve import paired_comparison


def run_heldout(settings):
    require_container()
    root = Path(settings['output'])
    root.mkdir(parents=True, exist_ok=True)
    source_manifest = read_config(settings['source_manifest'])
    forbidden_ids = {row['id'] for row in source_manifest['train'] + source_manifest['validation']}
    forbidden_components = {row['component'] for row in source_manifest['train'] + source_manifest['validation']}
    database = settings['database']
    with sqlite3.connect(database) as db:
        all_rows = db.execute('SELECT id,sequence,component FROM geometry WHERE split=? AND tokenizer_candidate=1',
                              ('test',)).fetchall()
    if not all_rows:
        raise ValueError('No eligible upstream test geometries found')
    if any(key in forbidden_ids or component in forbidden_components for key, _, component in all_rows):
        raise ValueError('Upstream test split overlaps frozen training/validation IDs or components')
    rows = sorted(all_rows, key=lambda row: (hashlib.sha256(
        f"{settings['seed']}:{row[0]}".encode()).hexdigest(), row[0]))[:settings['case_count']]
    cases = [{'id': key, 'sequence': sequence, 'component': component, 'features': features(sequence)}
             for key, sequence, component in rows]
    manifest = {'source_split': 'upstream test', 'database_sha256': file_hash(database),
                'source_training_manifest': settings['source_manifest'], 'seed': settings['seed'],
                'case_count_requested': settings['case_count'], 'case_count': len(cases),
                'case_ids': [row['id'] for row in cases], 'cases': cases,
                'candidate_checkpoint': settings['candidate_checkpoint'],
                'baseline_checkpoint': settings['baseline_checkpoint']}
    manifest_path = root / 'manifest.json'
    if manifest_path.exists() and read_config(manifest_path) != manifest:
        raise ValueError('Held-out sample provenance changed; use a fresh output directory')
    write_json(manifest_path, manifest)
    eval_settings = {key: settings[key] for key in ('seed', 'point_count', 'kernel_workers')}
    base = evaluate_checkpoint(settings['baseline_checkpoint'], cases, eval_settings, root / 'baseline')
    candidate = evaluate_checkpoint(settings['candidate_checkpoint'], cases, eval_settings, root / 'candidate')
    comparison = paired_comparison(read_config(root / 'baseline/per_case.json'),
                                   read_config(root / 'candidate/per_case.json'))
    result = {'manifest': str(manifest_path), 'source_split': 'upstream test',
              'baseline': base['summary'], 'candidate': candidate['summary'],
              'paired_comparison': comparison,
              'interpretation': 'Independent test estimate; do not use to select or tune checkpoints.'}
    write_json(root / 'results.json', result)
    lines = ['# Untouched upstream test evaluation', '',
             f"Fixed hash-seeded sample of {len(cases)} geometries from the upstream test split.", '',
             '| Model | Valid / known-valid | Mean normalized squared CD | Median relative volume error |',
             '| --- | ---: | ---: | ---: |']
    for name, report in [('Existing tokenizer', base), ('Selected 40k tokenizer', candidate)]:
        summary = report['summary']
        lines.append(f"| {name} | {summary['valid_reconstructions_of_valid_references']}/{summary['known_valid_references']} | {summary['mean_normalized_squared_chamfer']} | {summary['median_relative_volume_error']} |")
    lines += ['', f"Paired valid cases: {comparison['common_valid_cases']}; mean CD existing/candidate: {comparison['baseline_mean_chamfer_common']} / {comparison['candidate_mean_chamfer_common']}.",
              '', 'The sample was fixed before inference. This is a tokenizer reconstruction test, not the paper’s full end-to-end benchmark.']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result), flush=True)
