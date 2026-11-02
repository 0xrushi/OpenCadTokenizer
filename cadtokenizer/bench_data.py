from __future__ import annotations

import collections
from contextlib import closing
import hashlib
import json
import pickle
import sqlite3
from pathlib import Path

from .common import file_hash, fraction, write_json
from .representation import split_pairs, stringify, tokenize, skexgen_to_sem


def canonical_sequence(sequence):
    tokens = tokenize(sequence)
    split_pairs(tokens)
    return stringify(tokens)


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def code_hash():
    root = Path(__file__).parent
    return content_hash({p.name: file_hash(p) for p in sorted(root.glob('*.py'))})


def validate_manifest(manifest):
    if manifest.get('schema') != 'cad-benchmark-v1' or not manifest.get('samples'):
        raise ValueError('Invalid or empty benchmark manifest')
    ids = []
    for sample in manifest['samples']:
        if sample['task'] not in ('generation', 'editing') or not isinstance(sample['id'], str):
            raise ValueError('Invalid benchmark sample identity/task')
        if (not isinstance(sample['instruction'], str) or not sample['instruction'].strip()) and not sample.get('instruction_error'):
            raise ValueError('Missing benchmark instruction')
        if sample['task'] == 'editing' and not isinstance(sample.get('source_sequence'), str):
            raise ValueError('Editing sample requires an original sequence')
        if sample.get('target_sequence') is None and not sample.get('target_error'):
            raise ValueError('Missing target must carry an explicit target_error')
        ids.append(sample['id'])
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate benchmark sample IDs')


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    validate_manifest(manifest)
    return manifest


def freeze_manifest(config, settings, output=None):
    """Freeze upstream heldouts BEFORE tokenizer-length filtering.

    Original paper annotations are not reconstructed or claimed to be available.
    Oversized inputs remain in the manifest and count as failures if unprocessable.
    """
    target = Path(output or Path(settings['output']) / 'manifest.json')
    if target.exists():
        raise FileExistsError(f'{target} is frozen; choose another manifest/output for a new benchmark')
    root = Path(config['data']['root'])
    captions = json.loads((root / 'cadfusion/test.json').read_text())
    edits = json.loads((root / 'cad_editor/dataset/test.json').read_text())
    all_samples = {'generation': [], 'editing': []}
    for i, row in enumerate(captions):
        all_samples['generation'].append({'id': f'cadfusion:test:{i}', 'task': 'generation',
                                          'instruction': row['description'], 'serial_num': int(row['serial_num'])})
    for i, row in enumerate(edits):
        all_samples['editing'].append({'id': f'cad_editor:test:{i}', 'task': 'editing',
                                       'instruction': row['instruction'], 'source_sequence': row['original_sequence'],
                                       'target_sequence': row['edited_sequence']})
    selected = []
    exclude_overlaps = settings.get('exclude_training_overlaps', False)
    for task, records in all_samples.items():
        requested = settings['test_counts'].get(task)
        if requested is not None and (not isinstance(requested, int) or requested < 1):
            raise ValueError('test_counts must be positive integers or null for all')
        if requested is not None and requested > len(records):
            raise ValueError(f'Requested {requested} {task} samples, only {len(records)} available')
        ordered = sorted(records, key=lambda x: (fraction(x['id'], settings['seed']), x['id']))
        selected.extend(ordered if requested is None or exclude_overlaps else ordered[:requested])
    with (root / 'skexgen/cad_data/train_deduplicate_s.pkl').open('rb') as f:
        skexgen = pickle.load(f)
    unknown_by_id, errors = collections.Counter(), []
    for sample in selected:
        if not isinstance(sample['instruction'], str) or not sample['instruction'].strip():
            sample['instruction_error'] = 'Missing or empty upstream instruction'
            errors.append({'sample_id': sample['id'], 'field': 'instruction', 'error': sample['instruction_error']})
        if sample['task'] == 'generation':
            sample['source_sequence'] = None
            try:
                item = skexgen[sample['serial_num']]
                sample['upstream_geometry_name'] = str(item['name'])
                sample['target_sequence'] = skexgen_to_sem(item)
            except (KeyError, IndexError, ValueError) as e:
                sample['target_sequence'], sample['target_error'] = None, str(e)
                errors.append({'sample_id': sample['id'], 'field': 'target_sequence', 'error': str(e)})
        for field in ('source_sequence', 'target_sequence'):
            if sample.get(field) is None:
                continue
            try:
                sample[field] = canonical_sequence(sample[field])
            except ValueError as e:
                # Preserve bad upstream records in the denominator and report them.
                sample[field.replace('_sequence', '_error')] = str(e)
                errors.append({'sample_id': sample['id'], 'field': field, 'error': str(e)})
    del skexgen
    database = config['data']['database']
    if not Path(database).is_file():
        raise FileNotFoundError('Prepare training data first so heldout/train overlaps can be audited')
    conflicts = []
    with closing(sqlite3.connect(database)) as db:
        for sample in selected:
            for field in ('source_sequence', 'target_sequence'):
                sequence = sample.get(field)
                if sequence is None or sample.get(field.replace('_sequence', '_error')):
                    unknown_by_id[sample['id']] += 1 if sequence is not None or field == 'target_sequence' else 0
                    continue
                key = hashlib.sha256(sequence.encode()).hexdigest()
                row = db.execute('SELECT split,tokenizer_split FROM geometry WHERE id=?', (key,)).fetchone()
                if row and (row[0] == 'train' or row[1] == 'train'):
                    conflicts.append({'sample_id': sample['id'], 'field': field, 'geometry_id': key})
    if conflicts and not exclude_overlaps:
        raise ValueError(f'Test/train geometry overlap found ({len(conflicts)} endpoints); first: {conflicts[0]}')
    blocked = {item['sample_id'] for item in conflicts}
    if exclude_overlaps:
        heldout = []
        for task in ('generation', 'editing'):
            eligible = [sample for sample in selected if sample['task'] == task and sample['id'] not in blocked]
            requested = settings['test_counts'].get(task)
            if requested is not None and len(eligible) < requested:
                raise ValueError(f'Only {len(eligible)} uncontaminated {task} cases remain; requested {requested}')
            heldout.extend(eligible if requested is None else eligible[:requested])
        selected = heldout
    selected_ids = {sample['id'] for sample in selected}
    errors = [error for error in errors if error['sample_id'] in selected_ids]
    unknown = sum(unknown_by_id[key] for key in selected_ids)
    source_files = [root / 'cadfusion/test.json', root / 'cad_editor/dataset/test.json',
                    root / 'skexgen/cad_data/train_deduplicate_s.pkl']
    manifest = {'schema': 'cad-benchmark-v1', 'seed': settings['seed'],
                'selection': 'Seeded upstream test subset; no filtering by token/primitive limits or inference success',
                'counts': dict(collections.Counter(x['task'] for x in selected)),
                'settings_sha256': content_hash(settings), 'training_database_sha256': file_hash(database),
                'source_files': {str(p): file_hash(p) for p in source_files},
                'overlap_audit': {'conflicting_endpoints': 0, 'unparseable_endpoints': unknown,
                                  'excluded_training_overlap_samples': sorted(blocked),
                                  'excluded_training_overlap_endpoints': conflicts,
                                  'scope': 'Exact canonical fingerprints; unknown upstream geometry lineage cannot be recovered'},
                'paper_test_set_verified': False,
                'comparability_notes': [
                    'Uses released CADFusion/CAD-Editor upstream test records, not the unreleased combined paper manifest.',
                    'The paper\'s additional 200 generation annotations have not been obtained.',
                    'Training uses grouped holdouts and explicit implementation choices described in README.md.',
                    'When enabled, training-overlap cases are excluded before seeded sampling and recorded in overlap_audit.',
                ], 'reference_errors': errors, 'samples': sorted(selected, key=lambda x: x['id'])}
    validate_manifest(manifest)
    write_json(target, manifest)
    print(json.dumps({'manifest': str(target), 'counts': manifest['counts'], 'reference_errors': len(errors)}), flush=True)
    return manifest


def prediction_paths(settings, override=None):
    path = Path(override or Path(settings['output']) / 'predictions.jsonl')
    return path, path.with_suffix('.meta.json')


def read_predictions(path, manifest, recover_tail=False):
    """Strict ID validation, with explicit interrupted-last-line recovery on resume."""
    path = Path(path)
    known = {s['id'] for s in manifest['samples']}
    rows = {}
    if not path.exists():
        return rows
    with path.open('rb') as f:
        while True:
            offset = f.tell()
            raw = f.readline()
            if not raw:
                break
            try:
                row = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if recover_tail and not raw.endswith(b'\n') and not f.read(1):
                    with path.open('r+b') as writable:
                        writable.truncate(offset)
                    break
                raise ValueError(f'Corrupted prediction JSONL at byte {offset}')
            key = row.get('sample_id')
            if key not in known or key in rows:
                raise ValueError(f'Unknown or duplicate prediction sample ID: {key}')
            if row.get('status') not in ('syntax_valid', 'inference_error'):
                raise ValueError(f'Invalid prediction status for {key}')
            if row['status'] == 'syntax_valid' and not isinstance(row.get('generated_sequence'), str):
                raise ValueError(f'Missing generated sequence for {key}')
            rows[key] = row
    return rows
