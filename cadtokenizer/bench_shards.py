"""Validate and merge independent GPU journals against one frozen manifest."""
import json
from pathlib import Path

from .bench_data import content_hash, load_manifest, prediction_paths, read_predictions
from .bench_predict import partition_samples
from .common import file_hash, write_json


def merge_shards(settings, count):
    if not isinstance(count, int) or count < 1:
        raise ValueError('Shard count must be positive')
    root = Path(settings['output'])
    manifest_path = root / 'manifest.json'
    manifest = load_manifest(manifest_path)
    manifest_hash = file_hash(manifest_path)
    records, metadata = {}, None
    provenance = {}
    for index in range(count):
        path, meta_path = prediction_paths(settings, root / 'shards' / f'predictions-{index}.jsonl')
        current = json.loads(meta_path.read_text())
        if (current.get('schema') != 'cad-predictions-v1'
                or current['signature']['manifest_sha256'] != manifest_hash
                or current['signature_sha256'] != content_hash(current['signature'])):
            raise ValueError('Invalid shard metadata or manifest fingerprint')
        if metadata is not None and current['signature'] != metadata['signature']:
            raise ValueError('Shard model/data/code signatures differ')
        metadata = current
        rows = read_predictions(path, manifest)
        expected = {s['id'] for s in partition_samples(manifest['samples'], index, count)}
        if set(rows) != expected:
            raise ValueError(f'Shard {index} is incomplete or has incorrectly assigned sample IDs')
        for row in rows.values():
            if (row.get('manifest_sha256') != manifest_hash
                    or row.get('run_signature_sha256') != current['signature_sha256']):
                raise ValueError('Shard row fingerprint mismatch')
        if records.keys() & rows.keys():
            raise ValueError('Duplicate prediction IDs across shards')
        records.update(rows)
        provenance[str(path)] = file_hash(path)
    if len(records) != len(manifest['samples']):
        raise ValueError('Merged predictions do not cover the complete manifest')
    path, meta_path = prediction_paths(settings)
    if path.exists() or meta_path.exists():
        if not meta_path.exists() or json.loads(meta_path.read_text())['signature'] != metadata['signature']:
            raise FileExistsError('Refusing to overwrite a different merged run')
        if path.exists():
            if read_predictions(path, manifest) != records:
                raise FileExistsError('Refusing to overwrite different merged predictions')
            return
    temporary = path.with_suffix('.jsonl.tmp')
    with temporary.open('w') as stream:
        for sample in manifest['samples']:
            stream.write(json.dumps(records[sample['id']], allow_nan=False) + '\n')
    write_json(meta_path, {**metadata, 'shard_files': provenance})
    temporary.replace(path)
    print(json.dumps({'merged_predictions': len(records), 'shards': count}), flush=True)
