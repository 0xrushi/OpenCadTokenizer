from __future__ import annotations

import collections
import hashlib
import json
import pickle
import sqlite3
from pathlib import Path

from .common import fraction, write_json
from .representation import analyze, skexgen_to_sem

SCHEMA_VERSION = 'cad-sem-v1'


class UnionFind:
    def __init__(self):
        self.parents = {}

    def find(self, x):
        self.parents.setdefault(x, x)
        root = x
        while self.parents[root] != root:
            root = self.parents[root]
        while x != root:
            x, self.parents[x] = self.parents[x], root
        return root

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        # Stable representative independent of ingestion order.
        self.parents[max(a, b)] = min(a, b)


def prepare(config, limit=None, overwrite=False):
    data = config['data']
    target = Path(data['database'])
    if target.exists() and not overwrite:
        raise FileExistsError(f'{target} exists; use --overwrite to rebuild explicitly')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.sqlite.tmp')
    temporary.unlink(missing_ok=True)
    db = sqlite3.connect(temporary)
    db.executescript('''
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE geometry (
            id TEXT PRIMARY KEY, sequence TEXT NOT NULL, split TEXT NOT NULL,
            tokenizer_candidate INTEGER NOT NULL DEFAULT 0, tokenizer_split TEXT,
            component TEXT
        );
        CREATE TABLE pairs (
            id INTEGER PRIMARY KEY, geometry_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
            tokens TEXT NOT NULL, groups TEXT NOT NULL,
            UNIQUE(geometry_id, ordinal)
        );
        CREATE TABLE examples (
            id TEXT PRIMARY KEY, task TEXT NOT NULL, instruction TEXT NOT NULL,
            source_id TEXT, target_id TEXT NOT NULL, dataset TEXT NOT NULL,
            upstream_split TEXT NOT NULL, split TEXT
        );
        CREATE INDEX pairs_geometry ON pairs(geometry_id, ordinal);
    ''')
    uf = UnionFind()
    errors, error_samples = collections.Counter(), []
    ranks = {'train': 0, 'val': 1, 'test': 2}
    seed = config.get('seed', 42)

    def add_geometry(text, split, candidate=False):
        canonical, pairs = analyze(
            text, data['max_pair_tokens'], data['max_primitives'], data['extrusion_partition'],
            data.get('sketch_partition', 'curve'))
        key = hashlib.sha256(canonical.encode()).hexdigest()
        existing = db.execute('SELECT split FROM geometry WHERE id=?', (key,)).fetchone()
        if existing:
            if ranks[split] > ranks[existing[0]]:
                db.execute('UPDATE geometry SET split=? WHERE id=?', (split, key))
            if candidate:
                db.execute('UPDATE geometry SET tokenizer_candidate=1 WHERE id=?', (key,))
        else:
            db.execute('INSERT INTO geometry(id,sequence,split,tokenizer_candidate) VALUES(?,?,?,?)',
                       (key, canonical, split, int(candidate)))
            db.executemany('INSERT INTO pairs(geometry_id,ordinal,tokens,groups) VALUES(?,?,?,?)',
                           [(key, i, json.dumps(p['tokens']), json.dumps(p['groups'])) for i, p in enumerate(pairs)])
            uf.find(key)
        return key

    def reject(dataset, index, exc):
        reason = str(exc).split(' exceeds ')[-1] if ' exceeds ' in str(exc) else type(exc).__name__
        errors[f'{dataset}:{reason}'] += 1
        if len(error_samples) < 50:
            error_samples.append({'dataset': dataset, 'index': str(index), 'reason': str(exc)})

    root = Path(data['root'])
    skex = root / 'skexgen/cad_data'
    with (skex / 'train_invalid.pkl').open('rb') as f:
        invalid = {str(v) for v in pickle.load(f)}
    training_geometry = None
    for split, filename in [('train', 'train_deduplicate_s.pkl'), ('val', 'val.pkl'), ('test', 'test.pkl')]:
        with (skex / filename).open('rb') as f:
            items = pickle.load(f)
        if split == 'train':
            # CADFusion serial_num indexes this exact list, never a sorted/rebuilt corpus.
            training_geometry = items
        for i, item in enumerate(items):
            if limit is not None and i >= limit:
                break
            if str(item['name']) in invalid:
                errors['skexgen:invalid_uid'] += 1
                continue
            try:
                add_geometry(skexgen_to_sem(item), split, candidate=True)
            except (ValueError, KeyError, IndexError) as e:
                reject('skexgen', i, e)
        db.commit()
        print(f'Imported SkexGen {split}', flush=True)
    for split in ('train', 'val', 'test'):
        annotations = json.loads((root / f'cadfusion/{split}.json').read_text())
        for i, item in enumerate(annotations[:limit] if limit is not None else annotations):
            try:
                index = int(item['serial_num'])
                geometry = training_geometry[index]
                if str(geometry['name']) in invalid:
                    raise ValueError('Caption references upstream invalid geometry')
                key = add_geometry(skexgen_to_sem(geometry), split, candidate=True)
                db.execute('INSERT INTO examples VALUES(?,?,?,?,?,?,?,NULL)',
                           (f'cadfusion:{split}:{i}', 'generation', item['description'], None, key, 'cadfusion', split))
            except (ValueError, KeyError, IndexError) as e:
                reject('cadfusion', i, e)
        db.commit()
    del training_geometry
    for split in ('train', 'test'):
        items = json.loads((root / f'cad_editor/dataset/{split}.json').read_text())
        for i, item in enumerate(items[:limit] if limit is not None else items):
            try:
                source = add_geometry(item['original_sequence'], split)
                dest = add_geometry(item['edited_sequence'], split)
                instruction = item['instruction'].strip()
                if not instruction:
                    raise ValueError('Empty instruction')
                uf.union(source, dest)
                db.execute('INSERT INTO examples VALUES(?,?,?,?,?,?,?,NULL)',
                           (f'cad_editor:{split}:{i}', 'editing', instruction, source, dest, 'cad_editor', split))
            except (ValueError, KeyError, IndexError) as e:
                reject('cad_editor', i, e)
        db.commit()
        print(f'Imported CAD-Editor {split}', flush=True)
    rows = db.execute('SELECT id,split FROM geometry').fetchall()
    components = {}
    for key, split in rows:
        component = uf.find(key)
        components[component] = max(components.get(component, 0), ranks[split])
    updates = []
    moved = collections.Counter()
    for key, old_split in rows:
        component = uf.find(key)
        rank = components[component]
        split = ('train', 'val', 'test')[rank]
        if split == 'train' and fraction(component, seed) < data['validation_fraction']:
            split = 'val'
        tokenizer_split = None
        if split == 'val':
            tokenizer_split = 'val'
        elif split == 'train' and fraction(key, seed + 1) < data['tokenizer_fraction']:
            tokenizer_split = 'train'
        if split != old_split:
            moved[f'{old_split}->{split}'] += 1
        updates.append((split, tokenizer_split, component, key))
    db.executemany('UPDATE geometry SET split=?,tokenizer_split=?,component=? WHERE id=?', updates)
    db.execute('UPDATE examples SET split=(SELECT split FROM geometry WHERE geometry.id=examples.target_id)')
    db.executescript('''
        CREATE INDEX geometry_split ON geometry(split);
        CREATE INDEX examples_split ON examples(split,task);
    ''')
    metadata = {'schema_version': SCHEMA_VERSION, 'data_config': data, 'seed': seed, 'limit_per_source_split': limit}
    db.executemany('INSERT INTO metadata VALUES(?,?)', [(k, json.dumps(v)) for k, v in metadata.items()])
    report = {
        **metadata,
        'geometry_counts': dict(db.execute('SELECT split,COUNT(*) FROM geometry GROUP BY split')),
        'example_counts': [{'split': a, 'task': b, 'count': c} for a, b, c in db.execute('SELECT split,task,COUNT(*) FROM examples GROUP BY split,task')],
        'tokenizer_pair_counts': dict(db.execute('SELECT g.tokenizer_split,COUNT(*) FROM pairs p JOIN geometry g ON g.id=p.geometry_id WHERE g.tokenizer_candidate=1 AND g.tokenizer_split IS NOT NULL GROUP BY g.tokenizer_split')),
        'promoted_geometry': dict(moved), 'excluded': dict(errors), 'exclusion_samples': error_samples,
        'split_policy': 'Upstream holdouts take precedence; exact canonical duplicates and linked edit endpoints share a component. Unknown upstream lineage cannot be reconstructed.',
    }
    if not report['tokenizer_pair_counts'].get('train') or not report['tokenizer_pair_counts'].get('val'):
        raise ValueError('No tokenizer training or validation pairs remain; inspect source data and limits')
    db.commit()
    db.close()
    temporary.replace(target)
    write_json(target.with_suffix('.report.json'), report)
    print(json.dumps(report, indent=2), flush=True)
