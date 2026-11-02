from __future__ import annotations

import collections
import json
import random
import sqlite3
from pathlib import Path

from .common import file_hash, write_json


def native_sequence(pairs, boundaries):
    return [token for pair in pairs for token in [*pair, boundaries['pair_end']]]


def instruction_tokens(tokenizer, boundaries, instruction, source=None):
    task = 'editing' if source is not None else 'generation'
    text = f'Task: CAD {task}\nInstruction: {instruction}\n'
    result = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    result += tokenizer.encode(text, add_special_tokens=False)
    if source is not None:
        result += tokenizer.encode('Original CAD:\n', add_special_tokens=False)
        result += [boundaries['start']] + native_sequence(source, boundaries) + [boundaries['end']]
    result += tokenizer.encode('Output CAD:\n', add_special_tokens=False) + [boundaries['start']]
    return result


def prepare_sft_tokens(config, tokenizer, alignment):
    prepared = config['data']['database']
    encoded = config['sft']['encoded_database']
    output = Path(config['sft']['output'])
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'tokens.sqlite'
    signature = {
        'config': config,
        'database_sha256': file_hash(prepared), 'encoded_sha256': file_hash(encoded),
        'alignment_sha256': file_hash(Path(config['alignment']['output']) / 'alignment.pt'),
    }
    if target.exists():
        with sqlite3.connect(target) as db:
            old = json.loads(db.execute('SELECT value FROM metadata WHERE key="signature"').fetchone()[0])
        if old != signature:
            raise ValueError('SFT token cache is incompatible; use a new output directory')
        return target
    temporary = target.with_suffix('.sqlite.tmp')
    temporary.unlink(missing_ok=True)
    dest = sqlite3.connect(temporary)
    dest.executescript('''
        CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE samples (id INTEGER PRIMARY KEY, example_id TEXT UNIQUE, split TEXT,
            task TEXT, input_ids TEXT, prompt_length INTEGER);
        CREATE INDEX samples_split ON samples(split);
    ''')
    source = sqlite3.connect(prepared)
    source.execute('ATTACH DATABASE ? AS encoded', (str(Path(encoded).resolve()),))
    code_metadata = dict(source.execute('SELECT key,value FROM encoded.metadata'))
    if json.loads(code_metadata['alignment_sha256']) != signature['alignment_sha256']:
        raise ValueError('Encoded geometry was produced with another alignment checkpoint')
    if json.loads(code_metadata['database_sha256']) != signature['database_sha256']:
        raise ValueError('Encoded geometry was produced from another prepared database')
    all_train = source.execute('SELECT id,task FROM examples WHERE split="train" ORDER BY id').fetchall()
    generations = [key for key, task in all_train if task == 'generation']
    edits = [key for key, task in all_train if task == 'editing']
    if not generations or not edits:
        raise ValueError('Joint training needs both generation and editing examples')
    random.Random(config.get('seed', 42)).shuffle(edits)
    selected = set(generations + edits[:int(len(generations) * config['sft']['editing_per_generation'])])
    counts, dropped = collections.Counter(), collections.Counter()
    query = '''SELECT e.id,e.split,e.task,e.instruction,s.native_pairs,t.native_pairs
               FROM examples e LEFT JOIN encoded.geometry s ON s.id=e.source_id
               JOIN encoded.geometry t ON t.id=e.target_id ORDER BY e.id'''
    for key, split, task, instruction, original, target_pairs in source.execute(query):
        if split == 'train' and key not in selected:
            continue
        source_pairs = json.loads(original) if original else None
        if task == 'editing' and source_pairs is None:
            raise ValueError(f'Missing encoded source geometry: {key}')
        prompt = instruction_tokens(tokenizer, alignment['boundaries'], instruction, source_pairs)
        target_ids = native_sequence(json.loads(target_pairs), alignment['boundaries'])
        ids = prompt + target_ids + [alignment['boundaries']['end'], tokenizer.eos_token_id]
        if len(ids) > config['sft']['context_length']:
            dropped[f'{split}:{task}:context_overflow'] += 1
            continue
        dest.execute('INSERT INTO samples(example_id,split,task,input_ids,prompt_length) VALUES(?,?,?,?,?)',
                     (key, split, task, json.dumps(ids), len(prompt)))
        counts[f'{split}:{task}'] += 1
    if not counts['train:generation'] or not counts['train:editing'] or not sum(v for k, v in counts.items() if k.startswith('val:')):
        raise ValueError(f'No usable joint train/validation examples after length filtering: {dict(counts)}')
    dest.execute('INSERT INTO metadata VALUES("signature",?)', (json.dumps(signature, sort_keys=True),))
    dest.commit()
    source.close()
    dest.close()
    temporary.replace(target)
    report = {'retained': dict(counts), 'dropped': dict(dropped), 'requested_editing_per_generation': config['sft']['editing_per_generation'],
              'mixture_note': 'Sampled without replacement before context filtering. Retained per-task counts show the realized mixture.'}
    write_json(output / 'data_report.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return target
