from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .common import device_for, file_hash, load_checkpoint, require_container
from .vq import PairDataset, collate_pairs, load_vq, to_device


def trailer_native(pair_tokens, trailer_ids):
    """Native IDs for the three exact scale/origin bins of one pair; empty when the trailer is off."""
    if not trailer_ids:
        return []
    from .scale_origin_codec import metadata_for_pair
    return [trailer_ids[value] for value in metadata_for_pair(pair_tokens)]


def encode_geometry(config, overwrite=False):
    require_container()
    device = device_for(config)
    vq_path = Path(config['vq']['output']) / 'best.pt'
    align_path = Path(config['alignment']['output']) / 'alignment.pt'
    alignment = load_checkpoint(align_path)
    if alignment['vq_sha256'] != file_hash(vq_path):
        raise ValueError('Alignment and VQ checkpoints do not match')
    model, checkpoint = load_vq(vq_path, device)
    if checkpoint['database_sha256'] != file_hash(config['data']['database']):
        raise ValueError('VQ model was trained on another prepared database')
    mapping = alignment['native_ids'].to(device)
    trailer_ids = alignment.get('trailer_ids')
    output = Path(config['sft']['encoded_database'])
    if output.exists() and not overwrite:
        raise FileExistsError(f'{output} exists; pass --overwrite to rebuild')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.sqlite.tmp')
    temporary.unlink(missing_ok=True)
    db = sqlite3.connect(temporary)
    db.executescript('''CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL);
                        CREATE TABLE geometry (id TEXT PRIMARY KEY,code_pairs TEXT,native_pairs TEXT);''')
    dataset = PairDataset(config['data']['database'])
    def collate(items):
        return collate_pairs(items), items
    loader = DataLoader(dataset, batch_size=config['vq']['batch_size'], collate_fn=collate, num_workers=0)
    current, codes, native, written = None, [], [], 0
    with torch.inference_mode():
        for step, (batch, items) in enumerate(loader):
            batch = to_device(batch, device)
            _, indices, mask = model.encode(batch['tokens'], batch['groups'])
            for row, item in enumerate(items):
                if current != item['geometry_id']:
                    if current is not None:
                        db.execute('INSERT INTO geometry VALUES(?,?,?)', (current, json.dumps(codes), json.dumps(native)))
                        written += 1
                    current, codes, native = item['geometry_id'], [], []
                if item['ordinal'] != len(codes):
                    raise ValueError('Sketch–extrusion pair ordering is inconsistent')
                selected = indices[row][mask[row]]
                codes.append(selected.cpu().tolist())
                native.append(mapping[selected].cpu().tolist() + trailer_native(item['tokens'], trailer_ids))
            if step % 100 == 0:
                db.commit()
                print(json.dumps({'batch': step, 'encoded_geometry': written}), flush=True)
    if current is not None:
        db.execute('INSERT INTO geometry VALUES(?,?,?)', (current, json.dumps(codes), json.dumps(native)))
    metadata = {'vq_sha256': file_hash(vq_path), 'alignment_sha256': file_hash(align_path),
                'database_sha256': file_hash(config['data']['database']),
                'trailer': bool(trailer_ids)}
    db.executemany('INSERT INTO metadata VALUES(?,?)', [(k, json.dumps(v)) for k, v in metadata.items()])
    db.commit()
    print(json.dumps({'encoded_geometry': db.execute('SELECT COUNT(*) FROM geometry').fetchone()[0]}), flush=True)
    db.close()
    temporary.replace(output)
