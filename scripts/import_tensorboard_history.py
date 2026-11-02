#!/usr/bin/env python3
"""Export completed VQ-VAE JSONL metrics as an epoch-indexed TensorBoard run."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


TAGS = {
    'train_loss': 'train/loss',
    'validation_loss': 'eval/loss',
    'validation_token_accuracy': 'eval/token_accuracy',
    'used_codes': 'codebook/used_codes',
    'codebook_perplexity': 'codebook/perplexity',
}


def export(metrics, logdir):
    source = Path(metrics)
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    rows = [json.loads(line) for line in content.decode().splitlines() if line.strip()]
    if not rows:
        raise ValueError('No historical metrics to export')
    previous = 0
    for row in rows:
        epoch = row['epoch']
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= previous:
            raise ValueError('Historical epochs must be strictly increasing positive integers')
        previous = epoch
        for key in TAGS:
            if key not in row or not math.isfinite(float(row[key])):
                raise ValueError(f'Missing or non-finite {key} at epoch {epoch}')
    destination = Path(logdir)
    manifest = destination / 'import.json'
    if manifest.exists():
        existing = json.loads(manifest.read_text())
        if existing['source_sha256'] == digest and any(destination.glob('events.out.tfevents*')):
            print(json.dumps({'status': 'already_imported', **existing}, indent=2))
            return
        raise FileExistsError('Historical export differs; select a fresh logdir')
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Refusing to overwrite an existing TensorBoard run')

    # TensorBoard's protobuf writer imports no model or GPU framework.
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    destination.mkdir(parents=True, exist_ok=True)
    writer = EventFileWriter(str(destination))
    try:
        for row in rows:
            summary = Summary(value=[Summary.Value(tag=tag, simple_value=float(row[key]))
                                     for key, tag in TAGS.items()])
            writer.add_event(Event(wall_time=time.time(), step=row['epoch'], summary=summary))
        writer.flush()
    finally:
        writer.close()
    best = min(rows, key=lambda row: row['validation_loss'])
    record = {'source': str(source), 'source_sha256': digest, 'epochs': len(rows),
              'first_epoch': rows[0]['epoch'], 'last_epoch': rows[-1]['epoch'],
              'best_epoch': best['epoch'], 'best_validation_loss': best['validation_loss'],
              'step_axis': 'epoch',
              'wall_time_note': 'Imported now; original epoch timestamps were not recorded.',
              'tags': list(TAGS.values())}
    manifest.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'status': 'imported', **record}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', default='artifacts/vq/metrics.jsonl')
    parser.add_argument('--logdir', default='artifacts/vq/tensorboard-history')
    arguments = parser.parse_args()
    export(arguments.metrics, arguments.logdir)
