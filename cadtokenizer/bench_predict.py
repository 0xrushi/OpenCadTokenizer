from __future__ import annotations

import json
import os
from pathlib import Path
import time

from .bench_data import code_hash, content_hash, load_manifest, prediction_paths, read_predictions
from .common import file_hash, require_container, seed_all, write_json


def model_fingerprints(config):
    adapter_dir = Path(config['sft']['output']) / 'final_adapter'
    weights = sorted(adapter_dir.glob('adapter_model.*'))
    if not weights:
        raise FileNotFoundError(f'No trained LoRA weights under {adapter_dir}')
    paths = [Path(config['vq']['output']) / 'best.pt', Path(config['alignment']['output']) / 'alignment.pt',
             Path(config['sft']['output']) / 'run.json', adapter_dir / 'adapter_config.json', *weights]
    tokenizer_dir = Path(config['alignment']['output']) / 'llm_tokenizer'
    paths += [path for path in sorted(tokenizer_dir.iterdir()) if path.is_file()]
    return {str(path): file_hash(path) for path in paths}


def predict_samples(samples, generator, output, context, existing=None, limit=None):
    """Reusable journaling loop. Ground truth is deliberately not passed to predict."""
    existing = existing or {}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # A complete last JSON object without a final newline is recoverable too.
    if output.exists() and output.stat().st_size:
        with output.open('rb+') as f:
            f.seek(-1, 2)
            if f.read(1) != b'\n':
                f.seek(0, 2)
                f.write(b'\n')
    written = 0
    batch_size = context.get('inference_batch_size', 1)
    pending = [s for s in samples if s['id'] not in existing]
    if limit is not None:
        pending = pending[:limit]
    with output.open('a') as f:
        for start in range(0, len(pending), batch_size):
            chunk = pending[start:start + batch_size]
            started = time.monotonic()
            if batch_size > 1:
                requests = [(s['instruction'], s.get('source_sequence')) for s in chunk]
                valid = [i for i, s in enumerate(chunk) if not s.get('instruction_error')]
                outputs = {}
                for i, result in zip(valid, generator.predict_batch([requests[i] for i in valid], context['max_new_tokens'], batch_size)):
                    outputs[i] = result
            for position, sample in enumerate(chunk):
                try:
                    if sample.get('instruction_error'):
                        raise ValueError(sample['instruction_error'])
                    if batch_size > 1:
                        prediction = outputs[position]
                        if isinstance(prediction, Exception):
                            raise prediction
                    else:
                        prediction = generator.predict(sample['instruction'], sample.get('source_sequence'), context['max_new_tokens'])
                    record = {'sample_id': sample['id'], 'status': 'syntax_valid',
                              'generated_sequence': prediction['generated_sequence'],
                              'generated_native_ids': prediction.get('generated_native_ids')}
                except (ValueError, KeyError) as e:
                    # Includes source/context overflow. These examples stay in the denominator.
                    record = {'sample_id': sample['id'], 'status': 'inference_error',
                              'generated_sequence': None, 'error': str(e), 'error_type': type(e).__name__}
                # Infrastructure/CUDA exceptions abort the run; they are not model-quality scores.
                record.update({'manifest_sha256': context['manifest_sha256'],
                               'run_signature_sha256': context['run_signature_sha256'],
                               'elapsed_seconds': (time.monotonic() - started) / len(chunk)})
                f.write(json.dumps(record, allow_nan=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
                written += 1
                print(json.dumps({'sample_id': sample['id'], 'status': record['status'], 'new_predictions': written}), flush=True)
    return written


def partition_samples(samples, index=0, count=1):
    if not isinstance(count, int) or not isinstance(index, int) or count < 1 or not 0 <= index < count:
        raise ValueError('Invalid prediction shard index/count')
    return samples[index::count]


def benchmark_predict(config, settings, manifest_path=None, predictions_path=None, resume=False, limit=None,
                      shard_index=0, shard_count=1, batch_size=None):
    require_container()
    manifest_path = Path(manifest_path or Path(settings['output']) / 'manifest.json')
    manifest = load_manifest(manifest_path)
    samples = partition_samples(manifest['samples'], shard_index, shard_count)
    if shard_count > 1 and predictions_path is None:
        raise ValueError('Sharded prediction requires a separate journal path per worker')
    path, metadata_path = prediction_paths(settings, predictions_path)
    run_metadata = json.loads((Path(config['sft']['output']) / 'run.json').read_text())
    if manifest.get('training_database_sha256') != run_metadata['database_sha256']:
        raise ValueError('Benchmark was audited against another training database; freeze/audit the test set against this model\'s data first')
    batch_size = batch_size or settings.get('inference_batch_size', 1)
    signature = {'manifest_sha256': file_hash(manifest_path), 'model_files': model_fingerprints(config),
                 'model_config': config, 'max_new_tokens': settings['max_new_tokens'],
                 'decoding': 'greedy_native_framing_and_vq_fsa' if batch_size == 1 else f'greedy_native_framing_and_vq_fsa_batched_{batch_size}', 'seed': settings['seed'],
                 'cad_decoder': config.get('inference', {}).get('cad_decoder', 'alignment'),
                 'code_sha256': code_hash(), 'image_id': os.environ.get('CADTOKENIZER_IMAGE_ID')}
    if path.exists() or metadata_path.exists():
        if not resume:
            raise FileExistsError('Prediction run exists; use --resume-predictions to continue the same frozen run')
        if not metadata_path.exists():
            raise ValueError('Prediction metadata is missing; cannot validate resume')
        previous = json.loads(metadata_path.read_text())
        if previous.get('signature') != signature:
            raise ValueError('Prediction resume mismatch: manifest, model, code, settings or pinned image changed')
    else:
        write_json(metadata_path, {'schema': 'cad-predictions-v1', 'signature': signature,
                                   'signature_sha256': content_hash(signature)})
    existing = read_predictions(path, manifest, recover_tail=resume)
    if not set(existing).issubset({sample['id'] for sample in samples}):
        raise ValueError('Prediction journal contains samples assigned to another shard')
    signature_hash = content_hash(signature)
    if any(row.get('run_signature_sha256') != signature_hash or row.get('manifest_sha256') != signature['manifest_sha256'] for row in existing.values()):
        raise ValueError('Prediction rows do not match the frozen run metadata')
    if len(existing) == len(samples):
        print('All frozen benchmark samples already have predictions.', flush=True)
        return
    seed_all(settings['seed'])
    from .inference import CadGenerator
    generator = CadGenerator(config)
    context = {'manifest_sha256': signature['manifest_sha256'], 'run_signature_sha256': signature_hash,
               'max_new_tokens': settings['max_new_tokens'], 'inference_batch_size': batch_size}
    predict_samples(samples, generator, path, context, existing, limit)
