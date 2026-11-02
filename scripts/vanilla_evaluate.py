"""Vanilla control evaluation: greedy text generation on the frozen manifest, then the unchanged kernel and scorer.

Stage `shard` (one GPU each, `CUDA_VISIBLE_DEVICES=i`): generate for manifest samples i, i+n, ...
Stage `score` (after all shards): merge, write predictions and metadata, run the kernel and benchmark scorer.
"""
import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from cadtokenizer.bench_data import content_hash, file_hash, load_manifest
from cadtokenizer.common import read_config, require_container, write_json
from cadtokenizer.vanilla import prompt_text


def generate_shard(config, settings, shard, shards):
    require_container()
    root = Path(settings['output'])
    manifest = load_manifest(root / 'manifest.json')
    manifest_hash = file_hash(root / 'manifest.json')
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], revision=config['revision'])
    base = AutoModelForCausalLM.from_pretrained(config['model_name'], revision=config['revision'], torch_dtype=torch.bfloat16,
                                                attn_implementation='sdpa').cuda()
    model = PeftModel.from_pretrained(base, Path(config['output']) / 'final_adapter').eval()
    eos = tokenizer.eos_token_id
    journal = root / 'shards' / f'text-{shard}.jsonl'
    journal.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)['sample_id'] for line in journal.read_text().splitlines()} if journal.exists() else set()
    with journal.open('a') as out:
        for index, sample in enumerate(manifest['samples']):
            if index % shards != shard or sample['id'] in done:
                continue
            prompt = tokenizer.encode(prompt_text(sample['task'], sample['instruction'], sample.get('source_sequence')), add_special_tokens=True)
            ids = torch.tensor([prompt], device='cuda')
            with torch.inference_mode():
                result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                        max_new_tokens=settings['max_new_tokens'], eos_token_id=eos, pad_token_id=eos)
            generated = result[0, len(prompt):].tolist()
            if eos in generated:
                generated = generated[:generated.index(eos)]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            out.write(json.dumps({'sample_id': sample['id'], 'status': 'syntax_valid', 'generated_sequence': text,
                                  'generated_native_ids': generated, 'manifest_sha256': manifest_hash}) + '\n')
            out.flush()


def score(config, settings):
    require_container()
    from cadtokenizer.bench_runtime import KernelRunner
    from cadtokenizer.bench_score import benchmark_score
    root = Path(settings['output'])
    manifest_path = root / 'manifest.json'
    manifest = load_manifest(manifest_path)
    manifest_hash = file_hash(manifest_path)
    rows = {}
    for path in sorted((root / 'shards').glob('text-*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            rows[row['sample_id']] = row
    if len(rows) != len(manifest['samples']):
        raise ValueError(f'Incomplete text predictions: {len(rows)} of {len(manifest["samples"])}')
    signature = {'manifest_sha256': manifest_hash, 'model_config': config, 'max_new_tokens': settings['max_new_tokens'],
                 'seed': settings['seed'], 'decoding': 'greedy_native_tokenizer_text_cad',
                 'cad_decoder': 'text', 'adapter_sha256': file_hash(Path(config['output']) / 'final_adapter' / 'adapter_model.safetensors')}
    signature_hash = content_hash(signature)
    write_json(root / 'predictions.meta.json', {'schema': 'cad-predictions-v1', 'signature': signature, 'signature_sha256': signature_hash})
    with (root / 'predictions.jsonl').open('w') as handle:
        for sample in manifest['samples']:
            row = rows[sample['id']]
            handle.write(json.dumps({**row, 'run_signature_sha256': signature_hash}) + '\n')
    runner = KernelRunner(settings)
    jobs = []
    for sample in manifest['samples']:
        jobs.append((rows[sample['id']]['generated_sequence'], sample['id'], 'prediction'))
        jobs.append((sample.get('target_sequence'), sample['id'], 'reference'))
    with ThreadPoolExecutor(max_workers=settings.get('kernel_workers', 8)) as pool:
        list(pool.map(lambda job: runner.reconstruct(*job), jobs))
    report = benchmark_score(config, settings)
    write_json(root / 'evaluation_complete.json', {
        'status': 'complete', 'variant': 'vanilla_text', 'tasks': {task: value['table_metrics'] for task, value in report['tasks'].items()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['shard', 'score'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--benchmark-config', required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=4)
    args = parser.parse_args()
    config, settings = read_config(args.config), read_config(args.benchmark_config)
    if args.stage == 'shard':
        generate_shard(config, settings, args.shard, args.shards)
    else:
        score(config, settings)


if __name__ == '__main__':
    main()
