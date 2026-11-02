"""RFT round 1, sampling: k candidates per training generation prompt, kernel-checked, one journal per GPU shard."""
import argparse
import json
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cadtokenizer.bench_runtime import KernelRunner
from cadtokenizer.common import read_config, require_container, seed_all
from cadtokenizer.inference import CadGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--settings', required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, required=True)
    args = parser.parse_args()
    require_container()
    config, settings = read_config(args.config), read_config(args.settings)
    root = Path(settings['output'])
    subset = json.loads((root / 'subset.json').read_text())[args.shard::args.shards]
    journal = root / 'shards' / f'candidates-{args.shard}.jsonl'
    journal.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)['sample_id'] for line in journal.read_text().splitlines()} if journal.exists() else set()
    seed_all(settings['seed'] + args.shard)
    generator = CadGenerator(config)
    runner = KernelRunner(settings)

    def kernel(key, sequence):
        return runner.reconstruct(sequence, key, 'rft')['status']

    pending = deque()
    written = 0

    def drain(out, keep):
        nonlocal written
        while pending and (len(pending) > keep or all(future is None or future.done() for future in pending[0][3])):
            key, instruction, candidates, futures = pending.popleft()
            results = []
            for index, (candidate, future) in enumerate(zip(candidates, futures)):
                if future is None:
                    results.append({'index': index, 'status': candidate.get('status'), 'native_ids': candidate.get('generated_native_ids'),
                                    'sequence': None, 'kernel': 'not_checked'})
                else:
                    results.append({'index': index, 'status': 'syntax_valid', 'native_ids': candidate['generated_native_ids'],
                                    'sequence': candidate['generated_sequence'], 'kernel': future.result()})
            out.write(json.dumps({'sample_id': key, 'instruction': instruction, 'candidates': results}) + '\n')
            out.flush()
            written += 1
            if written % 50 == 0:
                print(json.dumps({'shard': args.shard, 'prompts_written': written, 'of': len(subset)}), flush=True)

    with ThreadPoolExecutor(max_workers=settings['kernel_workers']) as pool, journal.open('a') as out:
        todo = [(key, instruction) for key, instruction in subset if key not in done]
        lengths = {key: generator.prompt_length(instruction, settings['max_new_tokens']) for key, instruction in todo}
        # Batch only prompts of equal token length, so no padding enters the batched attention.
        buckets = defaultdict(list)
        for item in todo:
            buckets[lengths[item[0]]].append(item)
        groups = []
        for items in buckets.values():
            for start in range(0, len(items), settings['prompts_per_call']):
                groups.append(items[start:start + settings['prompts_per_call']])
        for group in groups:
            batches = generator.predict_candidates_batch([instruction for _, instruction in group], settings['max_new_tokens'],
                                                         settings['k'], settings['temperature'])
            for (key, instruction), candidates in zip(group, batches):
                shared, futures = {}, []
                for candidate in candidates:
                    if candidate.get('status') != 'syntax_valid':
                        futures.append(None)
                    elif candidate['generated_sequence'] in shared:
                        futures.append(shared[candidate['generated_sequence']])
                    else:
                        shared[candidate['generated_sequence']] = pool.submit(kernel, key, candidate['generated_sequence'])
                        futures.append(shared[candidate['generated_sequence']])
                pending.append((key, instruction, candidates, futures))
            drain(out, keep=64)
        drain(out, keep=0)
    print(json.dumps({'shard': args.shard, 'complete': True}), flush=True)


if __name__ == '__main__':
    main()
