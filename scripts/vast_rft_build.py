"""RFT round 1, build: keep kernel-valid, structure-matched candidates and add them to the SFT cache.

The base cache is built by the unchanged `prepare_sft_tokens`, so the cache signature still covers the
config; the RFT rows are then inserted under example IDs `rft:<prompt>:<n>` with split `train`.
"""
import argparse
import collections
import glob
import json
import sqlite3
from pathlib import Path

from transformers import AutoTokenizer

from cadtokenizer.common import load_checkpoint, read_config, write_json
from cadtokenizer.instruction_data import instruction_tokens, prepare_sft_tokens
from cadtokenizer.rft import accept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    rft = config['rft']
    alignment = load_checkpoint(Path(config['alignment']['output']) / 'alignment.pt')
    tokenizer = AutoTokenizer.from_pretrained(Path(config['alignment']['output']) / 'llm_tokenizer')
    tokens = prepare_sft_tokens(config, tokenizer, alignment)
    boundaries = alignment['boundaries']
    end, eos = boundaries['end'], tokenizer.eos_token_id
    report = collections.Counter()
    rows = []
    for path in sorted(glob.glob(str(Path(rft['source']) / 'shards' / 'candidates-*.jsonl'))):
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            kept = {}
            for candidate in record['candidates']:
                report['candidates'] += 1
                if candidate['status'] != 'syntax_valid':
                    report['not_decoded'] += 1
                    continue
                if candidate['kernel'] != 'valid':
                    report['kernel_invalid'] += 1
                    continue
                if not accept(candidate['sequence'], candidate['native_ids'], end, True, rft['loop_limit']):
                    report['structure_rejected'] += 1
                    continue
                if candidate['sequence'] in kept:
                    report['duplicate'] += 1
                    continue
                if len(kept) >= rft['per_prompt_cap']:
                    report['over_cap'] += 1
                    continue
                kept[candidate['sequence']] = candidate['native_ids']
            prompt = instruction_tokens(tokenizer, boundaries, record['instruction'])
            for n, native_ids in enumerate(kept.values()):
                ids = prompt + native_ids + [eos]
                if len(ids) > config['sft']['context_length']:
                    report['context_overflow'] += 1
                    continue
                rows.append((f"rft:{record['sample_id']}:{n}", 'train', 'generation', json.dumps(ids), len(prompt)))
            report['prompts'] += 1
            report['prompts_with_rows'] += bool(kept)
    with sqlite3.connect(tokens) as db:
        db.execute('DELETE FROM samples WHERE example_id LIKE "rft:%"')
        db.executemany('INSERT INTO samples(example_id,split,task,input_ids,prompt_length) VALUES(?,?,?,?,?)', rows)
        db.commit()
    report['rows_added'] = len(rows)
    write_json(Path(config['sft']['output']) / 'rft_report.json', dict(report))
    print(json.dumps(dict(report), indent=2), flush=True)


if __name__ == '__main__':
    main()
