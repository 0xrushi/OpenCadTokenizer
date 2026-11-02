"""Read-only, pure-Python coverage and target-label audit for the lossless pilot."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sqlite3

from cadtokenizer.common import file_hash, write_json
from cadtokenizer.lossless_sem import LosslessSemCodec
from scripts.analyze_edit_correctness import align, units

OUT = Path('artifacts/lossless-edit-supervision-v1')


def verb(instruction):
    match = re.search(r'\w+', instruction)
    return match.group().lower() if match else '(empty)'


def has_numbers(instruction):
    return bool(re.search(r'(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])', instruction))


def changed_numeric_fields(source, target):
    """Count target numbers introduced/changed on the fixed command edit path."""
    _, events, _ = align(units(source), units(target))
    old = units(source)
    changed = Counter()
    fields = Counter()
    for (operation, index, value), multiplicity in events.items():
        if operation == 'delete':
            continue
        parts = value.split(',')
        previous = old[index].split(',') if operation == 'replace' else []
        for field, number in enumerate(parts[1:], 1):
            if previous and previous[0] == parts[0] and field < len(previous) and previous[field] == number:
                continue
            changed[number] += multiplicity
            fields[parts[0] + ':' + str(field)] += multiplicity
    return changed, fields


def main():
    pilot_path = Path('artifacts/lossless-sft-pilot-v1/manifest.json')
    pilot = json.loads(pilot_path.read_text())
    codec_path = Path('artifacts/lossless-sem-v2/codec.json')
    cache_path = Path('artifacts/lossless-sem-v2/tokens.sqlite')
    database = Path('data/prepared/cad.sqlite')
    hashes = {'database': file_hash(database), 'codec': file_hash(codec_path), 'cache': file_hash(cache_path)}
    expected = pilot['signature']
    if hashes != {'database': expected['source_database_sha256'], 'codec': expected['codec_sha256'], 'cache': expected['cache_sha256']}:
        raise ValueError('Pilot data or codec fingerprint changed')
    codec = LosslessSemCodec.from_dict(json.loads(codec_path.read_text()))
    selected = {r['example_id'] for r in pilot['train']}
    with sqlite3.connect(database) as db, sqlite3.connect(cache_path) as cache:
        # Restrict all queries to train and validation; do not inspect test cases.
        geometries = dict(db.execute("SELECT id,sequence FROM geometry WHERE split IN ('train','val')"))
        examples = {r[0]: r[1:] for r in db.execute("SELECT id,split,task,instruction,source_id,target_id FROM examples WHERE split IN ('train','val')")}
        cached = {r[0]: r[1:] for r in cache.execute("SELECT example_id,input_ids,prompt_length,context_overflow FROM samples WHERE split='train'")}
    coverage = defaultdict(Counter)
    for key, (split, task, instruction, source, target) in examples.items():
        if task != 'editing':
            continue
        tag = verb(instruction)
        coverage['prepared_' + split][tag] += 1
        if key in cached:
            coverage['cached_train'][tag] += 1
        if key in selected:
            coverage['pilot_train'][tag] += 1
    buckets = defaultdict(Counter)
    changed_bins = Counter()
    changed_field_counts = Counter()
    per_case = []
    all_target_tokens = all_edit_tokens = source_tokens_masked = 0
    decoded = 0
    for key in sorted(selected):
        split, task, instruction, source, target = examples[key]
        ids_text, prefix, overflow = cached[key]
        ids = json.loads(ids_text)
        if split != 'train' or overflow or len(ids) > 1024 or not 0 < prefix < len(ids):
            raise ValueError('Training split, prefix or context error')
        if ids[prefix-1] != codec.boundaries['start'] or ids[-1] != 128001:
            raise ValueError('Target frame or LM EOS mismatch')
        # labels = [-100]*prefix + ids[prefix:] in the actual trainer;
        # CAD-start is in the prompt, CAD-end and EOS are supervised.
        if codec.decode([codec.boundaries['start']] + ids[prefix:-1]) != geometries[target]:
            raise ValueError('Cached supervised target is not the reference')
        if source is not None:
            start = ids.index(codec.boundaries['start'])
            end = ids.index(codec.boundaries['end'], start) + 1
            if end >= prefix or codec.decode(ids[start:end]) != geometries[source]:
                raise ValueError('Cached source is incorrect or enters target labels')
            source_tokens_masked += end-start
        decoded += 1
        all_target_tokens += len(ids)-prefix
        if task != 'editing':
            continue
        src, tgt = geometries[source], geometries[target]
        changed, fields = changed_numeric_fields(src, tgt)
        numeric_count = sum(len(value.split(','))-1 for value in units(tgt))
        changes = sum(changed.values())
        tag = verb(instruction)
        counters = Counter(examples=1, explicit_numeric_instruction=int(has_numbers(instruction)),
                           numeric_target_fields=numeric_count, changed_numeric_target_fields=changes,
                           supervised_target_tokens=len(ids)-prefix,
                           numeric_only_edit=int([x.split(',')[0] for x in units(src)] == [x.split(',')[0] for x in units(tgt)]),
                           unchanged_target_numeric_fields=numeric_count-changes)
        if numeric_count < changes:
            raise ValueError('Changed numeric accounting exceeds target size')
        buckets['all'].update(counters)
        buckets[tag].update(counters)
        all_edit_tokens += len(ids)-prefix
        changed_bins.update(changed)
        changed_field_counts.update(fields)
        per_case.append({'id': key, 'instruction_first_word': tag, **dict(counters)})
    logs = [json.loads(line) for line in Path('artifacts/lossless-sft-pilot-v1/training.jsonl').read_text().splitlines()]
    final = logs[-1]
    if final['step'] != 500 or final['epoch'] != 1 or final['offset'] != 928:
        raise ValueError('Unexpected final pilot data cursor')
    findings = {
        'coverage': {k: dict(v.most_common()) for k, v in coverage.items()},
        'pilot_training_exposure': {'unique_examples': len(selected), 'optimizer_updates': 500,
            'total_example_presentations': 4000, 'full_epochs': 1, 'examples_in_second_epoch': 928,
            'mean_presentations_per_selected_example': 4000/len(selected),
            'note': 'Every selected example was seen once, and 928 were seen twice. Per-category second-pass counts were not reconstructed.'},
        'supervision_checks': {'cached_targets_verified': decoded, 'source_and_target_frame_checks_passed': True,
            'all_supervised_target_tokens': all_target_tokens, 'editing_supervised_target_tokens': all_edit_tokens,
            'source_CAD_tokens_inside_masked_prompt': source_tokens_masked,
            'prompt_labels': '-100 throughout prompt; full target through CAD-end and LM EOS supervised',
            'loss': 'Uniform target-token causal cross entropy, token-weighted across each effective batch of eight. No explicit weighting of changed positions.'},
        'numeric_supervision_by_instruction_first_word': {k: dict(v) for k, v in buckets.items()},
        'changed_numeric_bin_counts': dict(sorted(changed_bins.items(), key=lambda kv: int(kv[0]))),
        'changed_numeric_fields_by_command_and_1based_field': dict(changed_field_counts.most_common()),
        'limits': 'Counts of changed target numbers depend on deterministic command alignment. They measure label exposure, not gradient strength or conditional entropy. An explicit numeral need not specify a geometric parameter; no numeral does not imply an unanswerable instruction. First words are descriptive only. No inference, training or test cases.'}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'results.json', findings)
    write_json(OUT / 'per_case.json', per_case)
    write_json(OUT / 'manifest.json', {'pilot_manifest_sha256': file_hash(pilot_path), 'hashes': hashes,
        'analysis_files': {str(p): file_hash(p) for p in [Path(__file__), Path('scripts/analyze_edit_correctness.py')]}})
    print(json.dumps(findings, indent=2))


if __name__ == '__main__':
    main()
