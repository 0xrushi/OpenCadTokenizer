"""How far from correct is the decoder where a reconstruction fails? Diagnostic only; trains nothing.

For each known-valid case whose reconstruction was rejected by the kernel, the decoder is teacher-forced on the
original pair tokens and, at every position where its argmax is wrong, the rank and probability of the correct
token are recorded. The first wrong position is where a free-running decoder first deviates, so its rank bounds
what n-best branching there could recover. Scale and origin are skipped: the literal trailer replaces them.
"""
import json
from pathlib import Path
import sqlite3

from .common import file_hash, read_config, require_container, write_json

TRAILER_FIELDS = ('scale', 'ox', 'oy')
FIELDS = ['boolean', 'lower', 'upper', 'tx', 'ty', 'tz', 'xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz',
          'scale', 'ox', 'oy', 'extrusion_end']


def labels_for(tokens, length):
    boundary = next(i for i, t in enumerate(tokens) if t in (8, 9, 10))
    labels = ['sketch'] * length
    labels[boundary:boundary + len(FIELDS)] = FIELDS
    return labels


def recoverable(ranks, k):
    return bool(ranks) and all(r <= k for r in ranks)


def run_topk_check(settings):
    require_container()
    import torch
    from .vq import collate_pairs, load_vq, to_device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = load_vq(settings['checkpoint'], device)
    cases = json.loads(Path(settings['per_case']).read_text())
    failing = [c for c in cases if c['known_valid_reference']
               and (c['geometry'].get('reconstruction') or {}).get('status') != 'valid']
    small_limit = settings.get('small_slots', 12)
    rows = []
    with sqlite3.connect(settings['database']) as db:
        for case in failing:
            pairs = db.execute('SELECT tokens,groups FROM pairs WHERE geometry_id=? ORDER BY ordinal', (case['id'],)).fetchall()
            errors = []
            for pair_index, (tokens, groups) in enumerate(pairs):
                tokens, groups = json.loads(tokens), json.loads(groups)
                batch = to_device(collate_pairs([{'tokens': tokens, 'groups': groups}]), device)
                with torch.no_grad():
                    pooled, mask = model.pool(batch['tokens'], batch['groups'])
                    memory, _, _ = model.quantize(pooled, mask)
                    probs = model.decode(batch['decoder_input'], memory, mask)[0].float().softmax(-1)
                target = batch['targets'][0].tolist()
                labels = labels_for(tokens, len(target))
                for position, actual in enumerate(target):
                    if actual == 0 or labels[position] in TRAILER_FIELDS:
                        continue
                    predicted = int(probs[position].argmax())
                    if predicted != actual:
                        rank = int((probs[position] > probs[position, actual]).sum()) + 1
                        errors.append({'pair': pair_index, 'position': position, 'field': labels[position], 'rank': rank,
                                       'p_predicted': round(float(probs[position, predicted]), 4),
                                       'p_correct': round(float(probs[position, actual]), 4)})
            rows.append({'id': case['id'], 'small': case['features']['max_curve_primitives'] <= small_limit
                         if 'max_curve_primitives' in case['features'] else None,
                         'errors': errors})
    summary = {'known_valid_failures': len(rows), 'small_slot_failures': sum(bool(r['small']) for r in rows),
               'no_decoder_error_outside_trailer': sum(not r['errors'] for r in rows)}
    for group, subset in (('all', rows), ('small_slots', [r for r in rows if r['small']])):
        entry = {'cases': len(subset), 'with_errors': sum(bool(r['errors']) for r in subset)}
        for k in (1, 2, 3, 5, 10):
            entry[f'all_errors_in_top_{k}'] = sum(recoverable([e['rank'] for e in r['errors']], k) for r in subset)
            entry[f'first_error_in_top_{k}'] = sum(bool(r['errors']) and r['errors'][0]['rank'] <= k for r in subset)
        entry['error_count_histogram'] = {str(n): sum(len(r['errors']) == n for r in subset) for n in range(0, 6)}
        entry['more_than_5_errors'] = sum(len(r['errors']) > 5 for r in subset)
        summary[group] = entry
    errors = [e for r in rows for e in r['errors']]
    if errors:
        summary['error_positions'] = len(errors)
        summary['rank_of_correct_token'] = {str(k): sum(e['rank'] <= k for e in errors) for k in (1, 2, 3, 5, 10)}
        summary['predicted_probability_below'] = {str(t): sum(e['p_predicted'] < t for e in errors) for t in (0.5, 0.8, 0.9, 0.99)}
        by_field = {}
        for e in errors:
            by_field[e['field']] = by_field.get(e['field'], 0) + 1
        summary['errors_by_field'] = by_field
    out = Path(settings['output'])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'report.json', {'summary': summary, 'cases': rows, 'checkpoint_sha256': file_hash(settings['checkpoint']),
                                     'per_case_sha256': file_hash(settings['per_case']), 'note': 'diagnostic only; teacher-forced'})
    print(json.dumps(summary, indent=2), flush=True)
    return summary
