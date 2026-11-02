"""Pure-Python analysis of saved CAD edits; performs no model execution."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics

from cadtokenizer.common import file_hash, write_json

ROOT = Path('artifacts/lossless-editing-validation-v1')
OUTPUT = Path('artifacts/lossless-edit-correctness-v1')


def units(sequence):
    # A curve with its coordinates or an extrusion with its parameters is one
    # command. Structural markers are separate commands. Whitespace is canonical.
    return (sequence or '').split()


def align(source, target):
    """Minimum unit-cost edit path; ties: match/substitute, delete, insert."""
    n, m = len(source), len(target)
    dp = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i-1][j-1] + (source[i-1] != target[j-1]),
                           dp[i-1][j] + 1, dp[i][j-1] + 1)
    events, preserved = [], set()
    i, j = n, m
    while i or j:
        if i and j and dp[i][j] == dp[i-1][j-1] + (source[i-1] != target[j-1]):
            if source[i-1] == target[j-1]:
                preserved.add(i-1)
            else:
                events.append(('replace', i-1, target[j-1]))
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i-1][j] + 1:
            events.append(('delete', i-1, source[i-1]))
            i -= 1
        else:
            events.append(('insert', i, target[j-1]))
            j -= 1
    return dp[n][m], Counter(events), preserved


def safe_mean(values):
    return statistics.mean(values) if values else None


def command_types(sequence):
    return [value.split(',')[0] for value in sequence]


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    arms = json.loads((ROOT / 'per_case.json').read_text())
    cases = manifest['cases']
    if len(cases) != 500:
        raise ValueError('Expected 500 frozen cases')
    for arm in arms.values():
        if [r['id'] for r in arm] != [c['example_id'] for c in cases]:
            raise ValueError('Case alignment mismatch')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, case in enumerate(cases):
        source, target = units(case['source']), units(case['target'])
        source_distance, wanted, unchanged = align(source, target)
        if not source_distance:
            raise ValueError('Expected all targets to differ from source')
        row = {'id': case['example_id'], 'instruction': case['instruction'],
               'component': case['components'], 'required_command_edits': source_distance,
               'source_commands': len(source), 'required_events': [list(k) + [v] for k, v in sorted(wanted.items())],
               'arms': {}}
        for name in ('lossless', 'codebook', 'source_copy'):
            prediction = units(arms[name][i]['sequence'])
            _, actual, retained = align(source, prediction)
            error, _, _ = align(target, prediction)
            hits = sum((wanted & actual).values())
            extras = sum((actual - wanted).values())
            missed = sum((wanted - actual).values())
            row['arms'][name] = {
                'exact_target': prediction == target, 'copies_source': prediction == source,
                'exact_command_type_sequence': command_types(prediction) == command_types(target),
                'valid_solid': arms[name][i]['prediction_geometry']['status'] == 'valid',
                'target_command_distance': error,
                'closer_than_source_by_command_distance': error < source_distance,
                'worse_than_source_by_command_distance': error > source_distance,
                'required_edit_hits': hits, 'missed_required_events': missed,
                'extra_events': extras, 'predicted_events': sum(actual.values()),
                'edit_precision': hits / sum(actual.values()) if actual else 0.,
                'edit_recall': hits / sum(wanted.values()),
                'unchanged_source_commands': len(unchanged),
                'preserved_unchanged_commands': len(unchanged & retained),
            }
        rows.append(row)
    summaries = {}
    for name in ('lossless', 'codebook', 'source_copy'):
        rr = [r['arms'][name] for r in rows]
        hits = sum(r['required_edit_hits'] for r in rr)
        predicted = sum(r['predicted_events'] for r in rr)
        required = sum(r['required_command_edits'] for r in rows)
        summaries[name] = {
            'cases': len(rr), 'exact_targets': sum(r['exact_target'] for r in rr),
            'exact_targets_with_valid_solid': sum(r['exact_target'] and r['valid_solid'] for r in rr),
            'exact_command_type_sequence': sum(r['exact_command_type_sequence'] for r in rr),
            'correct_command_types_but_wrong_numeric_fields': sum(r['exact_command_type_sequence'] and not r['exact_target'] for r in rr),
            'unchanged_source_outputs': sum(r['copies_source'] for r in rr),
            'closer_to_target_than_source_by_command_distance': sum(r['closer_than_source_by_command_distance'] for r in rr),
            'worse_than_source_by_command_distance': sum(r['worse_than_source_by_command_distance'] for r in rr),
            'equal_to_source_by_command_distance': sum(not r['closer_than_source_by_command_distance'] and not r['worse_than_source_by_command_distance'] for r in rr),
            'mean_target_command_distance': safe_mean([r['target_command_distance'] for r in rr]),
            'median_target_command_distance': statistics.median(r['target_command_distance'] for r in rr),
            'micro_edit_precision': hits / predicted if predicted else 0.,
            'micro_edit_recall': hits / required,
            'macro_edit_precision': safe_mean([r['edit_precision'] for r in rr]),
            'macro_edit_recall': safe_mean([r['edit_recall'] for r in rr]),
            'cases_with_no_required_edit_event_recovered': sum(r['required_edit_hits'] == 0 for r in rr),
            'cases_with_some_required_edits_but_not_exact': sum(r['required_edit_hits'] > 0 and not r['exact_target'] for r in rr),
            'micro_unchanged_command_preservation': sum(r['preserved_unchanged_commands'] for r in rr) / sum(r['unchanged_source_commands'] for r in rr),
        }
    groups = defaultdict(list)
    for r in rows:
        verb = re.match(r'\w+', r['instruction']).group().lower()
        groups[verb].append(r)
    by_verb = {verb: {'cases': len(group), 'lossless_exact': sum(r['arms']['lossless']['exact_target'] for r in group),
                      'lossless_correct_command_types_but_wrong_numbers': sum(r['arms']['lossless']['exact_command_type_sequence'] and not r['arms']['lossless']['exact_target'] for r in group),
                      'lossless_copies': sum(r['arms']['lossless']['copies_source'] for r in group),
                      'lossless_closer_by_command_distance': sum(r['arms']['lossless']['closer_than_source_by_command_distance'] for r in group)}
               for verb, group in sorted(groups.items(), key=lambda kv: -len(kv[1]))}
    result = {'summaries': summaries, 'instruction_first_word_groups': by_verb,
              'source_target_mean_command_distance': safe_mean([r['required_command_edits'] for r in rows]),
              'source_target_median_command_distance': statistics.median(r['required_command_edits'] for r in rows),
              'limits': 'Reference-program fidelity proxies, not a semantic instruction judge. One command includes all numeric fields; markers count separately. Exact anchored edit matching is sensitive to equivalent programs, repeated commands and alignment ties. A matching command edit does not prove a correct solid or instruction. First-word groups are descriptive, not validated edit categories. 500 prompts span 109 components; no training or inference performed.'}
    write_json(OUTPUT / 'results.json', result)
    write_json(OUTPUT / 'per_case.json', rows)
    write_json(OUTPUT / 'manifest.json', {'inputs': {str(p): file_hash(p) for p in [ROOT / 'manifest.json', ROOT / 'per_case.json', Path(__file__)]},
                                        'analysis': 'command_levenshtein_source_anchored_v1'})
    # Deterministic review packet, containing easy successes and distinct failures.
    pools = {
        'Exact target': [i for i, r in enumerate(rows) if r['arms']['lossless']['exact_target']],
        'Unchanged source': [i for i, r in enumerate(rows) if r['arms']['lossless']['copies_source']],
        'Partial required edits': [i for i, r in enumerate(rows) if r['arms']['lossless']['required_edit_hits'] and not r['arms']['lossless']['exact_target']],
        'No required edit recovered': [i for i, r in enumerate(rows) if not r['arms']['lossless']['required_edit_hits'] and not r['arms']['lossless']['copies_source']],
    }
    packet = ['# Saved edit review packet', '', 'Six deterministic examples per group, when available. These are reference comparisons, not blinded human judgments.', '']
    for label, indices in pools.items():
        indices.sort(key=lambda i: hashlib.sha256(rows[i]['id'].encode()).hexdigest())
        packet += ['## ' + label, '']
        for i in indices[:6]:
            case, row = cases[i], rows[i]
            packet += ['### ' + row['id'], '', case['instruction'], '',
                       'Scores: `' + json.dumps(row['arms']['lossless']) + '`', '',
                       '**Source**', '', '```text', case['source'], '```', '',
                       '**Target**', '', '```text', case['target'], '```', '',
                       '**Lossless output**', '', '```text', arms['lossless'][i]['sequence'] or '(missing)', '```', '']
    (OUTPUT / 'review.md').write_text('\n'.join(packet))
    (OUTPUT / 'findings.md').write_text('# Edit correctness analysis\n\n' + result['limits'] + '\n\n```json\n' + json.dumps(result, indent=2) + '\n```\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
