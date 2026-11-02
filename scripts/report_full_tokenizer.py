"""Report completed full-data milestones; pure JSON analysis, safe on the host."""
import argparse
from pathlib import Path
import statistics

from cadtokenizer.common import read_config, write_json
from cadtokenizer.tokenizer_learning_curve import paired_comparison


def volume_error(row):
    truth = row['geometry']['reference']['volume']
    return abs(row['geometry']['reconstruction']['volume']-truth)/truth


def summary(rows):
    valid = [r for r in rows if r['paired_valid']]
    references = sum(r['known_valid_reference'] for r in rows)
    return {'cases': len(rows), 'known_valid_references': references, 'valid': len(valid),
            'valid_rate': len(valid)/references if references else None,
            'mean_chamfer_valid_pairs': statistics.mean(r['chamfer'] for r in valid) if valid else None,
            'median_relative_volume_error': statistics.median(volume_error(r) for r in valid) if valid else None}


def report(root):
    source = read_config(root/'full-input-manifest.json')
    anchors = set(source['anchor_validation_ids'])
    training = root/'training'
    baseline = read_config(training/'baseline/per_case.json')
    baseline_map = {r['id']:r for r in baseline}
    base_summary = summary(baseline)
    anchor_base = summary([r for r in baseline if r['id'] in anchors])
    points = []
    for step in read_config(root/'training-settings.json')['evaluation_updates']:
        directory = training/'curve'/f'evaluation-{step}'
        if not (directory/'paired_comparison.json').exists():
            continue
        rows = read_config(directory/'per_case.json')
        pairs = paired_comparison(baseline, rows)
        shared = [r for r in rows if r['paired_valid'] and baseline_map[r['id']]['paired_valid']]
        volume = {'cases':len(shared),
                  'baseline_median_relative_error':statistics.median(volume_error(baseline_map[r['id']]) for r in shared) if shared else None,
                  'candidate_median_relative_error':statistics.median(volume_error(r) for r in shared) if shared else None,
                  'candidate_lower_error_cases':sum(volume_error(r)<volume_error(baseline_map[r['id']]) for r in shared)}
        points.append({'step':step,'full_panel':summary(rows),'paired_geometry':pairs,'paired_volume':volume,
                       'historical_panel':summary([r for r in rows if r['id'] in anchors])})
    output={'baseline':base_summary,'historical_baseline':anchor_base,'points':points,
            'job_state':read_config(root/'state.json'),'training_state':read_config(training/'state.json')}
    write_json(root/'progress.json',output)
    lines=['# Full-data CE tokenizer progress','',
           'Fixed full training corpus and 512-case validation panel. Original pair boundaries are supplied. Rates are conditional on kernel-valid originals. No end-to-end LLM quality claim is made.','',
           '| Model / updates | Valid / known-valid | Own-valid mean CD | Paired CD: reference / candidate | Paired median volume error: reference / candidate |',
           '| --- | ---: | ---: | --- | --- |',
           f'| Existing tokenizer | {base_summary["valid"]}/{base_summary["known_valid_references"]} | {base_summary["mean_chamfer_valid_pairs"]} | — | — |']
    for point in points:
        s,p,v=point['full_panel'],point['paired_geometry'],point['paired_volume']
        lines.append(f'| {point["step"]} | {s["valid"]}/{s["known_valid_references"]} | {s["mean_chamfer_valid_pairs"]} | {p["baseline_mean_chamfer_common"]} / {p["candidate_mean_chamfer_common"]} (n={p["common_valid_cases"]}) | {v["baseline_median_relative_error"]} / {v["candidate_median_relative_error"]} |')
    lines+=['','## Historical 128-case anchor','',
            f'Existing tokenizer: {anchor_base["valid"]}/{anchor_base["known_valid_references"]} valid.','']
    for point in points:
        a=point['historical_panel'];lines.append(f'- {point["step"]} updates: {a["valid"]}/{a["known_valid_references"]} valid; mean CD {a["mean_chamfer_valid_pairs"]}.')
    lines+=['','Paired geometry/volume statistics use identical IDs valid in both models. Different checkpoints have different shared subsets, so their paired means are not directly interchangeable. No checkpoint is promoted automatically.']
    (root/'progress.md').write_text('\n'.join(lines)+'\n')
    # The older generic learning-curve formatter describes the original subset.
    # Correct descriptive labels only; never alter frozen cases or model inputs.
    generic = training/'comparison.md'
    if generic.exists():
        text = generic.read_text()
        text = text.replace('Fixed original 512 training geometries and 128 validation geometries.',
                            f'Full tokenizer pool: {len(source["train"]):,} training geometries and {len(source["validation"])} validation geometries.')
        text = text.replace('512 training geometries are a subset of the full-data baseline corpus; this is not a matched full-data reproduction.',
                            'The full prepared tokenizer training pool is used. This does not verify the paper\'s original split or constitute an exact reproduction.')
        generic.write_text(text)
    print({'baseline':base_summary,'points':[(p['step'],p['full_panel']['valid'],p['historical_panel']['valid']) for p in points], 'state':output['training_state']})
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('artifacts/tokenizer-full-ce-v1'))
    report(parser.parse_args().root)
