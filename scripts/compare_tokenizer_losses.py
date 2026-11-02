"""Pure-JSON paired analysis of completed tokenizer loss controls; no model imports.

From the repository root: python -m scripts.compare_tokenizer_losses
"""
import argparse
import difflib
import json
from pathlib import Path
import statistics

from cadtokenizer.common import file_hash, read_config, write_json
from cadtokenizer.tokenizer_learning_curve import paired_comparison


def compare(ce_root, emd_root, output):
    a, b = [read_config(root / 'manifest.json') for root in (ce_root, emd_root)]
    assert a['train'] == b['train'] and a['validation'] == b['validation'], 'Geometry IDs differ'
    for field in ('source_database_sha256', 'source_manifest_sha256', 'baseline_checkpoint_sha256', 'environment'):
        assert a['signature'][field] == b['signature'][field], field
    physics = ('vq.py', 'representation.py', 'geometry_worker.py', 'bench_runtime.py', 'bench_metrics.py', 'tokenizer_experiment.py')
    for name in physics:
        assert file_hash(ce_root / 'source' / name) == file_hash(emd_root / 'source' / name), name
    configs = [read_config(root / 'curve' / 'config.json') for root in (ce_root, emd_root)]
    assert configs[0]['vq']['reconstruction_loss'] == 'cross_entropy' and configs[1]['vq']['reconstruction_loss'] == 'emd'
    for cfg in configs:
        cfg['data'].pop('database'); cfg['vq'].pop('output'); cfg['vq'].pop('reconstruction_loss')
    assert configs[0] == configs[1], 'Other training settings differ'
    protocols = [dict(manifest['signature']['settings']) for manifest in (a, b)]
    for protocol in protocols:
        for key in ('output', 'run_loop_if_improved', 'reconstruction_loss'):
            protocol.pop(key, None)
    assert protocols[0] == protocols[1], 'Other update/evaluation settings differ'
    old = (ce_root / 'source' / 'tokenizer_learning_curve.py').read_text().splitlines(True)
    new = (emd_root / 'source' / 'tokenizer_learning_curve.py').read_text().splitlines(True)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'protocol.diff').write_text(''.join(difflib.unified_diff(old, new, fromfile='cross-entropy protocol', tofile='EMD protocol')))
    points = []
    lines = ['# Matched EMD versus cross-entropy control', '',
             'Same training/validation geometries, environment, model/geometry implementation, optimizer, regularization, deterministic sampler and update budget. The runner diff records the objective selection and report-label support added for EMD.', '',
             '| Updates | CE valid | EMD valid | Shared valid | CE CD, shared cases | EMD CD, shared cases |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for step in protocols[0]['evaluation_updates']:
        dirs = [root / 'curve' / f'evaluation-{step}' for root in (ce_root, emd_root)]
        summaries = [read_config(path / 'report.json')['summary'] for path in dirs]
        rows = [read_config(path / 'per_case.json') for path in dirs]
        comparison = paired_comparison(*rows)
        maps = [{row['id']: row for row in group} for group in rows]
        shared = [key for key in maps[0] if maps[0][key]['paired_valid'] and maps[1][key]['paired_valid']]
        volumes = [[], []]
        identical_reference_points = 0
        for key in maps[0]:
            if maps[0][key]['known_valid_reference']:
                refs = [mapping[key]['geometry']['reference'] for mapping in maps]
                identical_reference_points += file_hash(refs[0]['points']) == file_hash(refs[1]['points'])
        for key in shared:
            for index in (0, 1):
                row = maps[index][key]
                truth = row['geometry']['reference']['volume']
                volumes[index].append(abs(row['geometry']['reconstruction']['volume'] - truth) / truth)
        volume = {'common_valid_cases': len(shared),
                  'ce_median_relative_error': statistics.median(volumes[0]) if shared else None,
                  'emd_median_relative_error': statistics.median(volumes[1]) if shared else None,
                  'ce_mean_relative_error': statistics.mean(volumes[0]) if shared else None,
                  'emd_mean_relative_error': statistics.mean(volumes[1]) if shared else None,
                  'emd_lower_volume_error_cases': sum(y < x for x, y in zip(*volumes))}
        point = {'step': step, 'ce_summary': summaries[0], 'emd_summary': summaries[1],
                 'paired_ce_as_baseline_emd_as_candidate': comparison, 'paired_volume': volume,
                 'identical_valid_reference_point_files': identical_reference_points,
                 'report_sha256': {'cross_entropy': file_hash(dirs[0] / 'report.json'), 'emd': file_hash(dirs[1] / 'report.json')}}
        points.append(point)
        lines.append(f'| {step} | {comparison["baseline_valid"]}/{summaries[0]["known_valid_references"]} | {comparison["candidate_valid"]}/{summaries[1]["known_valid_references"]} | {len(shared)} | {comparison["baseline_mean_chamfer_common"]} | {comparison["candidate_mean_chamfer_common"]} |')
    lines += ['', 'CD is compared on identical case IDs valid in both losses. Both remain separate from the unchanged full-data tokenizer baseline (87/112 valid). Volume comparisons use those same shared IDs; detailed values and reference point-file checks are in loss_comparison.json.', '',
              'This is one seed on 512 training shapes, with equal update budgets rather than proven convergence. No full-corpus loss superiority or paper reproduction claim is made.']
    write_json(output / 'loss_comparison.json', {'matched_training_inputs_verified': True, 'model_geometry_source_hashes_match': True,
                                                'points': points})
    (output / 'loss_comparison.md').write_text('\n'.join(lines) + '\n')
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ce', type=Path, default=Path('artifacts/tokenizer-learning-curve-v1'))
    parser.add_argument('--emd', type=Path, default=Path('artifacts/tokenizer-learning-curve-emd-v1'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    points = compare(args.ce, args.emd, args.output or args.emd)
    for point in points:
        print(json.dumps({'step': point['step'], 'paired': point['paired_ce_as_baseline_emd_as_candidate'], 'volume': point['paired_volume']}))


if __name__ == '__main__':
    main()
