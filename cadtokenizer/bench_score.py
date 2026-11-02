from __future__ import annotations

import collections
import json
from pathlib import Path

from .bench_data import canonical_sequence, code_hash, content_hash, load_manifest, prediction_paths, read_predictions
from .bench_judging import read_human_scores, read_vlm_scores, _write_jsonl
from .bench_metrics import all_sample_count_f1, cadfusion_f1, geometry_scores, mean, normalize_points
from .common import file_hash, require_container, write_json


METRIC_NAMES = ('F1-Skt', 'F1-Ext', 'CD', 'COV', 'JSD', 'MMD', 'IR', 'VLM', 'HE')


def display_metrics(raw):
    return {key: None if raw.get(key) is None else raw[key] * (100 if key in ('F1-Skt', 'F1-Ext', 'CD', 'COV', 'JSD', 'MMD', 'IR') else 1)
            for key in METRIC_NAMES}


def aggregate_task(cases, geometry, vlm_scores, human_scores):
    """Pure aggregation, including failures and explicit conditional denominators."""
    pairs = [(case['target_sequence'] or '', case['prediction_sequence']) for case in cases]
    f1 = cadfusion_f1(pairs)
    strict = all_sample_count_f1(pairs)
    matched = geometry['matched_valid']
    attempted = len(cases)
    invalid = sum(case['prediction_geometry']['status'] != 'valid' for case in cases)
    reference_invalid = sum(case['reference_geometry']['status'] != 'valid' for case in cases)
    vlm = [vlm_scores[case['sample_id']] for case in cases if case['sample_id'] in vlm_scores]
    human = [rank for case in cases for rank in human_scores.get(case['sample_id'], [])]
    raw = {'F1-Skt': f1['F1-Skt'], 'F1-Ext': f1['F1-Ext'],
           **{key: matched[key] for key in ('CD', 'COV', 'JSD', 'MMD')},
           'IR': invalid / attempted if attempted else None, 'VLM': mean(vlm), 'HE': mean(human)}
    return {'attempted': attempted, 'invalid_predictions': invalid, 'invalid_references': reference_invalid,
            'failure_categories': dict(collections.Counter(c['prediction_geometry']['status'] for c in cases if c['prediction_geometry']['status'] != 'valid')),
            'raw_metrics': raw, 'table_metrics': display_metrics(raw),
            'cadfusion_f1_details': f1, 'all_sample_count_f1': strict,
            'matched_valid_geometry': matched, 'all_valid_reference_geometry': geometry['all_valid_references'],
            'vlm_scored_samples': len(vlm), 'human_rank_count': len(human),
            'human_scored_samples': sum(bool(human_scores.get(c['sample_id'])) for c in cases),
            'exact_sequence_rate': mean(c['exact_sequence'] for c in cases),
            'copied_source_rate': mean(c['copied_source'] for c in cases if c['task'] == 'editing')}


def score_cases(manifest, predictions, settings, runner, point_loader, geometry_metric=geometry_scores,
                vlm_scores=None, human_scores=None):
    """Orchestration accepts test doubles; production uses real OpenCascade artifacts."""
    if set(predictions) != {sample['id'] for sample in manifest['samples']}:
        raise ValueError('Predictions must cover exactly the frozen manifest; resume missing inference first')
    cases = []
    pred_clouds, ref_clouds = {}, {}
    for index, sample in enumerate(manifest['samples']):
        key = sample['id']
        prediction = predictions[key]
        text = prediction.get('generated_sequence')
        if prediction['status'] == 'syntax_valid':
            pred_result = runner.reconstruct(text, key, 'prediction')
        else:
            pred_result = {'status': 'inference_error', 'error': prediction.get('error', 'Inference did not produce CAD')}
        ref_result = runner.reconstruct(sample.get('target_sequence'), key, 'reference')
        source_result = runner.reconstruct(sample['source_sequence'], key, 'source') if settings['render'] and sample['task'] == 'editing' else None
        case = {'sample_id': key, 'task': sample['task'], 'target_sequence': sample.get('target_sequence'),
                'prediction_sequence': text, 'prediction_geometry': pred_result, 'reference_geometry': ref_result,
                'source_geometry': source_result, 'exact_sequence': False, 'copied_source': False}
        if pred_result['status'] == 'valid':
            pred_clouds[key] = normalize_points(point_loader(pred_result), settings['normalization'])
        if ref_result['status'] == 'valid':
            ref_clouds[key] = normalize_points(point_loader(ref_result), settings['normalization'])
        def equals(sequence):
            if not text or not isinstance(sequence, str):
                return False
            try:
                return canonical_sequence(text) == canonical_sequence(sequence)
            except ValueError:
                return False
        case['exact_sequence'] = equals(sample.get('target_sequence'))
        if sample['task'] == 'editing':
            case['copied_source'] = equals(sample['source_sequence'])
        cases.append(case)
        if index % 10 == 0:
            print(json.dumps({'scored_cases': index + 1, 'total': len(manifest['samples'])}), flush=True)
    tasks = {}
    for task in ('generation', 'editing'):
        selected = [case for case in cases if case['task'] == task]
        if not selected:
            continue
        ids = {case['sample_id'] for case in selected}
        geometry = geometry_metric({key: value for key, value in pred_clouds.items() if key in ids},
                                   {key: value for key, value in ref_clouds.items() if key in ids}, settings)
        tasks[task] = aggregate_task(selected, geometry, vlm_scores or {}, human_scores or {})
        for case in selected:
            case['paired_cd_raw'] = geometry.get('paired_cd_by_id', {}).get(case['sample_id'])
    return tasks, cases


def comparison_markdown(report, references):
    lines = ['# CAD benchmark comparison', '',
             '**Comparison status: provisional; this is not a verified reproduction of the paper test set.**', '',
             f'Paper reference: {references["source"]}', '',
             f'Manifest SHA-256: `{report["manifest_sha256"]}`', '',
             'Values use the paper table units. Delta is ours minus the published value; a small delta alone does not establish reproduction.', '',
             '| Task | Metric | Paper | Ours | Delta | Preferred |',
             '| --- | --- | ---: | ---: | ---: | --- |']
    for task, result in report['tasks'].items():
        for name in METRIC_NAMES:
            value, reference = result['table_metrics'][name], references[task][name]
            ours = 'not scored' if value is None else f'{value:.4f}'
            delta = '—' if value is None else f'{value-reference:+.4f}'
            lines.append(f'| {task} | {name} | {reference:g} | {ours} | {delta} | {references["direction"][name]} |')
    lines += ['', '## Denominators', '',
              '| Task | All cases | Invalid predictions | Invalid references | Valid paired geometry | VLM scored | Human ranks |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for task, result in report['tasks'].items():
        lines.append(f'| {task} | {result["attempted"]} | {result["invalid_predictions"]} | {result["invalid_references"]} | {result["matched_valid_geometry"]["paired_count"]} | {result["vlm_scored_samples"]} | {result["human_rank_count"]} |')
    lines += ['', '## Limits on comparability', '']
    lines += [f'- {note}' for note in report['comparability_notes']]
    lines += ['', 'F1 compatibility uses upstream count-only matching and conditional type averages. '
              'The JSON report also includes all-sample count F1, full-reference distribution metrics, exact-sequence and copied-source rates. '
              'Missing judgments remain null; they are never replaced by zero.', '']
    return '\n'.join(lines)


def benchmark_score(config, settings, manifest_path=None, predictions_path=None, vlm_path=None, human_path=None,
                    runner_override=None):
    require_container()
    from .bench_runtime import KernelRunner, evaluation_doctor
    import numpy as np
    root = Path(settings['output'])
    manifest_path = Path(manifest_path or root / 'manifest.json')
    manifest = load_manifest(manifest_path)
    prediction_path, metadata_path = prediction_paths(settings, predictions_path)
    metadata = json.loads(metadata_path.read_text())
    signature = metadata['signature']
    manifest_hash, predictions_hash = file_hash(manifest_path), file_hash(prediction_path)
    if metadata.get('schema') != 'cad-predictions-v1' or signature['manifest_sha256'] != manifest_hash:
        raise ValueError('Prediction metadata does not match the frozen benchmark')
    if metadata.get('signature_sha256') != content_hash(signature):
        raise ValueError('Prediction metadata fingerprint is invalid')
    for key in ('seed', 'max_new_tokens'):
        if key in signature and signature[key] != settings[key]:
            raise ValueError(f'Prediction and evaluation settings differ: {key}')
    predictions = read_predictions(prediction_path, manifest)
    if any(row.get('manifest_sha256') != manifest_hash or row.get('run_signature_sha256') != metadata['signature_sha256'] for row in predictions.values()):
        raise ValueError('Prediction rows do not match their run metadata')
    if len(predictions) != len(manifest['samples']):
        raise ValueError('Incomplete predictions; use benchmark-predict --resume-predictions before scoring')
    vlm, vlm_metadata = read_vlm_scores(vlm_path, manifest_hash, predictions_hash, manifest, content_hash(settings))
    human, human_metadata = read_human_scores(human_path, manifest_hash, predictions_hash, manifest, content_hash(settings))
    environment = evaluation_doctor()
    runner = runner_override if runner_override is not None else KernelRunner(settings)
    def load_points(result):
        points = np.load(result['points'], allow_pickle=False)
        if points.shape != (settings['point_count'], 3) or not np.isfinite(points).all():
            raise RuntimeError('Cached point-cloud artifact is corrupt or has incompatible sampling settings')
        return points
    tasks, cases = score_cases(manifest, predictions, settings, runner, load_points, vlm_scores=vlm, human_scores=human)
    per_sample_path = root / 'per_sample.jsonl'
    for case in cases:
        case.pop('target_sequence', None)
        case.pop('prediction_sequence', None)
        case['manifest_sha256'] = manifest_hash
    _write_jsonl(per_sample_path, cases)
    notes = list(manifest.get('comparability_notes', [])) + [
        'The exact paper test manifest/additional annotations are not verified; scores cannot establish an exact reproduction.',
        'CadQuery/OCP replaces upstream pythonocc geometry repair; kernel validity can differ.',
        'The table geometry metrics use matching valid prediction/reference pairs, following the upstream filtering convention; IR always includes all frozen cases.',
        'Surface sampling is independently seeded by case and role; the upstream release does not define identical random draws.',
        'Independent max-absolute normalization follows the reference utility but can hide scale differences.',
        'VLM and human judgments depend on the judging protocol and candidate baselines; coverage is reported separately.',
    ]
    report = {'schema': 'cad-benchmark-results-v1', 'manifest_sha256': manifest_hash, 'predictions_sha256': predictions_hash,
              'prediction_run_signature': metadata['signature_sha256'], 'prediction_metadata': metadata,
              'evaluation_code_sha256': code_hash(), 'settings': settings, 'environment': environment,
              'geometry_cache_provenance': getattr(runner, 'provenance', None),
              'metric_reference': {'repository': 'https://github.com/microsoft/CADFusion',
                                   'inspected_commit': '5bef5bb632b66ede9ca3d438b68adca0460e27f9',
                                   'files': ['src/test/f1_eval.py', 'src/test/chamfer_dist.py', 'src/test/dist_eval.py']},
              'paper_test_set_verified': False, 'comparison_status': 'provisional', 'comparability_notes': notes,
              'tasks': tasks, 'vlm_judging': vlm_metadata, 'human_judging': human_metadata,
              'artifacts': {'per_sample': str(per_sample_path), 'manifest': str(manifest_path), 'predictions': str(prediction_path)}}
    write_json(root / 'metrics.json', report)
    references = json.loads(Path('configs/paper_reference.json').read_text())
    (root / 'comparison.md').write_text(comparison_markdown(report, references))
    print(json.dumps({'comparison': str(root / 'comparison.md'), 'metrics': {k: v['table_metrics'] for k, v in tasks.items()}}, indent=2), flush=True)
    return report
