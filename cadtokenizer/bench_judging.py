from __future__ import annotations

import collections
import html
import json
import math
import os
from pathlib import Path
import random

from .bench_data import load_manifest
from .bench_metrics import mean
from .common import file_hash, require_container, write_json


def _bound_file(path, manifest_hash, predictions_hash, settings_hash=None):
    value = json.loads(Path(path).read_text())
    if value.get('manifest_sha256') != manifest_hash or value.get('predictions_sha256') != predictions_hash:
        raise ValueError('Judging scores belong to another manifest or prediction file')
    if settings_hash is not None and value.get('evaluation_settings_sha256') != settings_hash:
        raise ValueError('Judging scores use another geometry/rendering protocol')
    return value


def read_vlm_scores(path, manifest_hash, predictions_hash, manifest, settings_hash=None):
    if not path:
        return {}, {'status': 'not_scored', 'reason': 'No external VLM judgments supplied'}
    data = _bound_file(path, manifest_hash, predictions_hash, settings_hash)
    if not data.get('judge_model') or data['judge_model'] == 'FILL_ME':
        raise ValueError('VLM score file must identify the actual judging model/version')
    known, seen, scores = {s['id'] for s in manifest['samples']}, set(), {}
    for row in data['records']:
        key, score = row['sample_id'], row.get('score')
        if key not in known or key in seen:
            raise ValueError('Unknown or duplicate VLM score sample ID')
        seen.add(key)
        if score is None:
            continue
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 10:
            raise ValueError('VLM scores must be finite numbers in [0,10]')
        scores[key] = float(score)
    return scores, {'status': 'scored' if scores else 'not_scored', 'judge_model': data['judge_model'],
                    'score_file_sha256': file_hash(path), 'scored_samples': len(scores)}


def read_human_scores(path, manifest_hash, predictions_hash, manifest, settings_hash=None):
    if not path:
        return {}, {'status': 'not_scored', 'reason': 'No blinded human rankings supplied'}
    data = _bound_file(path, manifest_hash, predictions_hash, settings_hash)
    key_path = Path(path).parent / data['key_file']
    if file_hash(key_path) != data['key_sha256']:
        raise ValueError('Blinding key fingerprint does not match the score template')
    key = _bound_file(key_path, manifest_hash, predictions_hash, settings_hash)
    known, seen, scores = {s['id'] for s in manifest['samples']}, set(), collections.defaultdict(list)
    candidate_counts = collections.Counter()
    for row in data['records']:
        sample, judge, ranking = row['sample_id'], row['judge_id'], row.get('ranking')
        if sample not in known or sample not in key['labels'] or not isinstance(judge, str) or not judge.strip() or (sample, judge) in seen:
            raise ValueError('Invalid or duplicate human judgment identity')
        seen.add((sample, judge))
        if ranking is None:
            continue
        labels = key['labels'][sample]
        if not isinstance(ranking, list) or len(ranking) != len(labels) or set(ranking) != set(labels):
            raise ValueError('Human ranking must be a permutation of the blinded candidate labels')
        our_label = next(label for label, name in labels.items() if name == 'ours')
        candidate_counts[len(labels)] += 1
        # Table-1 rank values concern three-way comparisons. Other packet sizes
        # remain recorded but are not mixed into the comparable HE column.
        if len(labels) == 3:
            scores[sample].append(ranking.index(our_label) + 1)
    return dict(scores), {'status': 'scored' if scores else 'not_scored',
                          'candidate_count_histogram': dict(candidate_counts), 'score_file_sha256': file_hash(path),
                          'scored_samples': len(scores), 'judges': sorted({judge for _, judge in seen}),
                          'protocol': 'Blinded rankings; only three-candidate packets enter HE'}


def export_judging(settings, manifest_path=None, baseline_reports=None, directory=None):
    require_container()
    root = Path(settings['output'])
    manifest_path = Path(manifest_path or root / 'manifest.json')
    manifest = load_manifest(manifest_path)
    metrics = json.loads((root / 'metrics.json').read_text())
    if metrics['manifest_sha256'] != file_hash(manifest_path):
        raise ValueError('Scoring report belongs to another benchmark manifest')
    per_sample = {r['sample_id']: r for r in _read_jsonl(metrics['artifacts']['per_sample'])}
    from .bench_data import content_hash
    if content_hash(metrics['settings']) != content_hash(settings):
        raise ValueError('Current judging settings differ from the completed scoring run')
    headers = {'manifest_sha256': metrics['manifest_sha256'], 'predictions_sha256': metrics['predictions_sha256'],
               'evaluation_settings_sha256': content_hash(settings)}
    baselines = {}
    for specification in baseline_reports or []:
        name, separator, path = specification.partition('=')
        if not separator or not name or name == 'ours' or name in baselines:
            raise ValueError('--baseline-report must be a unique NAME=metrics.json (NAME cannot be ours)')
        report = json.loads(Path(path).read_text())
        if report['manifest_sha256'] != headers['manifest_sha256']:
            raise ValueError('Human comparison baselines must use the same frozen manifest')
        for setting in ('render_axis_limit', 'mesh_tolerance', 'mesh_angular_tolerance'):
            if report['settings'][setting] != settings[setting]:
                raise ValueError(f'Baseline rendering protocol differs: {setting}')
        baselines[name] = {r['sample_id']: r for r in _read_jsonl(report['artifacts']['per_sample'])}
    directory = Path(directory or root / 'judging')
    if directory.exists():
        raise FileExistsError('Judging export exists; use --directory to preserve ratings and create another export')
    directory.mkdir(parents=True)
    vlm_records, vlm_template = [], []
    for sample in manifest['samples']:
        row = per_sample[sample['id']]
        images = {'prediction': row['prediction_geometry'].get('image')}
        if sample['task'] == 'editing':
            images['original'] = (row.get('source_geometry') or {}).get('image')
            prompt = ('Rate how faithfully the edited CAD geometry performs the requested change to the original geometry. '
                      'Ignore color and rendering style. Return a score from 0 to 10. Instruction: ' + sample['instruction'])
        else:
            prompt = ('Rate how faithfully the CAD geometry matches the description. Ignore color and rendering style. '
                      'Return a score from 0 to 10. Description: ' + sample['instruction'])
        vlm_records.append({'sample_id': sample['id'], 'task': sample['task'], 'prompt': prompt, 'images': images,
                            'prediction_status': row['prediction_geometry']['status'],
                            'ready': all(images.values()), 'note': 'No network request is made by this export'})
        vlm_template.append({'sample_id': sample['id'], 'score': None})
    _write_jsonl(directory / 'vlm_requests.jsonl', vlm_records)
    write_json(directory / 'vlm_scores.template.json', {**headers, 'judge_model': 'FILL_ME',
                                                      'records': vlm_template})
    available = []
    for task, count in [('editing', 40), ('generation', 10)]:
        cases = [s for s in manifest['samples'] if s['task'] == task]
        random.Random(settings['seed']).shuffle(cases)
        for sample in cases:
            if sum(sample['id'] in rows for rows in baselines.values()) >= 2:
                available.append(sample)
                count -= 1
                if count == 0:
                    break
    labels, packets, answer_rows = {}, [], []
    fragments = ['<!doctype html><html><meta charset="utf-8"><title>Blinded CAD comparison</title>',
                 '<style>body{font:16px sans-serif;max-width:1100px;margin:30px auto}section{border-bottom:1px solid #aaa;padding:20px 0}figure{display:inline-block;margin:5px}img{width:270px}code{font-size:13px}</style>',
                 '<h1>Blinded CAD comparison</h1><p>Rank candidates from best to worst by instruction following and geometric quality. Use each candidate label once.</p>']
    def image_tag(geometry, label):
        path = geometry.get('image') if geometry else None
        if path:
            relative = os.path.relpath(Path(path).resolve(), directory.resolve())
            return f'<figure><img src="{html.escape(relative, quote=True)}"><figcaption>{html.escape(label)}</figcaption></figure>'
        return f'<figure><p>{html.escape(label)}: image unavailable / invalid geometry</p></figure>'
    for sample in available:
        sid = sample['id']
        names = ['ours'] + sorted(name for name, rows in baselines.items() if sid in rows)[:2]
        random.Random(f'{settings["seed"]}:{sid}').shuffle(names)
        mapping = dict(zip(('A', 'B', 'C'), names))
        labels[sid] = mapping
        fragments += [f'<section><h2>{html.escape(sid)}</h2><p>{html.escape(sample["instruction"])}</p>']
        ours = per_sample[sid]
        if sample['task'] == 'editing':
            fragments.append(image_tag(ours.get('source_geometry'), 'Original'))
        fragments.append(image_tag(ours['reference_geometry'], 'Reference'))
        fragments.append('<div>')
        for label, name in mapping.items():
            row = ours if name == 'ours' else baselines[name][sid]
            fragments.append(image_tag(row['prediction_geometry'], label))
        fragments.append('</div></section>')
        packets.append({'sample_id': sid, 'candidate_labels': list(mapping)})
        for judge in range(1, 6):
            answer_rows.append({'sample_id': sid, 'judge_id': f'judge_{judge}', 'ranking': None})
    fragments.append('</html>')
    (directory / 'human_comparison.html').write_text('\n'.join(fragments))
    write_json(directory / 'human_key.json', {**headers, 'labels': labels})
    write_json(directory / 'human_scores.template.json', {**headers, 'key_file': 'human_key.json',
                                                         'key_sha256': file_hash(directory / 'human_key.json'),
                                                         'records': answer_rows})
    write_json(directory / 'export_report.json', {'vlm_requests': len(vlm_records), 'human_packets': len(packets),
                                                 'human_task_counts': dict(collections.Counter(s['task'] for s in available)),
                                                 'baseline_names': sorted(baselines),
                                                 'human_status': 'ready' if packets else 'needs_two_baseline_reports',
                                                 'note': 'Keep human_key.json away from judges. Null template entries are unscored, not zero.'})
    print(json.dumps({'judging_directory': str(directory), 'human_packets': len(packets), 'vlm_requests': len(vlm_records)}), flush=True)


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path, rows):
    with open(path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')
