"""Matched CE versus origin-distance continuation on training/validation only."""
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import statistics

from .common import file_hash, load_checkpoint, read_config, require_container, save_checkpoint, seed_all, write_json


def warm_start(parent, cfg, signature, trajectory_seed=None):
    """Keep weights/moments/cursor; optionally choose a new dropout stream."""
    checkpoint = dict(parent, config=cfg, experiment_signature=signature)
    if trajectory_seed is not None:
        import torch
        seed_all(trajectory_seed)
        checkpoint.update(rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    return checkpoint


def reuse_reference(settings, label, checkpoint, cases, protocol, numerical_protocol, root):
    """Reuse frozen inference only when weights, cases, code and backend match."""
    from .tokenizer_experiment import code_signature
    previous_root = Path(settings['replication_results']).parent
    report = read_config(previous_root/label/'report.json')
    old_signature = report['signature']
    previous_result = read_config(settings['replication_results'])
    previous_manifest = read_config(previous_root/'manifest.json')
    import torch
    if (old_signature['checkpoint_sha256'] != file_hash(checkpoint)
            or old_signature['cases'] != cases or old_signature['code'] != code_signature()
            or old_signature['image_id'] != os.environ.get('CADTOKENIZER_IMAGE_ID')
            or previous_manifest['signature']['torch'] != str(torch.__version__)
            or previous_result['numerical_protocol'] != numerical_protocol
            or any(old_signature['settings'][key] != protocol[key] for key in ('seed','point_count','kernel_workers'))):
        raise ValueError('Frozen reference inference provenance differs')
    # Geometry parameters are stored beside the original sampled surfaces.
    benchmark = read_config('configs/benchmark.json')
    expected_geometry = {key:benchmark[key] for key in (
        'kernel_backend','point_count','mesh_tolerance','mesh_angular_tolerance',
        'boolean_tolerance','minimum_volume','render','render_axis_limit','render_max_triangles')}
    expected_geometry.update(render=False,point_count=protocol['point_count'])
    rows = read_config(previous_root/label/'per_case.json')
    for row in rows:
        reference = row['geometry']['reference']
        if reference['status']=='valid':
            request = read_config(Path(reference['points']).parent/'request.json')
            if request['settings'] != expected_geometry:
                raise ValueError('Reference geometry parameters changed')
    write_json(root/label/'report.json',report)
    write_json(root/label/'per_case.json',rows)
    write_json(root/label/'reused_reference.json',{
        'source_report':str(previous_root/label/'report.json'),
        'source_report_sha256':file_hash(previous_root/label/'report.json'),
        'source_rows_sha256':file_hash(previous_root/label/'per_case.json'),
        'verified':'Same checkpoint, cases, evaluator/model/geometry source hashes, image, torch, numerical backend, sampling seed and geometry parameters. Original report signature retained; inference not rerun.'})
    return report


def origin_metrics(rows, originals):
    from .tokenizer_shape_audit import parts
    errors = []
    exact = count = 0
    for row in rows:
        if row['sequence'] is None:
            continue
        truth, prediction = parts(originals[row['id']]), parts(row['sequence'])
        if len(truth) != len(prediction):
            continue
        count += 1
        for a, b in zip(truth, prediction):
            for offset in (16, 17):
                errors.append(abs(a[1][offset]-b[1][offset])/63)
                exact += a[1][offset] == b[1][offset]
    return {'decoded_cases': count, 'origin_fields': len(errors),
            'normalized_mean_absolute_bin_error': statistics.mean(errors) if errors else None,
            'exact_field_rate': exact/len(errors) if errors else None}


def run_origin_ablation(settings):
    """Apply the audited numerical protocol before initializing CUDA."""
    require_container()
    if settings.get('strict_determinism', False):
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        import torch
        from torch.nn.attention import sdpa_kernel, SDPBackend
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        with sdpa_kernel(SDPBackend.MATH):
            return _run_origin_ablation(settings)
    return _run_origin_ablation(settings)


def _run_origin_ablation(settings):
    import torch
    from .tokenizer_learning_curve import train_variant, paired_comparison, share_geometry_cache, evaluation_settings
    from .tokenizer_experiment import evaluate_checkpoint
    from .vq import LOSS_NORMALIZATION
    root = Path(settings['output'])
    root.mkdir(parents=True, exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            torch.set_num_threads(2)
            torch.set_float32_matmul_precision('high')
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is required')
            numerical_protocol = {
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                'math_sdpa_only': not torch.backends.cuda.flash_sdp_enabled()
                                  and not torch.backends.cuda.mem_efficient_sdp_enabled()
                                  and torch.backends.cuda.math_sdp_enabled()
                                  and not torch.backends.cuda.cudnn_sdp_enabled(),
                'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'matmul_precision': torch.get_float32_matmul_precision(),
                'bf16_autocast': torch.cuda.is_bf16_supported()}
            if settings.get('strict_determinism', False) and not (
                    numerical_protocol['deterministic_algorithms']
                    and numerical_protocol['math_sdpa_only']
                    and numerical_protocol['cublas_workspace_config'] == ':4096:8'
                    and not numerical_protocol['cudnn_benchmark']):
                raise RuntimeError('Audited strict numerical protocol is not active')
            write_json(root/'numerical_protocol.json', numerical_protocol)
            if settings['evaluation_updates'] != [42500, 45000] or settings['max_updates'] != 45000:
                raise ValueError('Expected the registered 5000-update pilot')
            parent_manifest = read_config(settings['parent_manifest'])
            parent = load_checkpoint(settings['checkpoint'])
            source = read_config(settings['source_manifest'])
            base_rows = read_config(settings['baseline_rows'])
            if (parent['step'] != 40000 or parent['loss_normalization'] != LOSS_NORMALIZATION
                    or parent['experiment_signature'] != parent_manifest['signature']
                    or parent['config']['vq']['reconstruction_loss'] != 'cross_entropy'
                    or parent['config']['vq']['learning_rate'] != 3e-5
                    or parent['config']['vq'].get('origin_loss_weight', 0) != 0):
                raise ValueError('Unexpected parent checkpoint')
            database = parent['config']['data']['database']
            if file_hash(database) != parent['database_sha256']:
                raise ValueError('Training database changed')
            if {r['id'] for r in base_rows} != {r['id'] for r in source['validation']}:
                raise ValueError('Reference validation panel mismatch')
            if any(group['lr'] != 3e-5 for group in parent['optimizer']['param_groups']):
                raise ValueError('Parent optimizer rate differs')
            signature = {'schema': 'origin-ordinal-ablation-v1', 'settings': settings,
                         'checkpoint_sha256': file_hash(settings['checkpoint']),
                         'database_sha256': parent['database_sha256'],
                         'input_sha256': {key: file_hash(settings[key]) for key in
                                          ('parent_manifest','source_manifest','training_settings','baseline_rows','historical_control_rows')},
                         'code': {str(p): file_hash(p) for p in sorted(Path('cadtokenizer').glob('*.py'))},
                         'numerical_protocol': numerical_protocol,
                         'replication_results_sha256':file_hash(settings['replication_results']) if settings.get('replication_results') else None,
                         'baseline_checkpoint_sha256': file_hash(settings['baseline_checkpoint']) if settings.get('strict_determinism') else None,
                         'image_id': os.environ.get('CADTOKENIZER_IMAGE_ID'), 'torch': str(torch.__version__)}
            path = root/'manifest.json'
            manifest = {'signature': signature, 'validation': source['validation'],
                        'design': 'Two arms warm-start the same 40000 state, optimizer moments, RNG and data cursor. Only the auxiliary origin weight differs. 5000 further updates at 3e-5; no test data.',
                        'objective': 'CE + weight/N_tokens * sum_ox,oy E_numeric[((bin-target)/63)^2] + VQ. Weight=0 control, 32 treatment. Numeric probabilities condition on tokens 13..76. All-token CE remains.',
                        'production_promotion': False}
            if 'trajectory_seed' in settings:
                manifest['design'] += (f" Replication changes the training permutation seed and dropout RNG to {settings['trajectory_seed']} in both arms; saved epoch/offset is retained. Geometry evaluation seed remains {settings.get('evaluation_seed',42)}. This replicates a continuation trajectory, not independent model initialization.")
            if path.exists() and read_config(path) != manifest:
                raise ValueError('Ablation provenance changed; choose a fresh output')
            write_json(path, manifest)
            archive = root/'source'
            archive.mkdir(exist_ok=True)
            for p in Path('cadtokenizer').glob('*.py'):
                shutil.copy2(p, archive/p.name)
            shutil.copy2('scripts/cad.sh', archive/'cad.sh')
            protocol = read_config(settings['training_settings'])
            protocol.update(output=str(root),max_updates=settings['max_updates'], evaluation_updates=settings['evaluation_updates'])
            if 'trajectory_seed' in settings:
                protocol.update(seed=settings['trajectory_seed'],evaluation_seed=settings.get('evaluation_seed',42))
            write_json(root/'training-settings.json',protocol)
            reference_report = None
            if settings.get('strict_determinism', False):
                write_json(root/'state.json', {'status':'running','phase':'existing_tokenizer_evaluation'})
                # Re-evaluate the existing tokenizer under the same strict
                # inference backend as both new arms, preserving the old rows.
                if settings.get('reuse_reference_reports'):
                    reference_report = reuse_reference(settings,'reference',settings['baseline_checkpoint'],source['validation'],evaluation_settings(protocol),numerical_protocol,root)
                else:
                    reference_report = evaluate_checkpoint(settings['baseline_checkpoint'], source['validation'], evaluation_settings(protocol), root/'reference')
                base_rows = read_config(root/'reference/per_case.json')
            write_json(root/'state.json', {'status':'running','phase':'parent_evaluation'})
            # Evaluate the unchanged parent under the current evaluator too.
            if settings.get('reuse_reference_reports'):
                parent_report = reuse_reference(settings,'parent',settings['checkpoint'],source['validation'],evaluation_settings(protocol),numerical_protocol,root)
            else:
                parent_report = evaluate_checkpoint(settings['checkpoint'], source['validation'], evaluation_settings(protocol), root/'parent')
            if settings.get('replication_results'):
                reference_checks = {}
                previous_root = Path(settings['replication_results']).parent
                for label in ('reference','parent'):
                    old = {row['id']:row for row in read_config(previous_root/label/'per_case.json')}
                    current_rows = read_config(root/label/'per_case.json')
                    if {row['id'] for row in current_rows} != old.keys():
                        raise ValueError('Replication reference case IDs changed')
                    reference_checks[label] = {
                        'cases':len(current_rows),
                        'inference_reused':settings.get('reuse_reference_reports',False),
                        'identical_sem_outputs':sum(row['sequence']==old[row['id']]['sequence'] for row in current_rows),
                        'identical_validity':sum(row['paired_valid']==old[row['id']]['paired_valid'] for row in current_rows)}
                write_json(root/'reference_consistency.json',reference_checks)
                if any(item['identical_sem_outputs']!=item['cases'] or item['identical_validity']!=item['cases']
                       for item in reference_checks.values()):
                    raise ValueError('Unchanged references differ across trajectories; audit before training')
            outputs = {}
            for name, weight in [('control', 0.0), ('origin', settings['origin_loss_weight'])]:
                arm_root = root/name
                arm_root.mkdir(exist_ok=True)
                cfg = copy.deepcopy(parent['config'])
                if 'trajectory_seed' in settings:
                    cfg['seed'] = settings['trajectory_seed']
                cfg['vq'].update(output=str(arm_root/'curve'), origin_loss_weight=weight)
                write_json(arm_root/'curve/config.json', cfg)
                write_json(arm_root/'baseline/per_case.json', base_rows)
                cache_sources = [root/'reference/geometry',root/'parent/geometry']
                if name == 'origin':
                    cache_sources += [root/'control/curve'/f'evaluation-{step}'/'geometry'
                                      for step in settings['evaluation_updates']]
                share_geometry_cache(cache_sources, arm_root/'baseline/geometry')
                arm_manifest = {'signature': {'experiment':signature,'arm':name,'origin_loss_weight':weight},
                                'validation': source['validation'], 'variants': {'curve':{
                                    'config_path': str(arm_root/'curve/config.json'),
                                    'database_sha256': parent['database_sha256']}}}
                write_json(arm_root/'manifest.json', arm_manifest)
                if not (arm_root/'curve/last.pt').exists():
                    checkpoint = warm_start(parent,cfg,arm_manifest['signature'],settings.get('trajectory_seed'))
                    save_checkpoint(arm_root/'curve/last.pt', checkpoint)
                    del checkpoint
                write_json(root/'state.json', {'status':'running','phase':'training_and_validation','arm':name})
                outputs[name] = train_variant(arm_manifest, protocol, 'curve', arm_root)
            originals = {r['id']:r['sequence'] for r in source['validation']}
            comparisons = []
            for step in settings['evaluation_updates']:
                control, treatment = [read_config(root/name/'curve'/f'evaluation-{step}'/'per_case.json')
                                      for name in ('control','origin')]
                comparisons.append({'step':step, 'paired':paired_comparison(control,treatment),
                                    'control_origin':origin_metrics(control, originals),
                                    'treatment_origin':origin_metrics(treatment, originals),
                                    'treatment_vs_existing':paired_comparison(base_rows,treatment)})
            historical = {r['id']:r for r in read_config(settings['historical_control_rows'])}
            current = read_config(root/'control/curve/evaluation-45000/per_case.json')
            if {r['id'] for r in current} != historical.keys():
                raise ValueError('Historical control case IDs differ')
            reproduction = {'cases':len(current), 'identical_sem_outputs':sum(
                r['sequence']==historical[r['id']]['sequence'] for r in current),
                'numerical_protocol_changed':settings.get('strict_determinism',False),
                'interpretation':'Historical agreement is descriptive; strict and historical runs use different numerical protocols.' if settings.get('strict_determinism') else 'Same declared production numerical protocol; exact reproduction was not guaranteed.'}
            result = {'arms':outputs, 'comparisons':comparisons, 'parent':parent_report['summary'],
                      'historical_control_reproduction':reproduction,
                      'reference': reference_report['summary'] if reference_report else None,
                      'numerical_protocol': numerical_protocol,
                      'production_promoted':False}
            if settings.get('replication_results'):
                previous = read_config(settings['replication_results'])
                endpoint = settings['max_updates']
                endpoint_rows = [next(row for row in run['comparisons'] if row['step']==endpoint)
                                 for run in (previous,result)]
                supported = []
                replication_lines = ['# Origin-loss continuation replication', '',
                                     'Same parent weights and optimizer; two training trajectories. Geometry cases and sampling seed stay fixed. This is not a comparison of independent model initializations.', '',
                                     '| Trajectory | Endpoint | Valid CE / origin | Paired CD CE / origin | Origin MAE CE / origin |',
                                     '| --- | ---: | ---: | ---: | ---: |']
                for label,row in zip(('Saved parent RNG, order seed 42',f"Reseeded {settings['trajectory_seed']}"),endpoint_rows):
                    p=row['paired']
                    supported.append(p['candidate_valid']>=p['baseline_valid']
                                     and p['candidate_mean_chamfer_common']<p['baseline_mean_chamfer_common']
                                     and row['treatment_origin']['normalized_mean_absolute_bin_error']<row['control_origin']['normalized_mean_absolute_bin_error'])
                    replication_lines.append(f"| {label} | {endpoint} | {p['baseline_valid']} / {p['candidate_valid']} | {p['baseline_mean_chamfer_common']} / {p['candidate_mean_chamfer_common']} | {row['control_origin']['normalized_mean_absolute_bin_error']} / {row['treatment_origin']['normalized_mean_absolute_bin_error']} |")
                result['replication']={'endpoint_benefit_by_trajectory':supported,
                                       'benefit_repeated':all(supported),
                                       'criterion':'At the registered 45k endpoint: no fewer valid solids, lower paired mean CD and lower origin MAE than matched CE in both trajectories. This is separate from beating the existing tokenizer.'}
                replication_lines += ['',json.dumps(result['replication'],indent=2)]
                (root/'replication.md').write_text('\n'.join(replication_lines)+'\n')
            write_json(root/'results.json',result)
            lines = ['# Origin offset objective: matched continuation', '', manifest['design'], '', manifest['objective'], '',
                     '| Updates | Valid control / treatment | Paired CD control / treatment | Origin MAE control / treatment |',
                     '| --- | ---: | ---: | ---: |']
            for row in comparisons:
                p = row['paired']
                lines.append(f"| {row['step']} | {p['baseline_valid']} / {p['candidate_valid']} | {p['baseline_mean_chamfer_common']} / {p['candidate_mean_chamfer_common']} | {row['control_origin']['normalized_mean_absolute_bin_error']} / {row['treatment_origin']['normalized_mean_absolute_bin_error']} |")
            lines += ['', 'Numerical protocol: '+json.dumps(numerical_protocol), '',
                      'The strict rerun re-evaluates the existing tokenizer under the same inference backend. Historical-control agreement is descriptive when the numerical protocol changes.', '',
                      'Origin MAE is from free-running SEM, across decoded cases with matching pair counts, including kernel-invalid outputs. Geometry compares shared valid IDs. One seed, short continuation, one registered auxiliary weight; this does not establish full convergence or resolve invalid-sketch failures. No automatic promotion or test evaluation.']
            (root/'findings.md').write_text('\n'.join(lines)+'\n')
            write_json(root/'state.json', {'status':'complete','results':str(root/'results.json')})
            print(json.dumps({'status':'complete','comparisons':comparisons}),flush=True)
        except Exception as exc:
            write_json(root/'state.json', {'status':'failed','error':repr(exc)})
            raise
