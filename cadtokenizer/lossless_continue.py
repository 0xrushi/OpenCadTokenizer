"""Bounded step-500 continuation, preserving moments, RNG and data cursor."""
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import time

from .common import file_hash, read_config, require_container, load_checkpoint, write_json


def schedule_factor(step, start, end, warmup):
    if not start <= step < end or warmup < 1:
        raise ValueError('Invalid continuation schedule position')
    elapsed = step - start
    return min(1., (elapsed + 1) / warmup) * .5 * (1 + math.cos(math.pi * elapsed / (end - start)))


def recall_evaluate(model, cases, samples, codec, settings, step):
    from .lossless_sft import generate_one
    from .lossless_editing_eval import read_journal
    from .representation import stringify, tokenize
    from scripts.analyze_edit_correctness import units, command_types
    root = Path(settings['output']) / 'recall' / f'step-{step:06d}'
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'results.json').exists():
        return read_config(root / 'results.json')
    journal = root / 'predictions.jsonl'
    existing = read_journal(journal, {c['example_id'] for c in cases})
    model.eval()
    with journal.open('a') as stream:
        for case, sample in zip(cases, samples):
            if case['example_id'] in existing:
                continue
            row = {'id': case['example_id'], 'group': case['group'], 'sequence': None, 'error': None}
            try:
                row['sequence'], row['generated_ids'] = generate_one(model,
                    sample['input_ids'][:sample['prompt_length']], codec, settings)
            except ValueError as exc:
                row['error'] = str(exc)
            target = stringify(tokenize(case['target']))
            row.update(exact_target=row['sequence'] == target,
                       correct_command_types=command_types(units(row['sequence'])) == command_types(units(target)))
            stream.write(json.dumps(row) + '\n'); stream.flush(); os.fsync(stream.fileno())
            existing[row['id']] = row
            print(json.dumps({'phase': 'train_recall', 'step': step, 'completed': len(existing), 'total': 64}), flush=True)
    report = {}
    for group in ('all', 'add', 'add_numeric', 'add_no_numeric', 'remove'):
        rows = [r for r in existing.values() if group == 'all' or r['group'] == group or group == 'add' and r['group'].startswith('add_')]
        report[group] = {'cases': len(rows), 'exact_targets': sum(r['exact_target'] for r in rows),
                         'correct_command_types_but_wrong_numbers': sum(r['correct_command_types'] and not r['exact_target'] for r in rows)}
    write_json(root / 'results.json', report)
    return report


def run_continuation(mode='train'):
    require_container()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from .lossless_sft import (freeze_pilot, load_samples, build_model, optimizer_for,
                               teacher_evaluate, generation_evaluate, save_pilot)
    from .lossless_sem import LosslessSemCodec
    from .sft import InstructionCollator
    from .tokenizer_learning_curve import epoch_order, preserve_rng
    settings = read_config('configs/lossless-sft-continuation.json')
    if mode == 'smoke':
        settings.update(output=settings['output'] + '-smoke', max_updates=502,
                        teacher_evaluation_updates=[500,502], checkpoint_updates=[502],
                        generation_evaluation_updates=[], recall_evaluation_updates=[], logging_updates=1)
    root = Path(settings['output']); root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock, sdpa_kernel(SDPBackend.MATH):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if (root / 'state.json').exists() and read_config(root / 'state.json').get('status') == 'complete':
                return read_config(root / 'results.json')
            write_json(root / 'state.json', {'status': 'running', 'phase': 'freeze'})
            torch.set_num_threads(2); torch.set_float32_matmul_precision('high')
            torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
            parent = Path(settings['parent'])
            original = read_config(parent.parents[1] / 'manifest.json')
            for key in ('seed','codec_root','model_path','train_counts','validation_counts','effective_batch_size',
                        'learning_rate','token_learning_rate','weight_decay','lora_rank','lora_alpha','lora_targets',
                        'max_new_tokens','max_pairs','context_length','point_count','kernel_workers'):
                if settings[key] != original['signature']['settings'][key]:
                    raise ValueError('Unexpected training/evaluation recipe change: ' + key)
            settings['parent_fingerprints'] = {str(p): file_hash(p) for p in [parent / 'state.pt', parent / 'complete.json', *sorted((parent / 'adapter').glob('*'))] if p.is_file()}
            settings['recall_plan_sha256'] = file_hash(settings['recall_plan'])
            manifest = freeze_pilot(settings)
            if manifest['train'] != original['train'] or manifest['validation'] != original['validation']:
                raise ValueError('Frozen training or validation examples changed')
            for key in ('cache_sha256','codec_sha256','source_database_sha256','model_sha256','image_id'):
                if manifest['signature'][key] != original['signature'][key]:
                    raise ValueError('Parent runtime/data/model differs: ' + key)
            for name in ('lossless_model.py','lossless_sem.py','lossless_sft.py'):
                path = 'cadtokenizer/' + name
                if file_hash(path) != original['signature']['code_sha256'][path]:
                    raise ValueError('Model/loss/decoding implementation changed')
            (root / 'source').mkdir(exist_ok=True)
            for path in Path('cadtokenizer').glob('*.py'):
                shutil.copy2(path, root / 'source' / path.name)
            codec = LosslessSemCodec.from_dict(read_config(Path(settings['codec_root']) / 'codec.json'))
            train = load_samples(settings, manifest['train'])
            validation = load_samples(settings, manifest['validation'])
            selected = {c['example_id']: c for c in manifest['train']}
            recall_ids = read_config(settings['recall_plan'])['sample_ids']
            recall_cases = []
            if len(recall_ids) != 64 or len(set(recall_ids)) != 64:
                raise ValueError('Expected frozen 64-example recall panel')
            with sqlite3.connect('data/prepared/cad.sqlite') as db:
                for i, key in enumerate(recall_ids):
                    target, split = db.execute('SELECT g.sequence,e.split FROM examples e JOIN geometry g ON g.id=e.target_id WHERE e.id=?', (key,)).fetchone()
                    if split != 'train' or key not in selected:
                        raise ValueError('Recall panel outside training')
                    recall_cases.append({**selected[key], 'target': target,
                        'group': 'add_numeric' if i < 16 else 'add_no_numeric' if i < 32 else 'remove'})
            recall_samples = load_samples(settings, recall_cases)
            write_json(root / 'recall_manifest.json', recall_cases)
            checkpoints = sorted(p for p in (root / 'checkpoints').glob('step-*') if (p / 'complete.json').exists())
            resume = checkpoints[-1] if checkpoints else parent
            write_json(root / 'state.json', {'status': 'running', 'phase': 'load_model'})
            model, _, _ = build_model(settings, codec, resume)
            optimizer, names = optimizer_for(model, settings)
            saved = load_checkpoint(resume / 'state.pt')
            expected = manifest['signature'] if checkpoints else original['signature']
            if saved['signature'] != expected or saved['parameter_names'] != names:
                raise ValueError('Optimizer/checkpoint provenance mismatch')
            optimizer.load_state_dict(saved['optimizer'])
            step, epoch, offset = (saved[k] for k in ('step','epoch','offset'))
            if not checkpoints and (step, epoch, offset) != (500,1,928):
                raise ValueError('Parent step/data cursor mismatch')
            torch.set_rng_state(saved['rng']); torch.cuda.set_rng_state_all(saved['cuda_rng']); del saved
            # Preserve the original nonzero peak rates after loading optimizer groups.
            for group, rate in zip(optimizer.param_groups, (settings['learning_rate'],settings['token_learning_rate'])):
                group['initial_lr'] = rate
            write_json(root / 'resume_verified.json', {'checkpoint': str(resume), 'step': step, 'epoch': epoch,
                'offset': offset, 'optimizer_moments_restored': True, 'rng_restored': True,
                'schedule': settings['schedule_note']})
            collator = InstructionCollator(128001)
            def evaluate(current):
                if current in settings['teacher_evaluation_updates']:
                    path = root / 'teacher' / f'step-{current:06d}.json'
                    if not path.exists():
                        write_json(root / 'state.json', {'status':'running','phase':'teacher_validation','step':current})
                        with preserve_rng(): report = teacher_evaluate(model, validation, collator)
                        if current == 500:
                            baseline = read_config(parent.parents[1] / 'teacher/step-000500.json')
                            if report['target_tokens'] != baseline['target_tokens'] or abs(report['loss'] - baseline['loss']) > 1e-6:
                                raise ValueError('Restored parent teacher loss does not reproduce the saved baseline')
                        write_json(path, report)
                if current in settings['recall_evaluation_updates']:
                    write_json(root / 'state.json', {'status':'running','phase':'train_recall','step':current})
                    with preserve_rng(): recall_evaluate(model, recall_cases, recall_samples, codec, settings, current)
                if current in settings['generation_evaluation_updates']:
                    write_json(root / 'state.json', {'status':'running','phase':'free_running_validation','step':current})
                    with preserve_rng(): generation_evaluate(model, manifest['validation'], validation, codec, settings, current)
            evaluate(step)
            started = time.monotonic(); first_step = step
            order = epoch_order(len(train), settings['seed'], epoch)
            while step < settings['max_updates']:
                chosen = order[offset:offset + settings['effective_batch_size']]
                samples = [train[i] for i in chosen]
                total = sum(sum(v != -100 for v in s['labels'][1:]) for s in samples)
                # Smoke uses the real continuation cycle's first two update rates.
                factor = schedule_factor(step, 500, 2000, settings['warmup_updates'])
                for group in optimizer.param_groups:
                    group['lr'] = group['initial_lr'] * factor
                model.train(); model.config.use_cache = False; optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.
                for sample in samples:
                    batch = {k:v.cuda() for k,v in collator([sample]).items()}
                    count = int(batch['labels'][:,1:].ne(-100).sum())
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        output = model(**batch); loss = output.loss * count / total
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Non-finite continuation loss')
                    loss.backward(); loss_sum += float(loss.detach()); del output,loss,batch
                if step == first_step:
                    checks = {n:bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0)
                        for n,p in model.named_parameters() if p.requires_grad and 'cad_rows.modules_to_save.' in n}
                    if len(checks) != 2 or not all(checks.values()):
                        raise RuntimeError('Both token row modules must receive finite nonzero gradients')
                    write_json(root / 'first_update_row_gradients.json', checks)
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                optimizer.step(); step += 1; offset += len(chosen)
                if offset == len(train):
                    epoch += 1; offset = 0; order = epoch_order(len(train),settings['seed'],epoch)
                if step % settings['logging_updates'] == 0:
                    metrics = {'status':'running','phase':'training','step':step,'epoch':epoch,'offset':offset,
                        'loss':loss_sum,'gradient_norm':float(norm),'seconds_per_update':(time.monotonic()-started)/(step-first_step)}
                    write_json(root / 'state.json',metrics)
                    with (root / 'training.jsonl').open('a') as stream: stream.write(json.dumps(metrics)+'\n')
                    print(json.dumps(metrics),flush=True)
                if step in settings['checkpoint_updates']:
                    save_pilot(model,optimizer,settings,manifest,step,epoch,offset,names)
                evaluate(step)
            result = {'updates':step,'additional_updates':step-500,'production_promoted':False,
                'mode':mode,
                'teacher':{str(i):read_config(root/'teacher'/f'step-{i:06d}.json') for i in settings['teacher_evaluation_updates']},
                'recall':{str(i):read_config(root/'recall'/f'step-{i:06d}'/'results.json') for i in settings['recall_evaluation_updates']},
                'generation':{str(i):read_config(root/'generation'/f'step-{i:06d}'/'results.json') for i in settings['generation_evaluation_updates']},
                'limits':'Same frozen data/architecture/loss, with a new learning-rate cycle. Fixed 2000-update endpoint, not a duration-only causal ablation. Recall is training data; validation is the existing 60-case panel. No new data, test use or automatic promotion.'}
            write_json(root/'results.json',result)
            (root/'findings.md').write_text('# Bounded lossless continuation\n\n'+json.dumps(result,indent=2)+'\n')
            write_json(root/'state.json',{'status':'complete','results':str(root/'results.json')})
        except Exception as exc:
            write_json(root/'state.json',{'status':'failed','error':repr(exc)})
            raise
