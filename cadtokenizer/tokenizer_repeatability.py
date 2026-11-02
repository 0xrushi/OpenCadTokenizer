"""Short fresh-process and save/resume replays of the same saved CE state."""
from contextlib import nullcontext
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .common import file_hash, load_checkpoint, read_config, require_container, save_checkpoint, seed_all, write_json


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(-1).view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def rng_hash(torch):
    return {'cpu': tensor_hash(torch.get_rng_state()),
            'cuda': [tensor_hash(state) for state in torch.cuda.get_rng_state_all()]}


def replay(source, output, stop, strict):
    require_container()
    import torch
    from .representation import PAD
    from .vq import PairDataset, PrimitiveVQVAE, accumulated_loss, collate_pairs, to_device
    from .tokenizer_learning_curve import epoch_order
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.use_deterministic_algorithms(strict)
    if strict:
        torch.backends.cudnn.benchmark = False
    seed_all(42)
    checkpoint = load_checkpoint(source)
    cfg = checkpoint['config']
    dataset = PairDataset(cfg['data']['database'], 'train')
    pairs = [dataset[i] for i in range(len(dataset))]
    model = PrimitiveVQVAE(cfg).cuda()
    model.load_state_dict(checkpoint['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['vq']['learning_rate'], weight_decay=cfg['vq']['weight_decay'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    step, epoch, offset = (checkpoint[key] for key in ('step','epoch','offset'))
    start_step = step
    torch.set_rng_state(checkpoint['rng'])
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    environment = {'torch':str(torch.__version__), 'cuda':torch.version.cuda,
                   'gpu':torch.cuda.get_device_name(), 'cudnn':torch.backends.cudnn.version(),
                   'matmul_precision':torch.get_float32_matmul_precision(),
                   'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
                   'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                   'strict_math_sdpa':strict, 'bf16_autocast':torch.cuda.is_bf16_supported(),
                   'image_id':os.environ.get('CADTOKENIZER_IMAGE_ID')}
    write_json(root/'environment.json',environment)
    # Strict replays intentionally use a new math-attention protocol. Default
    # replays retain the production attention selection and precision settings.
    if strict:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        context = sdpa_kernel(SDPBackend.MATH)
    else:
        context = nullcontext()
    trace = []
    order = epoch_order(len(pairs),42,epoch)
    with context:
        while step < stop:
            indices = order[offset:offset+64]
            batch = to_device(collate_pairs([pairs[i] for i in indices]), torch.device('cuda'))
            total_tokens = int(batch['targets'].ne(PAD).sum())
            total_primitives = int((batch['groups'].max(dim=1).values+1).sum())
            before_rng = rng_hash(torch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                result = model(**batch)
                loss = accumulated_loss(result,total_tokens,total_primitives)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Replay loss is not finite')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['vq']['gradient_clip'],error_if_nonfinite=True)
            optimizer.step()
            step += 1
            trace.append({'step':step,'epoch_before':epoch,'offset_before':offset,
                          'batch_sha256':hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
                          'rng_before':before_rng,'loss':float(loss.detach()),
                          'gradient_norm':float(norm),'codes_sha256':tensor_hash(result['indices'])})
            offset += len(indices)
            if offset == len(pairs):
                epoch,offset = epoch+1,0
                order = epoch_order(len(pairs),42,epoch)
    checkpoint.update(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,epoch=epoch,offset=offset,
                      rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    save_checkpoint(root/'last.pt',checkpoint)
    write_json(root/'trace.json',trace)
    write_json(root/'complete.json',{'start_step':start_step,'end_step':step,'epoch':epoch,'offset':offset,
                                   'final_rng':rng_hash(torch),'strict':strict})


def compare(left, right):
    import torch
    a,b = [load_checkpoint(Path(p)/'last.pt') for p in (left,right)]
    differences = []
    max_abs = 0.0
    for key in a['model']:
        x,y = a['model'][key],b['model'][key]
        if not torch.equal(x,y):
            differences.append(key)
            max_abs = max(max_abs,float((x.float()-y.float()).abs().max()))
    def equal(x,y):
        if isinstance(x,torch.Tensor):
            return isinstance(y,torch.Tensor) and torch.equal(x,y)
        if isinstance(x,dict):
            return isinstance(y,dict) and x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
        if isinstance(x,(tuple,list)):
            return type(x)==type(y) and len(x)==len(y) and all(equal(u,v) for u,v in zip(x,y))
        return x==y
    ta,tb = [read_config(Path(p)/'trace.json') for p in (left,right)]
    by_step = {row['step']:row for row in ta}
    shared = [(by_step[row['step']],row) for row in tb if row['step'] in by_step]
    numerical = [(u,v) for u,v in shared if any(u[k]!=v[k] for k in ('loss','gradient_norm','codes_sha256'))]
    return {'weights_exact':not differences,'different_weight_tensors':len(differences),
            'max_absolute_weight_difference':max_abs,'optimizer_exact':equal(a['optimizer'],b['optimizer']),
            'final_rng_exact':equal(a['rng'],b['rng']) and equal(a['cuda_rng'],b['cuda_rng']),
            'cursor_exact':all(a[k]==b[k] for k in ('step','epoch','offset')),
            'shared_trace_steps':len(shared),'batch_order_exact':all(u['batch_sha256']==v['batch_sha256'] for u,v in shared),
            'per_step_rng_exact':all(u['rng_before']==v['rng_before'] for u,v in shared),
            'first_numerical_trace_difference':numerical[0][0]['step'] if numerical else None}


def run_repeatability(settings):
    require_container()
    root = Path(settings['output'])
    root.mkdir(parents=True,exist_ok=True)
    signature = {'settings':settings,'checkpoint_sha256':file_hash(settings['checkpoint']),
                 'code':{str(p):file_hash(p) for p in sorted(Path('cadtokenizer').glob('*.py'))},
                 'image_id':os.environ.get('CADTOKENIZER_IMAGE_ID')}
    path = root/'manifest.json'
    if path.exists() and read_config(path)!=signature:
        raise ValueError('Replay inputs changed; use a fresh directory')
    write_json(path,signature)
    archive = root/'source'
    archive.mkdir(exist_ok=True)
    for p in Path('cadtokenizer').glob('*.py'):
        shutil.copy2(p,archive/p.name)
    # Source comparison is read-only, independent of any historical metric.
    source_notes = {}
    for filename in ('tokenizer_learning_curve.py','vq.py','representation.py'):
        old = Path('artifacts/tokenizer-low-lr-v1/source')/filename
        new = Path('artifacts/tokenizer-origin-ablation-v1/source')/filename
        source_notes[filename]={'old_sha256':file_hash(old),'new_sha256':file_hash(new),
                                'ast_equal':ast.dump(ast.parse(old.read_text()))==ast.dump(ast.parse(new.read_text()))}
    write_json(root/'source_comparison.json',source_notes)
    outcomes = {}
    comparisons = {}
    try:
        for strict in (False,True):
            prefix = 'strict' if strict else 'default'
            specs = [('a',settings['checkpoint'],40064),('b',settings['checkpoint'],40064),
                     ('head',settings['checkpoint'],40032),('resumed',str(root/f'{prefix}-head/last.pt'),40064)]
            for name,source,stop in specs:
                run_name = f'{prefix}-{name}'
                output = root/run_name
                write_json(root/'state.json',{'status':'running','phase':run_name})
                if not (output/'complete.json').exists():
                    output.mkdir(exist_ok=True)
                    env = os.environ.copy()
                    if strict:
                        env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
                    command = [sys.executable,'-m','cadtokenizer.tokenizer_repeatability','--worker',
                               '--source',str(source),'--output',str(output),'--stop',str(stop)]
                    if strict:
                        command.append('--strict')
                    with (output/'worker.log').open('w') as log:
                        result = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                    if result.returncode:
                        outcomes[run_name]={'status':'failed','log':str(output/'worker.log')}
                        break
                outcomes[run_name]={'status':'complete'}
            for label,other in [('fresh_process','b'),('save_resume','resumed')]:
                if all(outcomes.get(f'{prefix}-{name}',{}).get('status')=='complete' for name in ('a',other)):
                    comparisons[f'{prefix}_{label}']=compare(root/f'{prefix}-a',root/f'{prefix}-{other}')
        write_json(root/'results.json',{'outcomes':outcomes,'comparisons':comparisons,'source_comparison':source_notes})
        lines = ['# Continuation repeatability audit','',
                 '64 updates from the same curve 40k checkpoint, fresh processes, and a split after 32 updates. Default uses production BF16/high precision and automatic attention selection. Strict uses deterministic algorithms, CUBLAS_WORKSPACE_CONFIG=:4096:8 and math SDPA; this is a different numerical protocol. No test data or geometry selection is used.','',
                 '| Comparison | Weights exact | Optimizer exact | RNG exact | Batch order exact | First numerical trace difference | Max weight difference |',
                 '| --- | --- | --- | --- | --- | ---: | ---: |']
        for name,row in comparisons.items():
            lines.append(f"| {name} | {row['weights_exact']} | {row['optimizer_exact']} | {row['final_rng_exact']} | {row['batch_order_exact']} | {row['first_numerical_trace_difference']} | {row['max_absolute_weight_difference']} |")
        lines += ['', 'A short replay can identify current repeatability limits; it cannot by itself prove the cause of historical 5,000-update drift. Compare RNG, sample order and numerical traces before attributing a cause. Strict-run failure logs remain preserved; no unsupported operation is silently allowed.']
        (root/'findings.md').write_text('\n'.join(lines)+'\n')
        write_json(root/'state.json',{'status':'complete','worker_failures':[k for k,v in outcomes.items() if v['status']=='failed']})
    except Exception as exc:
        write_json(root/'state.json',{'status':'failed','error':repr(exc)})
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--stop',type=int,required=True)
    parser.add_argument('--strict',action='store_true')
    args = parser.parse_args()
    replay(args.source,args.output,args.stop,args.strict)
