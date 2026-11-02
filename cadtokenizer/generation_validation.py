"""Frozen 500-shape validation of the generation specialist versus its parent."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gc
import json
import os
from pathlib import Path
import sqlite3
import statistics
import time

from .common import read_config,write_json,file_hash,require_container,fraction,seed_all
from .representation import stringify,tokenize

ROOT=Path('artifacts/lossless-generation-validation-v1')


def select_generation_cases():
    pilot=read_config('artifacts/lossless-sft-pilot-v1/manifest.json')
    prior=read_config('artifacts/lossless-editing-validation-v1/manifest.json')
    with sqlite3.connect('data/prepared/cad.sqlite') as db,sqlite3.connect('artifacts/lossless-sem-v2/tokens.sqlite') as cache:
        geometries={r[0]:r[1:] for r in db.execute("SELECT id,sequence,component,split FROM geometry WHERE split IN ('train','val')")}
        examples={r[0]:r[1:] for r in db.execute("SELECT id,instruction,source_id,target_id,split,task FROM examples WHERE split IN ('train','val')")}
        cached={r[0]:r[1] for r in cache.execute("SELECT example_id,id FROM samples WHERE split='val' AND task='generation' AND context_overflow=0")}
    excluded={x for c in prior['cases'] for x in c['components']}
    train_components=set()
    for cases,components in [(pilot['validation'],excluded),(pilot['train'],train_components)]:
        for case in cases:
            _,source,target,_,_=examples[case['example_id']]
            for endpoint in (source,target):
                if endpoint:components.add(geometries[endpoint][1])
    candidates=[]
    for key in sorted(cached,key=lambda k:(fraction(k,242),k)):
        instruction,source,target,split,task=examples[key]
        seq,component,endpoint_split=geometries[target]
        if source is not None or task!='generation' or split!=endpoint_split or split!='val':
            raise ValueError('Generation validation split/source mismatch')
        if component in train_components:raise ValueError('Training component leakage')
        if component in excluded:continue
        candidates.append({'example_id':key,'cache_id':cached[key],'task':'generation','instruction':instruction,
            'source':None,'target':seq,'target_id':target,'component':component})
    # One prompt per component makes the shape count unambiguous.
    unique={}
    for c in candidates:unique.setdefault(c['component'],c)
    cases=list(unique.values())[:500]
    if len(cases)!=500:raise ValueError('Insufficient fresh unique validation components')
    return cases,{'selection_seed':242,'eligible_prompts':len(candidates),'eligible_components':len(unique),
        'selected_prompts':len(cases),'selected_components':len({c['component'] for c in cases}),
        'training_component_overlap':0,'initial_panel_component_overlap':0,'prior_editing_panel_component_overlap':0}


def run_generation_validation():
    require_container()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    import numpy as np
    from scipy.spatial import cKDTree
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from .lossless_sft import build_model,load_samples,generate_one
    from .lossless_sem import LosslessSemCodec
    from .lossless_editing_eval import read_journal
    from .bench_data import code_hash
    from .bench_runtime import KernelRunner
    from .bench_metrics import normalize_points,cadfusion_f1,all_sample_count_f1
    ROOT.mkdir(parents=True,exist_ok=True)
    def status(phase,**extra):write_json(ROOT/'state.json',{'status':'running','phase':phase,**extra})
    with (ROOT/'run.lock').open('a') as lock,sdpa_kernel(SDPBackend.MATH):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            status('audit')
            s=read_config('configs/lossless-sft-pilot.json')
            pilot=read_config(Path(s['output'])/'manifest.json');original=pilot['signature']
            checkpoints={'parent':Path(s['output'])/'checkpoints/step-000500',
                'expert':Path('artifacts/lossless-task-experts-v1/generation/checkpoints/step-000500')}
            if file_hash('data/prepared/cad.sqlite')!=original['source_database_sha256'] or file_hash(Path(s['codec_root'])/'tokens.sqlite')!=original['cache_sha256'] or file_hash(Path(s['codec_root'])/'codec.json')!=original['codec_sha256']:
                raise ValueError('Frozen data or codec changed')
            for name,digest in original['model_sha256'].items():
                if file_hash(Path(s['model_path'])/name)!=digest:raise ValueError('Cached base model changed')
            for name in ('lossless_model.py','lossless_sem.py','lossless_sft.py'):
                path='cadtokenizer/'+name
                if file_hash(path)!=original['code_sha256'][path]:raise ValueError('Model or decoding implementation changed')
            if os.environ.get('CADTOKENIZER_IMAGE_ID')!=original['image_id']:raise ValueError('Pinned image changed')
            router=read_config('artifacts/lossless-task-experts-v1/router.json')
            expert_manifest=read_config('artifacts/lossless-task-experts-v1/manifest.json')
            for name,digest in expert_manifest['signature']['adapter'].items():
                if file_hash(checkpoints['parent']/'adapter'/name)!=digest:raise ValueError('Original parent adapter changed')
            for name,digest in router['adapter_sha256']['generation'].items():
                if file_hash(checkpoints['expert']/'adapter'/name)!=digest:raise ValueError('Generation expert fingerprint mismatch')
            cases,audit=select_generation_cases()
            codec=LosslessSemCodec.from_dict(read_config(Path(s['codec_root'])/'codec.json'))
            samples=load_samples(s,cases)
            if any(len(v['input_ids'])-v['prompt_length']-1>s['max_new_tokens'] for v in samples):
                raise ValueError('Fixed output budget does not cover every selected target')
            manifest={'signature':{'pilot_manifest_sha256':file_hash(Path(s['output'])/'manifest.json'),
                'expert_manifest_sha256':file_hash('artifacts/lossless-task-experts-v1/manifest.json'),
                'prior_editing_panel_sha256':file_hash('artifacts/lossless-editing-validation-v1/manifest.json'),
                'adapters':{arm:{p.name:file_hash(p) for p in sorted((path/'adapter').iterdir()) if p.is_file()} for arm,path in checkpoints.items()},
                'settings':s,'code_sha256':code_hash(),'image_id':os.environ.get('CADTOKENIZER_IMAGE_ID')},
                'audit':audit,'cases':cases,
                'decision_rule':'Expert must have no fewer valid outputs overall or on valid references, lower paired mean Chamfer on shared valid cases, and no fewer exact command-type sequences. This is a validation screen only, with no automatic promotion.',
                'limits':'500 fresh, unique validation components. Same fixed greedy policy and kernel/point-sampling settings for both checkpoints. No new training or upstream test cases.'}
            if (ROOT/'manifest.json').exists() and read_config(ROOT/'manifest.json')!=manifest:raise ValueError('Evaluation provenance changed; use a fresh output')
            write_json(ROOT/'manifest.json',manifest)
            torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
            torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
            arms={}
            for arm,path in checkpoints.items():
                journal=ROOT/f'{arm}_predictions.jsonl'
                rows=read_journal(journal,{c['example_id'] for c in cases})
                if len(rows)<500:
                    seed_all(42);status('load_'+arm,completed=len(rows),total=500)
                    model,_,_=build_model(s,codec,path);model.eval()
                    with journal.open('a') as stream:
                        for case,example in zip(cases,samples):
                            if case['example_id'] in rows:continue
                            row={'id':case['example_id'],'task':'generation','sequence':None,'error':None}
                            started=time.monotonic()
                            try:
                                # Only the instruction-derived prefix enters inference.
                                row['sequence'],row['generated_ids']=generate_one(model,
                                    example['input_ids'][:example['prompt_length']],codec,s)
                            except ValueError as exc:row['error']=str(exc)
                            row['seconds']=time.monotonic()-started
                            stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
                            rows[row['id']]=row
                            status(arm+'_inference',completed=len(rows),total=500)
                            print(json.dumps({'arm':arm,'completed':len(rows),'total':500}),flush=True)
                    del model;gc.collect();torch.cuda.empty_cache()
                arms[arm]=[rows[c['example_id']] for c in cases]
            cfg={**read_config('configs/benchmark.json'),'output':str(ROOT/'geometry-cache'),
                'render':False,'point_count':2000,'seed':42}
            runner=KernelRunner(cfg)
            jobs=[]
            for i,c in enumerate(cases):
                jobs.append((c['example_id'],'reference',c['target']))
                jobs.extend((c['example_id'],'prediction',arms[arm][i]['sequence']) for arm in arms)
            jobs=list(dict.fromkeys(jobs));geometry={}
            with ThreadPoolExecutor(max_workers=8) as pool:
                for i,(job,value) in enumerate(zip(jobs,pool.map(lambda j:runner.reconstruct(j[2],j[0],j[1]),jobs)),1):
                    geometry[job]=value;status('geometry',completed=i,total=len(jobs))
            summaries={}
            for arm,rows in arms.items():
                for c,row in zip(cases,rows):
                    ref=geometry[(row['id'],'reference',c['target'])];pred=geometry[(row['id'],'prediction',row['sequence'])]
                    row.update(reference_geometry=ref,prediction_geometry=pred,chamfer=None,relative_volume_error=None,
                        exact_sem=row['sequence']==stringify(tokenize(c['target'])),
                        exact_command_types=row['sequence'] is not None and [x.split(',')[0] for x in row['sequence'].split()]==[x.split(',')[0] for x in c['target'].split()])
                    if ref['status']==pred['status']=='valid':
                        a,b=[np.asarray(normalize_points(np.load(g['points']))) for g in (ref,pred)]
                        row['chamfer']=float(np.mean(cKDTree(b).query(a)[0]**2)+np.mean(cKDTree(a).query(b)[0]**2))
                        row['relative_volume_error']=abs(pred['volume']-ref['volume'])/ref['volume']
                valid=[r for r in rows if r['chamfer'] is not None]
                summaries[arm]={'cases':len(rows),'syntax_decoded':sum(r['sequence'] is not None for r in rows),
                    'known_valid_references':sum(r['reference_geometry']['status']=='valid' for r in rows),
                    'valid_predictions':sum(r['prediction_geometry']['status']=='valid' for r in rows),
                    'valid_of_valid_references':len(valid),'exact_sem':sum(r['exact_sem'] for r in rows),
                    'exact_command_types':sum(r['exact_command_types'] for r in rows),
                    'mean_chamfer':statistics.mean(r['chamfer'] for r in valid) if valid else None,
                    'median_volume_error':statistics.median(r['relative_volume_error'] for r in valid) if valid else None,
                    'count_f1':cadfusion_f1([(c['target'],r['sequence']) for c,r in zip(cases,rows)]),
                    'all_sample_count_f1':all_sample_count_f1([(c['target'],r['sequence']) for c,r in zip(cases,rows)])}
            pairs=[(a,b) for a,b in zip(arms['parent'],arms['expert']) if a['chamfer'] is not None and b['chamfer'] is not None]
            paired={'shared_valid_cases':len(pairs),'parent_mean_chamfer':statistics.mean(a['chamfer'] for a,b in pairs) if pairs else None,
                'expert_mean_chamfer':statistics.mean(b['chamfer'] for a,b in pairs) if pairs else None,
                'expert_lower_error_cases':sum(b['chamfer']<a['chamfer'] for a,b in pairs),
                'expert_only_valid_on_valid_reference':sum(a['reference_geometry']['status']=='valid' and a['chamfer'] is None and b['chamfer'] is not None for a,b in zip(arms['parent'],arms['expert'])),
                'parent_only_valid_on_valid_reference':sum(a['chamfer'] is not None and b['chamfer'] is None for a,b in zip(arms['parent'],arms['expert']))}
            passed=bool(pairs and summaries['expert']['valid_predictions']>=summaries['parent']['valid_predictions']
                and summaries['expert']['valid_of_valid_references']>=summaries['parent']['valid_of_valid_references']
                and paired['expert_mean_chamfer']<paired['parent_mean_chamfer']
                and summaries['expert']['exact_command_types']>=summaries['parent']['exact_command_types'])
            result={'summaries':summaries,'paired':paired,'passes_frozen_validation_screen':passed,
                'decision_rule':manifest['decision_rule'],'audit':audit,'production_promoted':False,'limits':manifest['limits']}
            write_json(ROOT/'per_case.json',arms);write_json(ROOT/'results.json',result)
            (ROOT/'findings.md').write_text('# Generation specialist: expanded validation\n\n'+json.dumps(result,indent=2)+'\n')
            write_json(ROOT/'state.json',{'status':'complete','results':str(ROOT/'results.json')})
        except Exception as exc:
            write_json(ROOT/'state.json',{'status':'failed','error':repr(exc)});raise
