"""Fixed batched evaluation and reference-edit metrics for numeric supervision."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import statistics

from .common import write_json,read_config
from .numeric_supervision import changed_numbers,command_origins


def edit_metrics(source,target,prediction):
    from scripts.analyze_edit_correctness import align
    a,b,p=source.split(),target.split(),(prediction or '').split()
    flags=changed_numbers(source,target);origins=command_origins(p,b);correct=required=offset=0
    for command,origin in zip(b,origins):
        fields=command.split(',');values=p[origin].split(',') if origin is not None else []
        local=flags[offset:offset+len(fields)];offset+=len(fields)
        for i,wanted in enumerate(local):
            if wanted:
                required+=1
                correct+=bool(values and values[0]==fields[0] and i<len(values) and values[i]==fields[i])
    _,_,unchanged=align(a,b);_,_,preserved=align(a,p)
    return {'correct_changed_numbers':correct,'required_changed_numbers':required,
        'unchanged_source_commands':len(unchanged),'preserved_unchanged_commands':len(unchanged&preserved),
        'target_command_distance':align(b,p)[0],'source_target_command_distance':align(a,b)[0],
        'copies_source':prediction==source,'exact_sem':prediction==target}


def summarize(rows):
    valid=[r for r in rows if r.get('chamfer') is not None]
    required=sum(r['required_changed_numbers'] for r in rows)
    unchanged=sum(r['unchanged_source_commands'] for r in rows)
    result={'cases':len(rows),'exact_sem':sum(r['exact_sem'] for r in rows),'copies_source':sum(r['copies_source'] for r in rows),
        'changed_numeric_accuracy':sum(r['correct_changed_numbers'] for r in rows)/required if required else None,
        'unchanged_command_preservation':sum(r['preserved_unchanged_commands'] for r in rows)/unchanged if unchanged else None,
        'mean_target_command_distance':statistics.mean(r['target_command_distance'] for r in rows) if rows else None}
    if rows and 'prediction_geometry' in rows[0]:
        result.update(valid_predictions=sum(r['prediction_geometry']['status']=='valid' for r in rows),
            known_valid_references=sum(r['reference_geometry']['status']=='valid' for r in rows),
            valid_of_valid_references=len(valid),mean_chamfer=statistics.mean(r['chamfer'] for r in valid) if valid else None)
    return result


def evaluate_numeric(model,data,codec,s,root,arm_root):
    import torch
    import numpy as np
    from scipy.spatial import cKDTree
    from .batch_decode import generate_many
    from .lossless_sft import load_samples
    from .lossless_editing_eval import read_journal
    from .bench_runtime import KernelRunner
    from .bench_metrics import normalize_points
    sets={'validation':data['validation'],'train_recall':data['recall']}
    cases=[{**c,'set':name} for name,items in sets.items() for c in items]
    raw=load_samples(s,cases)
    entries=sorted(zip(cases,raw),key=lambda pair:(len(pair[1]['input_ids'][:pair[1]['prompt_length']]),pair[0]['example_id']))
    journal=arm_root/'predictions.jsonl';existing=read_journal(journal,{c['example_id'] for c in cases})
    model.zero_grad(set_to_none=True);model.eval();torch.cuda.empty_cache()
    with journal.open('a') as stream:
        for i in range(0,len(entries),s['evaluation_batch_size']):
            chunk=entries[i:i+s['evaluation_batch_size']]
            if all(c['example_id'] in existing for c,x in chunk):continue
            rows=generate_many(model,[x['input_ids'][:x['prompt_length']] for c,x in chunk],codec,s)
            for (c,x),row in zip(chunk,rows):
                key=c['example_id']
                if key in existing:
                    if row['generated_ids']!=existing[key]['generated_ids']:raise ValueError('Resumed evaluation batch changed outputs')
                    continue
                row.update(id=key,set=c['set']);existing[key]=row
                stream.write(json.dumps(row)+'\n');stream.flush()
            write_json(root/'state.json',{'status':'running','arm':arm_root.name,'phase':'batched_evaluation',
                'completed':len(existing),'total':len(cases),'batch_size':s['evaluation_batch_size']})
    rows=[{**existing[c['example_id']],**edit_metrics(c['source'],c['target'],existing[c['example_id']]['sequence'])} for c in cases]
    cfg={**read_config('configs/benchmark.json'),'output':str(root/'geometry-cache'),'render':False,'seed':42,'point_count':2000}
    runner=KernelRunner(cfg);validation=[(c,r) for c,r in zip(cases,rows) if c['set']=='validation']
    jobs=list(dict.fromkeys([(c['example_id'],'reference',c['target']) for c,r in validation]+[(r['id'],'prediction',r['sequence']) for c,r in validation]))
    geometry={}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,(job,value) in enumerate(zip(jobs,pool.map(lambda j:runner.reconstruct(j[2],j[0],j[1]),jobs)),1):
            geometry[job]=value
            write_json(root/'state.json',{'status':'running','arm':arm_root.name,'phase':'geometry','completed':i,'total':len(jobs)})
    for c,r in validation:
        ref=geometry[(r['id'],'reference',c['target'])];pred=geometry[(r['id'],'prediction',r['sequence'])]
        r.update(reference_geometry=ref,prediction_geometry=pred,chamfer=None)
        if ref['status']==pred['status']=='valid':
            a,b=[np.asarray(normalize_points(np.load(g['points']))) for g in (ref,pred)]
            r['chamfer']=float(np.mean(cKDTree(b).query(a)[0]**2)+np.mean(cKDTree(a).query(b)[0]**2))
    summaries={name:summarize([r for r in rows if r['set']==name]) for name in sets}
    by_verb={}
    for c,r in validation:
        match=re.search(r'\w+',c['instruction']);verb=match.group().lower() if match else '(empty)'
        by_verb.setdefault(verb,[]).append(r)
    summaries['validation_by_first_word']={verb:summarize(group) for verb,group in by_verb.items()}
    by_id={c['example_id']:c for c in data['recall']}
    summaries['recall_by_group']={group:summarize([r for r in rows if r['set']=='train_recall' and
        (by_id[r['id']]['group']==group or group=='add' and by_id[r['id']]['group'].startswith('add_'))])
        for group in ('add','add_numeric','add_no_numeric','remove')}
    write_json(arm_root/'per_case.json',rows);write_json(arm_root/'evaluation.json',summaries)
    return summaries


def compare_arms(root,arms):
    result={}
    treatment={r['id']:r for r in read_config(root/'weighted/per_case.json') if r['set']=='validation'}
    for name in ('control','parent'):
        control={r['id']:r for r in read_config(root/name/'per_case.json') if r['set']=='validation'}
        if set(control)!=set(treatment):raise ValueError('Comparison IDs differ')
        pairs=[(control[k],treatment[k]) for k in control if control[k]['chamfer'] is not None and treatment[k]['chamfer'] is not None]
        before=arms[name]['evaluation'];after=arms['weighted']['evaluation']
        means={'control':statistics.mean(a['chamfer'] for a,b in pairs) if pairs else None,
               'weighted':statistics.mean(b['chamfer'] for a,b in pairs) if pairs else None}
        adds_before=before['validation_by_first_word'].get('add',{}).get('exact_sem',0)
        adds_after=after['validation_by_first_word'].get('add',{}).get('exact_sem',0)
        passes=bool(pairs and adds_after>adds_before
            and after['validation']['valid_predictions']>=before['validation']['valid_predictions']
            and after['validation']['valid_of_valid_references']>=before['validation']['valid_of_valid_references']
            and after['validation']['unchanged_command_preservation']>=before['validation']['unchanged_command_preservation']
            and means['weighted']<=means['control'])
        result[name]={'shared_valid_cases':len(pairs),'paired_mean_chamfer':means,
            'exact_add_targets_before':adds_before,'exact_add_targets_weighted':adds_after,'passes_screen':passes}
    return result
