"""Frozen-executor expansion to remaining supported training parts.

Solver execution and private grading run in separate subprocesses. Failures are
retained; this module never tunes the executor against the expanded targets.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from .common import require_container,read_config,write_json,file_hash

ROOT=Path('artifacts/step-bore-expansion-v1')
DATA=Path('data/external/cad-preserve-bore-expansion-v0.1.0')


def error(exc):return {'passed':False,'error':type(exc).__name__+': '+str(exc)}


def grading_worker(job,output):
    require_container()
    import cadquery as cq
    from .step_adapter_study import grade
    from .step_bore_adapter import edit,public_request,make_editor
    info=read_config(job);request=read_config(info['request']);checks=read_config(info['checks'])
    source=cq.importers.importStep(request['source_step']).val();gold=cq.importers.importStep(info['gold']).val()
    result={}
    if info['mode']=='parameter':
        try:
            candidate=cq.importers.importStep(info['candidate']).val()
            result['adapter']=grade(candidate,gold,checks)
        except Exception as exc:result['adapter']=error(exc)
        try:result['copy']=grade(source,gold,checks)
        except Exception as exc:result['copy']=error(exc)
        try:
            naive,_=edit(source,request['value_mm'],prompt=request['instruction'],preserve_passages=False)
            result['naive_plug']=grade(naive,gold,checks)
        except Exception as exc:result['naive_plug']=error(exc)
    else:
        try:
            old=public_request(request['instruction'],request['value_mm'])
            identity=make_editor(request['instruction'])(source,old['old_diameter_mm'])
            result['identity']=grade(identity,source,{'probes':[]})
        except Exception as exc:result['identity']=error(exc)
        try:
            shift=[7.,-3.,11.];xy=old['center_xy_mm']
            prompt=re.sub(r'\(x,y\)=\([\d.]+,\s*[\d.]+\)',f'(x,y)=({xy[0]+shift[0]},{xy[1]+shift[1]})',request['instruction'])
            candidate=make_editor(prompt)(source.translate(cq.Vector(*shift)),request['value_mm'])
            moved={'probes':[{**p,'point':[x+y for x,y in zip(p['point'],shift)]} for p in checks['probes']]}
            result['translation']=grade(candidate,gold.translate(cq.Vector(*shift)),moved)
        except Exception as exc:result['translation']=error(exc)
    write_json(output,result)


def run_expansion():
    require_container()
    ROOT.mkdir(exist_ok=True,parents=True)
    if (ROOT/'manifest.json').exists():
        frozen=read_config(ROOT/'manifest.json')
        if any(file_hash(p)!=v for p,v in frozen['hashes'].items()):raise ValueError('Frozen expansion provenance changed')
        print(json.dumps(read_config(ROOT/'results.json'),indent=2));return
    selected=read_config(ROOT/'selection.json');rows=selected['records']
    if len(rows)!=18 or any(r['split']!='train' for r in rows):raise ValueError('Selection is not the frozen 18 train cases')
    for p,digest in selected['frozen_code'].items():
        if file_hash(p)!=digest:raise ValueError('The adapter or grading contract changed since selection')
    old=read_config('artifacts/step-bore-adapter-v1/manifest.json')
    for p in ['cadtokenizer/step_bore_adapter.py','cadtokenizer/step_adapter_study.py']:
        if old['hashes'].get(p,old['hashes'].get('/workspace/'+p))!=file_hash(p):raise ValueError('Executor differs from the original successful pilot')
    downloaded=read_config(DATA/'download_manifest.json')
    for entry in downloaded['files']:
        if file_hash(DATA/entry['path'])!=entry['sha256']:raise ValueError('Downloaded asset changed')
    write_json(ROOT/'state.json',{'status':'running','phase':'staging_source_only_requests'})
    jobs=[];invariance=[]
    for row in rows:
        directory=ROOT/'inputs'/row['id'];directory.mkdir(parents=True,exist_ok=True)
        source=directory/'source.step';shutil.copyfile(DATA/row['input_file'],source)
        for i,(gold,checks) in enumerate(zip(row['gold_files'],row['check_files'])):
            request=directory/f'parameter-{i}.json';value=read_config(DATA/checks)['value']
            write_json(request,{'source_step':str(source),'instruction':row['prompt'],'value_mm':value})
            output=ROOT/'outputs'/row['id']/f'parameter-{i}.step'
            private=ROOT/'evaluation'/row['id']/f'parameter-{i}.request.json'
            evaluation=private.with_name(f'parameter-{i}.result.json')
            job={'id':row['id'],'family':row['family'],'parameter_index':i,'value_mm':value,
                'request':str(request),'candidate':str(output),'gold':str(DATA/gold),'checks':str(DATA/checks),
                'mode':'parameter','private_job':str(private),'grade_output':str(evaluation)}
            write_json(private,job);jobs.append(job)
            if i==0:
                private=private.with_name('invariance.request.json');result=private.with_name('invariance.result.json')
                inv={**job,'mode':'invariance','private_job':str(private),'grade_output':str(result)}
                write_json(private,inv);invariance.append(inv)
    def invoke(args):
        start=time.monotonic()
        try:
            process=subprocess.run(args,capture_output=True,text=True,timeout=120)
            return {'success':process.returncode==0,'seconds':time.monotonic()-start,'stderr':process.stderr[-2500:] if process.returncode else ''}
        except subprocess.TimeoutExpired:return {'success':False,'seconds':time.monotonic()-start,'stderr':'120-second timeout'}
    def execute(job):
        return invoke([sys.executable,'-m','cadtokenizer.step_bore_adapter','--request',job['request'],'--output',job['candidate']])
    executions=[];start=time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(job,result) in enumerate(zip(jobs,pool.map(execute,jobs)),1):
            executions.append({'id':job['id'],'parameter_index':job['parameter_index'],**result})
            write_json(ROOT/'state.json',{'status':'running','phase':'source_only_execution','completed':i,'total':len(jobs)})
    execution_seconds=time.monotonic()-start
    write_json(ROOT/'execution.json',executions)
    def measure(job):
        result=invoke([sys.executable,'-m','cadtokenizer.step_bore_expansion','--job',job['private_job'],'--output',job['grade_output']])
        if result['success'] and Path(job['grade_output']).exists():return {**read_config(job['grade_output']),'worker':result}
        return {'worker':result,'error':'Grading worker failed'}
    per_case=[];invariance_results=[]
    all_jobs=jobs+invariance;start=time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(job,result) in enumerate(zip(all_jobs,pool.map(measure,all_jobs)),1):
            record={'id':job['id'],'family':job['family'],'parameter_index':job['parameter_index'],'value_mm':job['value_mm'],**result}
            (per_case if job['mode']=='parameter' else invariance_results).append(record)
            write_json(ROOT/'state.json',{'status':'running','phase':'private_grading_and_controls','completed':i,'total':len(all_jobs)})
    grading_seconds=time.monotonic()-start
    write_json(ROOT/'per_case.json',per_case);write_json(ROOT/'invariance.json',invariance_results)
    failures=[r for r in per_case if not r.get('adapter',{}).get('passed')]
    invariance_pass=sum(r.get(k,{}).get('passed',False) for r in invariance_results for k in ('identity','translation'))
    control_errors=sum('error' in r.get(k,{}) or k not in r for r in per_case for k in ('copy','naive_plug'))
    copy_rejected=sum(not r.get('copy',{}).get('passed',True) for r in per_case)
    naive_rejected=sum(not r.get('naive_plug',{}).get('passed',True) for r in per_case)
    passed=len(per_case)-len(failures)
    gate=passed==72 and invariance_pass==36 and control_errors==0 and copy_rejected==72 and naive_rejected==54
    results={'status':'complete','adapter_gate_passed':gate,
        'new_train_sources':18,'parameter_targets':72,'adapter_passed':passed,'adapter_failed':len(failures),
        'by_family':{family:{'targets':sum(r['family']==family for r in per_case),
            'passed':sum(r['family']==family and r.get('adapter',{}).get('passed',False) for r in per_case)} for family in sorted({r['family'] for r in per_case})},
        'identity_and_translation_passed':invariance_pass,'identity_and_translation_total':36,
        'copy_control_rejected':copy_rejected,'naive_plug_control_rejected':naive_rejected,'expected_naive_rejected':54,
        'control_errors':control_errors,'functional_probes_passed':sum(p['expected']==p['observed'] for r in per_case for p in r.get('adapter',{}).get('functional_probes',[])),
        'functional_probes_total':sum(len(read_config(j['checks'])['probes']) for j in jobs),
        'execution_wall_seconds':execution_seconds,'grading_wall_seconds':grading_seconds,'workers':4,
        'solver_changed':False,'model_execution':False,'training_started':False,'validation_or_test_accessed':False,
        'scope':'All 18 remaining training parts in the same two procedural bore families; cases unused in executor development, but not unseen task families or independent model validation.',
        'next':'Prepare an input-only structured bore-edit interface and freeze a model diagnostic' if gate else 'Document expanded failures before changing solver; any fix and retest requires a new version and separate new cases.'}
    write_json(ROOT/'failures.json',failures);write_json(ROOT/'results.json',results)
    paths=[__file__,'scripts/prepare_step_adapter_expansion.py',str(ROOT/'selection.json'),str(DATA/'download_manifest.json')]
    paths += list(selected['frozen_code'])
    paths += [str(p) for folder in ['inputs','outputs','evaluation'] for p in sorted((ROOT/folder).rglob('*')) if p.is_file()]
    paths += [str(ROOT/p) for p in ['execution.json','per_case.json','invariance.json','failures.json','results.json']]
    write_json(ROOT/'manifest.json',{'image_id':os.environ.get('CADTOKENIZER_IMAGE_ID'),'dataset_revision':selected['revision'],
        'hashes':{str(p):file_hash(p) for p in paths},'frozen_solver_sha256':selected['frozen_code']['cadtokenizer/step_bore_adapter.py']})
    write_json(ROOT/'state.json',{'status':'complete','adapter_gate_passed':results['adapter_gate_passed'],'passed':passed,'total':72})
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();grading_worker(args.job,args.output)
