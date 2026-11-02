"""Train-only STEP adapter check, with answers confined to the evaluator path."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .common import read_config,write_json,file_hash,require_container

ROOT=Path('artifacts/step-bore-adapter-v1')
DATA=Path('data/external/cad-preserve-train-v0.1.0')


def grade(shape,gold,checks):
    import cadquery as cq
    valid=bool(shape.isValid() and len(shape.Solids())==1 and shape.Volume()>0)
    extra=float(shape.cut(gold).Volume());missing=float(gold.cut(shape).Volume())
    tolerance=max(1e-5,gold.Volume()*1e-8)
    probes=[{'name':p['name'],'expected':p['material'],'observed':bool(shape.isInside(cq.Vector(*p['point']),1e-7))} for p in checks['probes']]
    a,b=shape.BoundingBox(),gold.BoundingBox()
    bounds_delta=max(abs(getattr(a,k)-getattr(b,k)) for k in ['xmin','xmax','ymin','ymax','zmin','zmax'])
    return {'passed':valid and extra<=tolerance and missing<=tolerance and all(p['expected']==p['observed'] for p in probes),
        'valid_single_solid':valid,'extra_material_mm3':extra,'missing_material_mm3':missing,
        'per_direction_tolerance_mm3':tolerance,'bounds_max_abs_error_mm':bounds_delta,'functional_probes':probes}


def run_study():
    require_container()
    import cadquery as cq
    from .step_bore_adapter import edit,public_request,make_editor
    ROOT.mkdir(exist_ok=True,parents=True)
    if (ROOT/'manifest.json').exists():
        frozen=read_config(ROOT/'manifest.json')
        if any(file_hash(p)!=v for p,v in frozen['hashes'].items()):raise ValueError('Frozen adapter provenance changed; use a new version')
        print(json.dumps(read_config(ROOT/'results.json'),indent=2));return
    download=read_config(DATA/'download_manifest.json')
    for item in download['files']:
        if file_hash(DATA/item['path'])!=item['sha256']:raise ValueError('Downloaded asset changed')
    records=read_config(DATA/'pilot_records.json')
    if len(records)!=2 or any(r['split']!='train' for r in records):raise ValueError('Only the two frozen train cases are allowed')
    write_json(ROOT/'state.json',{'status':'running','phase':'source_only_execution'})
    jobs=[]
    for row in records:
        directory=ROOT/'inputs'/row['id'];directory.mkdir(exist_ok=True,parents=True)
        source=directory/'source.step';shutil.copyfile(DATA/row['input_file'],source)
        for index,check_path in enumerate(row['check_files']):
            value=read_config(DATA/check_path)['value'];request=directory/f'parameter-{index}.json'
            write_json(request,{'source_step':str(source),'instruction':row['prompt'],'value_mm':value})
            output=ROOT/'outputs'/row['id']/f'parameter-{index}.step'
            jobs.append((row,index,request,output))
    def execute(job):
        row,index,request,output=job;start=time.monotonic()
        try:
            done=subprocess.run([sys.executable,'-m','cadtokenizer.step_bore_adapter','--request',str(request),'--output',str(output)],capture_output=True,text=True,timeout=120)
            if done.returncode or not output.exists():return {'success':False,'stderr':done.stderr[-2500:],'seconds':time.monotonic()-start}
            return {'success':True,'seconds':time.monotonic()-start}
        except subprocess.TimeoutExpired:return {'success':False,'stderr':'120-second timeout','seconds':time.monotonic()-start}
    execution=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index,(job,result) in enumerate(zip(jobs,pool.map(execute,jobs)),1):
            row,parameter,request,output=job
            execution.append({'id':row['id'],'parameter_index':parameter,'request':str(request),'output':str(output),**result})
            write_json(ROOT/'state.json',{'status':'running','phase':'source_only_execution','completed':index,'total':len(jobs)})
    write_json(ROOT/'execution.json',execution)
    evaluated=[];controls=[];invariance=[]
    for job,run in zip(jobs,execution):
        row,index,request,output=job;checks=read_config(DATA/row['check_files'][index])
        source=cq.importers.importStep(str(ROOT/'inputs'/row['id']/'source.step')).val()
        gold=cq.importers.importStep(str(DATA/row['gold_files'][index])).val()
        result={'id':row['id'],'parameter_index':index,'value_mm':checks['value'],'execution':run}
        if run['success']:
            candidate=cq.importers.importStep(str(output)).val();result['grade']=grade(candidate,gold,checks)
            naive,_=edit(source,checks['value'],prompt=row['prompt'],preserve_passages=False)
            controls.append({'id':row['id'],'parameter_index':index,'copy':grade(source,gold,checks),
                'naive_plug_without_passage_continuation':grade(naive,gold,checks)})
        evaluated.append(result)
        if index==0:
            old=public_request(row['prompt'],checks['value'])
            identity=make_editor(row['prompt'])(source,old['old_diameter_mm'])
            identity_grade=grade(identity,source,{'probes':[]})
            # A shifted source/prompt defeats case-specific origin/height lookups.
            shift=[7.,-3.,11.];xy=old['center_xy_mm']
            import re
            shifted_prompt=re.sub(r'\(x,y\)=\([\d.]+,\s*[\d.]+\)',f'(x,y)=({xy[0]+shift[0]},{xy[1]+shift[1]})',row['prompt'])
            changed=make_editor(shifted_prompt)(source.translate(cq.Vector(*shift)),checks['value'])
            shifted_checks={'probes':[{**p,'point':[a+b for a,b in zip(p['point'],shift)]} for p in checks['probes']]}
            shifted_grade=grade(changed,gold.translate(cq.Vector(*shift)),shifted_checks)
            invariance.append({'id':row['id'],'identity':identity_grade,'translation':shifted_grade})
    write_json(ROOT/'per_case.json',evaluated);write_json(ROOT/'controls.json',controls);write_json(ROOT/'invariance.json',invariance)
    passed=sum(r.get('grade',{}).get('passed',False) for r in evaluated)
    invariants=all(r[k]['passed'] for r in invariance for k in ('identity','translation'))
    copy_rejected=sum(not r['copy']['passed'] for r in controls)
    naive_rejected=sum(not r['naive_plug_without_passage_continuation']['passed'] for r in controls)
    results={'status':'complete' if passed==8 and invariants and copy_rejected==8 and naive_rejected==6 else 'needs_review',
        'source_cases':2,'parameter_targets':8,'adapter_passed':passed,'identity_and_translation_checks_passed':sum(r[k]['passed'] for r in invariance for k in ('identity','translation')),
        'copy_control_rejected':copy_rejected,'naive_plug_control_rejected':naive_rejected,
        'functional_probes_passed':sum(p['expected']==p['observed'] for r in evaluated for p in r.get('grade',{}).get('functional_probes',[])),
        'reference_programs_read_by_adapter':False,'reference_programs_executed':False,'model_execution':False,'training_started':False,
        'scope':'Deterministic source-surface bore edit on two already-selected training cases, not a learned model result or a general STEP editor.',
        'next':'Expand source-only adapter tests on additional training families before integrating a model-facing structured edit route.'}
    write_json(ROOT/'results.json',results)
    if results['status']=='complete':
        paths=[__file__,'cadtokenizer/step_bore_adapter.py','tests/test_step_bore_adapter.py',str(DATA/'download_manifest.json'),str(DATA/'pilot_records.json')]
        paths += [str(p) for p in sorted((ROOT/'inputs').rglob('*')) if p.is_file()]
        paths += [str(ROOT/n) for n in ['execution.json','per_case.json','controls.json','invariance.json','results.json']]
        paths += [str(p) for p in sorted((ROOT/'outputs').rglob('*')) if p.is_file()]
        write_json(ROOT/'manifest.json',{'image_id':os.environ.get('CADTOKENIZER_IMAGE_ID'),'dataset_revision':download['revision'],
            'hashes':{str(p):file_hash(p) for p in paths},'comparison':'two-sided BRep differences <= max(1e-5 mm3, reference volume*1e-8), functional probes, valid single solid; answer data accessed only by evaluator'})
    write_json(ROOT/'state.json',results)
    print(json.dumps(results,indent=2),flush=True)
