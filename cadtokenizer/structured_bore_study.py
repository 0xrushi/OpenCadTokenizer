"""Freeze public structured inputs and independently grade executed intents."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .common import require_container,read_config,write_json,file_hash
from .structured_bore import SCHEMA,baseline,model_prompt,parse,same_intent,execute

ROOT=Path('artifacts/structured-bore-diagnostic-v1')


def prepare():
    require_container()
    import cadquery as cq
    from .step_bore_adapter import inventory,continued_cylinders
    ROOT.mkdir(exist_ok=True,parents=True)
    if (ROOT/'protocol.json').exists():
        for path,digest in read_config(ROOT/'protocol.json')['frozen_code'].items():
            if file_hash(path)!=digest:raise ValueError('Frozen structured interface changed')
        print('Structured inputs already frozen');return
    sources=[('main',Path('artifacts/step-bore-expansion-v1/selection.json'),Path('data/external/cad-preserve-bore-expansion-v0.1.0')),
             ('development',Path('data/external/cad-preserve-train-v0.1.0/pilot_records.json'),Path('data/external/cad-preserve-train-v0.1.0'))]
    groups={'main':[],'development':[]};private=[]
    for group,record_path,data in sources:
        records=read_config(record_path);records=records['records'] if isinstance(records,dict) else records
        for row in records:
            if row['split']!='train':raise ValueError('Only train records allowed')
            directory=ROOT/'inputs'/row['id'];directory.mkdir(exist_ok=True,parents=True)
            source=directory/'source.step';shutil.copyfile(data/row['input_file'],source)
            shape=cq.importers.importStep(str(source)).val();faces=inventory(shape);b=shape.BoundingBox()
            cylinders=continued_cylinders(faces)
            summary={'units':'mm','bounds_min_mm':[b.xmin,b.ymin,b.zmin],'bounds_max_mm':[b.xmax,b.ymax,b.zmax],
                'cylindrical_voids':[{'axis':[round(v,6) for v in c['axis']],
                    'axis_origin_mm':[round(v,6) for v in c['origin_mm']],'radius_mm':round(c['radius_mm'],6)} for c in cylinders],
                'toroidal_surface_present':any(f['type']=='TORUS' and f.get('concave') for f in faces)}
            for i,(gold,checks) in enumerate(zip(row['gold_files'],row['check_files'])):
                value=read_config(data/checks)['value']
                public={'instruction':row['prompt'],'requested_value_mm':value,'source_geometry':summary}
                item={'id':f"{row['id']}:{i}",'case_id':row['id'],'parameter_index':i,'source_step':str(source),'public':public}
                groups[group].append(item)
                if group=='main':private.append({'id':item['id'],'gold':str(data/gold),'checks':str(data/checks),'expected':baseline(public)})
    demos=[r['public'] for r in groups['development'] if r['parameter_index']==0]
    assert len(demos)==2 and len(groups['main'])==72 and len(groups['development'])==8
    for group,rows in groups.items():
        for row in rows:
            row['prompt']=model_prompt(row['public'],demos)
            row['prompt_sha256']=__import__('hashlib').sha256(row['prompt'].encode()).hexdigest()
        write_json(ROOT/(group+'_inputs.json'),rows)
    write_json(ROOT/'private_evaluation.json',private)
    write_json(ROOT/'schema.json',SCHEMA)
    files=['cadtokenizer/structured_bore.py','cadtokenizer/step_bore_adapter.py','cadtokenizer/step_adapter_study.py','tests/test_structured_bore.py']
    protocol={'schema':'structured-bore-diagnostic-v1','main_sources':18,'main_requests':72,'development_requests':8,
        'few_shot_demonstrations':2,'demo_ids':[r['id'] for r in groups['development'] if r['parameter_index']==0],
        'model':'existing original step-500 editing adapter, same frozen LLaMA-3-8B backbone/new CAD rows; no instruction-model substitution',
        'model_format':'Few-shot plain-text completion: input-only STEP summary, original public instruction and numeric requested value; JSON output.',
        'max_new_tokens':192,'context_budget':4096,'profile_batch_sizes':[4,8,16,32],
        'profile_selection':'Fastest throughput on 32 repeated development requests with 10% GPU memory headroom and exact token agreement with eight serial development outputs. Serial fallback if none qualify. No main-outcome selection.',
        'generation':'Greedy, BF16, default PyTorch SDPA with deterministic algorithms; no CAD grammar, JSON repair or parser fallback. Stop at a completed root object or EOS/192 tokens.',
        'evaluation':'Strict JSON/schema, exact input-derived intent (numeric tolerance 1e-6 mm), fresh execution of every valid predicted intent, unchanged two-sided BRep grader plus functional points.',
        'limits':'All cases are train-origin, two related procedural families. New task/prompt/context format for an SEM-trained model, not comparable to old CAD decoding accuracy. Source summary is input-only. No general-language or unsupported-operation claims.',
        'frozen_code':{p:file_hash(p) for p in files},'input_hashes':{str(ROOT/n):file_hash(ROOT/n) for n in ['main_inputs.json','development_inputs.json','private_evaluation.json','schema.json']},
        'selection_sha256':file_hash('artifacts/step-bore-expansion-v1/selection.json')}
    write_json(ROOT/'protocol.json',protocol)
    write_json(ROOT/'state.json',{'status':'prepared','main_requests':72,'development_requests':8})
    print('Frozen 72 main requests, eight development profile requests, two public-only demonstrations.',flush=True)


def evaluate_worker(job,output):
    require_container()
    import cadquery as cq
    from .step_adapter_study import grade
    value=read_config(job);source=cq.importers.importStep(value['source_step']).val()
    result={}
    try:
        candidate=execute(source,value['intent'])
        destination=Path(output).with_suffix('.step');cq.exporters.export(candidate,str(destination))
        gold=cq.importers.importStep(value['gold']).val()
        result={'executed':True,'grade':grade(candidate,gold,read_config(value['checks'])),'output_step':str(destination)}
    except Exception as exc:result={'executed':False,'error':type(exc).__name__+': '+str(exc)}
    write_json(output,result)


def evaluate(root=ROOT):
    root=Path(root)
    require_container();protocol=read_config(root/'protocol.json')
    for p,digest in {**protocol['frozen_code'],**protocol['input_hashes']}.items():
        if file_hash(p)!=digest:raise ValueError('Frozen evaluation inputs/contract changed')
    inputs=read_config(root/'main_inputs.json');answers={r['id']:r for r in read_config(root/'private_evaluation.json')}
    predictions={r['id']:r for r in read_config(root/'model_predictions.json')}
    if set(predictions)!=set(answers):raise ValueError('Predictions incomplete or mismatched')
    prepared=[];jobs=[]
    for row in inputs:
        answer=answers[row['id']]
        for arm in ['parser','model']:
            record={'id':row['id'],'case_id':row['case_id'],'arm':arm,'schema_valid':False,'intent_exact':False}
            try:
                intent=baseline(row['public']) if arm=='parser' else parse(predictions[row['id']]['text'])
                record.update(schema_valid=True,intent=intent,intent_exact=same_intent(intent,answer['expected']))
                directory=root/'geometry'/arm/row['case_id'];directory.mkdir(exist_ok=True,parents=True)
                request=directory/f"parameter-{row['parameter_index']}.request.json";output=request.with_name(f"parameter-{row['parameter_index']}.result.json")
                write_json(request,{'source_step':row['source_step'],'intent':intent,'gold':answer['gold'],'checks':answer['checks']})
                jobs.append((record,request,output))
            except Exception as exc:record['parse_error']=type(exc).__name__+': '+str(exc)
            prepared.append(record)
    def run(job):
        record,request,output=job
        try:
            process=subprocess.run([sys.executable,'-m','cadtokenizer.structured_bore_study','--job',str(request),'--output',str(output)],capture_output=True,text=True,timeout=120)
            if process.returncode or not output.exists():return {'executed':False,'error':process.stderr[-1500:]}
            return read_config(output)
        except subprocess.TimeoutExpired:return {'executed':False,'error':'120-second geometry timeout'}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(job,result) in enumerate(zip(jobs,pool.map(run,jobs)),1):
            job[0]['geometry']=result
            write_json(root/'state.json',{'status':'running','phase':'fresh_geometry_evaluation','completed':i,'total':len(jobs)})
    write_json(root/'per_case.json',prepared)
    summary={}
    for arm in ['parser','model']:
        rows=[r for r in prepared if r['arm']==arm]
        summary[arm]={'requests':len(rows),'schema_valid':sum(r['schema_valid'] for r in rows),
            'intent_exact':sum(r['intent_exact'] for r in rows),'executed':sum(r.get('geometry',{}).get('executed',False) for r in rows),
            'geometry_correct':sum(r.get('geometry',{}).get('grade',{}).get('passed',False) for r in rows),
            'joint_intent_and_geometry':sum(r['intent_exact'] and r.get('geometry',{}).get('grade',{}).get('passed',False) for r in rows),
            'parse_error_counts':dict(Counter(r['parse_error'] for r in rows if 'parse_error' in r))}
    results={'status':'complete','arms':summary,'model_profile':read_config(root/'profile_selection.json'),
        'model_generation':read_config(root/'model_run.json'),'training_started':False,'router_changed':False,
        'decision':'No model advantage over the exact literal parser on this explicit-input panel; keep parser/executor as baseline.' if summary['model']['joint_intent_and_geometry']<=summary['parser']['joint_intent_and_geometry'] else 'Investigate unexpected baseline failures before any deployment or training.',
        'limits':protocol['limits']}
    write_json(root/'results.json',results)
    files=[str(root/n) for n in ['protocol.json','model_predictions.json','model_run.json','profile_selection.json','profile.json','per_case.json','results.json']]
    files += list(protocol['frozen_code'])+[__file__,'cadtokenizer/structured_bore_inference.py']
    files += [str(p) for p in sorted((root/'geometry').rglob('*')) if p.is_file()]
    write_json(root/'manifest.json',{'image_id':os.environ.get('CADTOKENIZER_IMAGE_ID'),'hashes':{str(p):file_hash(p) for p in files}})
    write_json(root/'state.json',{'status':'complete','arms':summary});print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();evaluate_worker(args.job,args.output)
