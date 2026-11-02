"""Read-only tracing of raw CAD-Editor numeric captions to SEM and solids."""
from concurrent.futures import ThreadPoolExecutor
import difflib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

from .common import read_config,write_json,file_hash,fraction,require_container
from .representation import tokenize,stringify,cross
from .geometry_worker import parse_sem

ROOT=Path('artifacts/numeric-instruction-audit-v1')


def deq(value,lo=-1.,hi=1.):return lo+(hi-lo)*value/63.


def unit(a):
    norm=math.sqrt(sum(x*x for x in a))
    if norm<1e-12:raise ValueError('Degenerate sketch axis')
    return [x/norm for x in a]


def describe(sequence):
    result=[]
    for index,op in enumerate(parse_sem(sequence)):
        p=op['parameters'];z=unit(p[11:14]);x=p[5:8];dot=sum(a*b for a,b in zip(x,z))
        x=unit([a-dot*b for a,b in zip(x,z)]);y=cross(z,x)
        translation=[deq(v) for v in p[2:5]];scale=deq(p[14],0.,1.4);offset=[deq(v,-.9,.9) for v in p[15:17]]
        low,high=map(deq,p[:2]);mid=(low+high)/2
        def world(point,plane):
            q=[deq(point[i])*scale+offset[i] for i in range(2)]
            return [translation[i]+q[0]*x[i]+q[1]*y[i]+plane*z[i] for i in range(3)]
        primitives=[]
        for fi,face in enumerate(op['faces']):
            for li,loop in enumerate(face):
                for ci,curve in enumerate(loop):
                    item={'face':fi,'loop':li,'curve':ci,'type':curve['type'],'raw_coordinates':curve['coordinates']}
                    if curve['type']=='circle':
                        points=[curve['coordinates'][i:i+2] for i in range(0,8,2)]
                        center=[(points[0][0]+points[1][0])/2,(points[2][1]+points[3][1])/2]
                        radius=(math.dist(points[0],points[1])+math.dist(points[2],points[3]))/4
                        item.update(raw_center=center,raw_radius=radius,raw_diameter=radius*2,
                            world_radius=radius*2/63*scale,world_axis_center=world(center,mid),
                            world_lower_center=world(center,low),world_upper_center=world(center,high),
                            boundary_role='outer' if li==0 else 'inner')
                    elif curve['type']=='arc':
                        a=curve['coordinates'][:2];b=curve['coordinates'][2:];c=loop[(ci+1)%len(loop)]['coordinates'][:2]
                        d=2*(a[0]*(b[1]-c[1])+b[0]*(c[1]-a[1])+c[0]*(a[1]-b[1]))
                        if abs(d)>1e-12:
                            aa=sum(v*v for v in a);bb=sum(v*v for v in b);cc=sum(v*v for v in c)
                            center=[(aa*(b[1]-c[1])+bb*(c[1]-a[1])+cc*(a[1]-b[1]))/d,
                                (aa*(c[0]-b[0])+bb*(a[0]-c[0])+cc*(b[0]-a[0]))/d]
                            radius=math.dist(a,center)
                            item.update(raw_arc_center=center,raw_arc_radius=radius,world_arc_radius=radius*2/63*scale)
                    primitives.append(item)
                if len(loop)==4 and all(c['type']=='line' for c in loop):
                    xx={c['coordinates'][0] for c in loop};yy={c['coordinates'][1] for c in loop}
                    if len(xx)==len(yy)==2:
                        dimensions=[max(xx)-min(xx),max(yy)-min(yy)]
                        primitives.append({'type':'axis_aligned_rectangle','face':fi,'loop':li,'raw_dimensions':dimensions,
                            'world_edge_lengths':[v*2/63*scale for v in dimensions]})
        result.append({'index':index,'operation':op['operation'],'raw_parameters':p,
            'raw_plane_bounds':p[:2],'raw_plane_span':p[1]-p[0],'raw_translation':p[2:5],
            'raw_scale_code':p[14],'raw_offset_codes':p[15:17],
            'world_plane_bounds':[low,high],'world_extrusion_height':high-low,'world_translation':translation,
            'world_scale':scale,'world_offset':offset,'world_axes':[x,y,z],'primitives':primitives})
    return result


def select_cases():
    pilot=read_config('artifacts/lossless-sft-pilot-v1/manifest.json');rows=[]
    with sqlite3.connect('data/prepared/cad.sqlite') as db:
        for c in pilot['train']:
            if c['task']!='editing':continue
            text,source,target,split=db.execute('SELECT e.instruction,a.sequence,b.sequence,e.split FROM examples e JOIN geometry a ON a.id=e.source_id JOIN geometry b ON b.id=e.target_id WHERE e.id=?',(c['example_id'],)).fetchone()
            if split!='train':raise ValueError('Audit must use training examples only')
            if text.lower().startswith(('add ','move ')) and re.search(r'\d',text):
                rows.append({**c,'instruction':text,'source':source,'target':target})
    selected=[]
    for verb,count in [('add',24),('move',8)]:
        candidates=sorted([r for r in rows if r['instruction'].lower().startswith(verb+' ')],
            key=lambda r:(fraction(r['example_id'],542),r['example_id']))
        if len(candidates)<count:raise ValueError('Insufficient numeric audit examples')
        selected.extend(candidates[:count])
    return selected


def solid_details(request_path,output_path):
    require_container()
    import cadquery as cq
    req=read_config(request_path)
    def info(shape):
        box=shape.BoundingBox()
        return {'volume':float(shape.Volume()),'solid_count':len(shape.Solids()),'bounds':{
            'min':[box.xmin,box.ymin,box.zmin],'max':[box.xmax,box.ymax,box.zmax],
            'lengths':[box.xlen,box.ylen,box.zlen]}}
    a=cq.importers.importStep(req['source']).val();b=cq.importers.importStep(req['target']).val()
    out={'source':info(a),'target':info(b)}
    try:out.update(added_material_volume=float(b.cut(a,tol=1e-5).Volume()),removed_material_volume=float(a.cut(b,tol=1e-5).Volume()))
    except Exception as exc:out['boolean_difference_error']=str(exc)
    write_json(output_path,out)


def run_audit():
    require_container()
    from .bench_runtime import KernelRunner
    from .bench_data import code_hash
    ROOT.mkdir(parents=True,exist_ok=True)
    write_json(ROOT/'state.json',{'status':'running','phase':'raw_record_checks'})
    cases=select_cases();raw_path=Path('data/extracted/cad_editor/dataset/train.json');raw=read_config(raw_path)
    manifest={'selection_seed':542,'cases':cases,'counts':{'add':24,'move':8},
        'source_database_sha256':file_hash('data/prepared/cad.sqlite'),'raw_train_sha256':file_hash(raw_path),
        'upstream_sources':read_config(ROOT/'upstream/sources.json'),'code_sha256':code_hash(),
        'image_id':os.environ.get('CADTOKENIZER_IMAGE_ID'),
        'limits':'Targeted 32 numeric training captions, not a representative dataset error-rate estimate. No model inference, training, test data or automatic dataset changes.'}
    if (ROOT/'manifest.json').exists() and read_config(ROOT/'manifest.json')!=manifest:raise ValueError('Audit provenance changed')
    write_json(ROOT/'manifest.json',manifest)
    rows=[]
    for case in cases:
        original=raw[int(case['example_id'].rsplit(':',1)[1])]
        if original['instruction'].strip()!=case['instruction']:raise ValueError('Raw instruction changed during preparation')
        if tokenize(original['original_sequence'])!=tokenize(case['source']) or tokenize(original['edited_sequence'])!=tokenize(case['target']):
            raise ValueError('Raw numeric/operation data changed during preparation')
        rows.append({**case,'raw_method':original.get('method'),'raw_type':original.get('type'),
            'raw_to_prepared_exact':True,'source_operations':describe(case['source']),'target_operations':describe(case['target'])})
    cfg={**read_config('configs/benchmark.json'),'output':str(ROOT/'geometry-cache'),'render':False,'point_count':64,'seed':542}
    runner=KernelRunner(cfg)
    jobs=[(c['example_id'],role,c[key]) for c in cases for role,key in [('source','source'),('reference','target')]]
    geometry={}
    write_json(ROOT/'state.json',{'status':'running','phase':'kernel_checks'})
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,(job,value) in enumerate(zip(jobs,pool.map(lambda j:runner.reconstruct(j[2],j[0],j[1]),jobs)),1):
            geometry[job]=value;write_json(ROOT/'state.json',{'status':'running','phase':'kernel_checks','completed':i,'total':len(jobs)})
    for row in rows:
        row['source_geometry']=geometry[(row['example_id'],'source',row['source'])]
        row['target_geometry']=geometry[(row['example_id'],'reference',row['target'])]
    def measure(row):
        if row['source_geometry']['status']!='valid' or row['target_geometry']['status']!='valid':return None
        name=row['example_id'].replace(':','_');directory=ROOT/'measurements';directory.mkdir(exist_ok=True)
        req=directory/(name+'.request.json');out=directory/(name+'.json')
        write_json(req,{'source':row['source_geometry']['step'],'target':row['target_geometry']['step']})
        if not out.exists():
            try:
                done=subprocess.run([sys.executable,'-m','cadtokenizer.instruction_audit','--request',str(req),'--output',str(out)],
                    capture_output=True,text=True,timeout=60)
                if done.returncode or not out.exists():return {'measurement_error':done.stderr[-1000:]}
            except subprocess.TimeoutExpired:return {'measurement_error':'Boolean difference timeout'}
        return read_config(out)
    write_json(ROOT/'state.json',{'status':'running','phase':'measure_material_changes'})
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row,measurement in zip(rows,pool.map(measure,rows)):row['solid_measurements']=measurement
    write_json(ROOT/'per_case.json',rows)
    packet=['# Numeric instruction audit: evidence','','Selection: 24 Add and 8 Move captions with numerals, from the existing training subset; seed 542. No model execution.','']
    for row in rows:
        packet+=['## '+row['example_id'],'',row['instruction'],'',
            'Raw metadata: '+str(row['raw_method'])+' / '+str(row['raw_type'])+'. Prepared numeric and operation tokens match upstream exactly.','',
            '```json',json.dumps({'source_operations':row['source_operations'],'target_operations':row['target_operations'],'solid_measurements':row['solid_measurements']},indent=2),'```','',
            '```diff','\n'.join(difflib.unified_diff(row['source'].split(),row['target'].split(),fromfile='source',tofile='target',lineterm='')),'```','']
    (ROOT/'evidence.md').write_text('\n'.join(packet))
    write_json(ROOT/'state.json',{'status':'evidence_complete','cases':len(rows),'classification':'pending evidence review'})


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();solid_details(args.request,args.output)
