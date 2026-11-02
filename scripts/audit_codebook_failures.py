#!/usr/bin/env python3
"""Diagnostic-only failure tracing and counterfactual probes on private copies."""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import copy
import itertools
import json
import linecache
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from cadtokenizer.common import file_hash, require_container, write_json
from cadtokenizer.geometry_worker import parse_sem


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def proper_cross(a, b, c, d):
    return orient(a,b,c)*orient(a,b,d)<0 and orient(c,d,a)*orient(c,d,b)<0


def loop_area(loop):
    if len(loop)==1 and loop[0]['type']=='circle':
        p=loop[0]['coordinates']; radius=(math.dist(p[:2],p[2:4])+math.dist(p[4:6],p[6:8]))/4
        return math.pi*radius*radius
    points=[c['coordinates'][:2] for c in loop]
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(points,points[1:]+points[:1])))/2


def flags_for(sequence):
    flags=[]
    for oi,op in enumerate(parse_sem(sequence)):
        p=op['parameters']
        def add(kind, **extra): flags.append({'kind':kind,'operation':oi,**extra})
        if p[0]>=p[1]: add('unordered_extrusion_bounds',lower=p[0],upper=p[1])
        if p[14]<=0: add('nonpositive_scale')
        if not any(p[11:14]): add('zero_normal')
        elif not any(cross(p[5:8],p[11:14])): add('parallel_or_zero_x_axis')
        if oi==0 and op['operation']!='add': add('initial_non_add_operation',value=op['operation'])
        for fi,face in enumerate(op['faces']):
            if len(face)>1 and max(range(len(face)),key=lambda i:loop_area(face[i]))!=0:
                add('outer_loop_not_largest_estimated_area',face=fi)
            for li,loop in enumerate(face):
                if len(loop)>1 and any(c['type']=='circle' for c in loop): add('circle_mixed_with_other_curves',face=fi,loop=li)
                for ci,curve in enumerate(loop):
                    start=curve['coordinates'][:2]; end=loop[(ci+1)%len(loop)]['coordinates'][:2]
                    if curve['type']=='line' and start==end: add('zero_length_line',face=fi,loop=li,curve=ci)
                    if curve['type']=='arc':
                        middle=curve['coordinates'][2:4]
                        if orient(start,middle,end)==0: add('degenerate_arc',face=fi,loop=li,curve=ci)
                if all(c['type']=='line' for c in loop):
                    points=[c['coordinates'][:2] for c in loop]; n=len(points)
                    if loop_area(loop)==0: add('zero_area_line_loop',face=fi,loop=li)
                    for i in range(n):
                        for j in range(i+1,n):
                            if j==i+1 or (i==0 and j==n-1): continue
                            if proper_cross(points[i],points[(i+1)%n],points[j],points[(j+1)%n]):
                                add('proper_line_loop_self_intersection',face=fi,loop=li,edges=[i,j])
    return flags


def serialize(operations):
    words=[]
    for op in operations:
        for face in op['faces']:
            for loop in face:
                for curve in loop:
                    words += [curve['type']+','+','.join(map(str,curve['coordinates'])),'<curve_end>']
                words.append('<loop_end>')
            words.append('<face_end>')
        words += ['<sketch_end>',op['operation']+','+','.join(map(str,op['parameters'])),'<extrude_end>']
    return ' '.join(words)


def probe_variant(sequence, kind):
    operations=copy.deepcopy(parse_sem(sequence)); changes=[]
    for oi,op in enumerate(operations):
        p=op['parameters']
        if kind in ('bounds','combined') and p[0]>=p[1]:
            before=p[:2]
            if p[0]>p[1]: p[:2]=sorted(p[:2])
            elif p[1]<63: p[1]+=1
            else: p[0]=62
            changes.append({'operation':oi,'field':'bounds','before':before,'after':p[:2]})
        if kind in ('initial_add','combined') and oi==0 and op['operation']!='add':
            changes.append({'operation':oi,'field':'boolean','before':op['operation'],'after':'add'});op['operation']='add'
        if kind in ('frame','combined') and (not any(p[11:14]) or not any(cross(p[5:8],p[11:14]))):
            before=p[5:8]+p[11:14]
            axes=list(itertools.product((-1,0,1),repeat=3))
            candidates=[(x,z) for x in axes for z in axes if any(z) and any(cross(x,z))]
            x,z=min(candidates,key=lambda v:sum((a-b)**2 for a,b in zip(before,list(v[0])+list(v[1]))))
            p[5:8]=x;p[11:14]=z
            changes.append({'operation':oi,'field':'frame_x_z','before':before,'after':list(x)+list(z)})
        for fi,face in enumerate(op['faces']):
            if kind in ('loop_order','combined') and len(face)>1:
                largest=max(range(len(face)),key=lambda i:loop_area(face[i]))
                if largest:
                    face.insert(0,face.pop(largest));changes.append({'operation':oi,'face':fi,'field':'outer_loop','before':largest,'after':0})
            if kind not in ('degenerate_curves','combined'): continue
            for li,loop in enumerate(face):
                for _ in range(len(loop)):
                    changed=False
                    for ci,curve in enumerate(loop):
                        start=curve['coordinates'][:2];end=loop[(ci+1)%len(loop)]['coordinates'][:2]
                        degenerate=(curve['type']=='line' and start==end) or (curve['type']=='arc' and orient(start,curve['coordinates'][2:4],end)==0)
                        if not degenerate: continue
                        before=copy.deepcopy(curve)
                        if start!=end: loop[ci]={'type':'line','coordinates':start}
                        else: loop.pop(ci)
                        changes.append({'operation':oi,'face':fi,'loop':li,'field':'degenerate_curve','before':before})
                        changed=True;break
                    if not changed or not loop: break
    return (serialize(operations),changes) if changes else (None,[])


def brep_statuses(shape):
    from OCP.BRepCheck import BRepCheck_Analyzer
    analyzer=BRepCheck_Analyzer(shape.wrapped)
    # IsValid also evaluates contextual topology checks before statuses are read.
    data={'valid':bool(analyzer.IsValid()),
          'face':[str(s).split('.')[-1] for s in analyzer.Result(shape.wrapped).Status()], 'wires':[], 'edges':[]}
    for wire in shape.Wires():
        result=analyzer.Result(wire.wrapped)
        item={'self':[str(s).split('.')[-1] for s in result.Status()]}
        try: item['on_face']=[str(s).split('.')[-1] for s in result.StatusOnShape(shape.wrapped)]
        except Exception as e: item['inspection_error']=str(e)
        data['wires'].append(item)
    for edge in shape.Edges():
        result=analyzer.Result(edge.wrapped)
        item={'self':[str(s).split('.')[-1] for s in result.Status()]}
        try: item['on_face']=[str(s).split('.')[-1] for s in result.StatusOnShape(shape.wrapped)]
        except Exception as e: item['inspection_error']=str(e)
        for wire in shape.Wires():
            try:
                contextual=[str(s).split('.')[-1] for s in result.StatusOnShape(wire.wrapped)]
                if any(s!='BRepCheck_NoError' for s in contextual):item.setdefault('on_wires',[]).extend(contextual)
            except Exception: pass
        data['edges'].append(item)
    return data


def stage_for(line, error):
    for fragment,name in [('lower plane','extrusion_bounds'),('initial extrusion','initial_boolean'),
                          ('normal','sketch_frame'),('x-axis','sketch_frame'),('scale must','sketch_scale'),
                          ('Invalid planar face','face_validity'),('Boolean operation produced','boolean_result')]:
        if fragment in error:return name
    for fragment,name in [('makeLine','line_construction'),('makeThreePointArc','arc_construction'),
                          ('assembleEdges','wire_assembly'),('makeFromWires','face_construction'),
                          ('extrudeLinear','extrusion_construction'),('.fuse(','boolean_fuse'),
                          ('.cut(','boolean_cut'),('.intersect(','boolean_intersect')]:
        if fragment in line:return name
    return 'mesh_export_or_other'


def traced_execute(sequence, settings, directory):
    from cadtokenizer import geometry_worker as geometry
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    failure={}; position={'operation':-1}
    targets=(geometry.build_solid.__code__,geometry.execute_request.__code__)
    def trace(frame,event,arg):
        if frame.f_code not in targets:return None
        line=linecache.getline(frame.f_code.co_filename,frame.f_lineno).strip()
        if event=='line' and line=="p = operation['parameters']":position['operation']+=1
        if event=='exception' and arg[0] not in (StopIteration, GeneratorExit) and not failure:
            error=str(arg[1]);local=frame.f_locals
            failure.update(stage=stage_for(line,error),source_line=line,line=frame.f_lineno,
                           function=frame.f_code.co_name,operation=position['operation'],error=error,
                           error_type=type(arg[1]).__name__)
            op=local.get('operation')
            if op:
                failure['operation_type']=op['operation'];failure['parameters']=op['parameters']
                if 'face' in local:failure['face_index']=next((i for i,f in enumerate(op['faces']) if f is local['face']),None)
                if 'curve' in local:
                    failure['curve']=local['curve'];failure['following_curve']=local.get('following')
            if 'Invalid planar face' in error and local.get('face_shape') is not None:
                try: failure['brep_check']=brep_statuses(local['face_shape'])
                except Exception as e:failure['brep_inspection_error']=str(e)
        return trace
    request={'sequence':sequence,'settings':{**settings,'render':False},'directory':str(directory),
             'sample_seed':42,'request_sha256':'diagnostic-only'}
    old=sys.gettrace();sys.settrace(trace)
    try:
        geometry.execute_request(request,directory/'result.json')
        result=json.loads((directory/'result.json').read_text())
        result['trace']=None
        return result
    except Exception as e:
        return {'status':'invalid','error':str(e),'error_type':type(e).__name__,'trace':failure}
    finally:sys.settrace(old)


def audit_one(request):
    root=Path(request['directory']);sequence=request['sequence'];settings=request['settings']
    baseline=traced_execute(sequence,settings,root/'baseline')
    row={'sample_id':request['sample_id'],'baseline':baseline,'flags':flags_for(sequence),'probes':{}}
    cache={}
    for kind in ('bounds','initial_add','frame','degenerate_curves','loop_order','combined'):
        changed,edits=probe_variant(sequence,kind)
        if changed is None:continue
        if changed not in cache:
            result=traced_execute(changed,settings,root/kind)
            if result['status']=='valid' and request.get('reference_geometry',{}).get('status')=='valid':
                import numpy as np
                from scipy.spatial import cKDTree
                from cadtokenizer.bench_metrics import normalize_points
                a=np.asarray(normalize_points(np.load(result['points']),settings['normalization']))
                b=np.asarray(normalize_points(np.load(request['reference_geometry']['points']),settings['normalization']))
                result['paired_cd_raw']=float(np.mean(cKDTree(b).query(a)[0]**2)+np.mean(cKDTree(a).query(b)[0]**2))
            cache[changed]=result
        row['probes'][kind]={'edits':edits,'result':cache[changed]}
    return row


def run_worker(request_path,result_path):
    request=json.loads(Path(request_path).read_text())
    write_json(result_path,audit_one(request))


def source_coverage(sample, config, tokenizer, boundaries, output_budget=256):
    from cadtokenizer.representation import analyze
    from cadtokenizer.instruction_data import instruction_tokens
    _,pairs=analyze(sample['source_sequence'],max_pair_tokens=100000,max_primitives=100000,
                    extrusion_partition=config['data']['extrusion_partition'])
    counts=[max(p['groups'])+1 for p in pairs]
    fake_pairs=[[0]*count for count in counts]
    prompt=instruction_tokens(tokenizer,boundaries,sample['instruction'],fake_pairs)
    return {'sample_id':sample['id'],'pairs':len(pairs),'primitive_counts':counts,
            'max_pair_tokens':max(len(p['tokens']) for p in pairs),'hypothetical_prompt_tokens':len(prompt),
            'within_encoder_token_limit':max(len(p['tokens']) for p in pairs)<=config['data']['max_pair_tokens'],
            'within_llm_context':len(prompt)+output_budget<=config['sft']['context_length']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',default='artifacts/benchmark-vastai-codebook')
    parser.add_argument('--output',default='artifacts/codebook-failure-audit')
    parser.add_argument('--request');parser.add_argument('--result')
    args=parser.parse_args();require_container()
    if args.request:
        run_worker(args.request,args.result);return
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    source=Path(args.benchmark)
    files=['per_sample.jsonl','predictions.jsonl','manifest.json','metrics.json']
    signature={name:file_hash(source/name) for name in files}
    signature['audit_script']=file_hash(__file__)
    marker=root/'inputs.json'
    if marker.exists() and json.loads(marker.read_text())!=signature:raise ValueError('Audit inputs changed; choose a fresh output')
    write_json(marker,signature)
    cases=[json.loads(line) for line in (source/'per_sample.jsonl').read_text().splitlines()]
    predictions={r['sample_id']:r for r in map(json.loads,(source/'predictions.jsonl').read_text().splitlines())}
    samples={r['id']:r for r in json.loads((source/'manifest.json').read_text())['samples']}
    metrics=json.loads((source/'metrics.json').read_text());settings=metrics['settings']
    failures=[r for r in cases if r['prediction_geometry']['status']!='valid']
    primary=collections.Counter();by_task={};static=[]
    for case in cases:
        key=case['sample_id'];p=predictions[key]
        if case in failures:primary[(case['task'],case['prediction_geometry'].get('error',''))]+=1
        if p['status']=='syntax_valid':
            static.append({'sample_id':key,'valid_solid':case['prediction_geometry']['status']=='valid',
                           'flags':flags_for(p['generated_sequence'])})
    jobs=[]
    for case in failures:
        key=case['sample_id']
        if case['prediction_geometry']['status']!='invalid':continue
        directory=root/'cases'/key.replace(':','_');directory.mkdir(parents=True,exist_ok=True)
        request={'sample_id':key,'sequence':predictions[key]['generated_sequence'],'settings':settings,
                 'directory':str(directory),'reference_geometry':case['reference_geometry']}
        request_path=directory/'request.json';write_json(request_path,request)
        jobs.append((request_path,directory/'audit.json'))
    def execute(job):
        request_path,result_path=job
        if result_path.exists():return json.loads(result_path.read_text())
        env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        try:
            result=subprocess.run([sys.executable,__file__,'--request',str(request_path),'--result',str(result_path)],
                                  capture_output=True,text=True,timeout=120,env=env)
        except subprocess.TimeoutExpired:
            return {'sample_id':json.loads(request_path.read_text())['sample_id'],'audit_error':'timeout'}
        if result.returncode!=0:raise RuntimeError(f'Audit subprocess failed: {request_path}\n{result.stderr[-3000:]}')
        return json.loads(result_path.read_text())
    audits=[]
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i,row in enumerate(pool.map(execute,jobs),1):
            audits.append(row)
            if i%10==0:print(json.dumps({'audited_kernel_failures':i,'total':len(jobs)}),flush=True)
    from transformers import AutoTokenizer
    config=json.loads((source/'predictions.meta.json').read_text())['signature']['model_config']
    tokenizer=AutoTokenizer.from_pretrained(Path(config['alignment']['output'])/'llm_tokenizer',local_files_only=True)
    # Framing affects token counts; recover IDs without loading any model weights.
    names=tokenizer.get_vocab();reserved=sorted(v for k,v in names.items() if 'reserved_special_token_' in k)
    boundaries=dict(zip(('start','pair_end','end'),reserved[:3]))
    coverage=[]
    for case in failures:
        if 'Primitive count' in case['prediction_geometry'].get('error',''):
            coverage.append(source_coverage(samples[case['sample_id']],config,tokenizer,boundaries,settings['max_new_tokens']))
    stages=collections.Counter();statuses=collections.Counter();probe_counts={};recovered=set();reproduced=0
    for row in audits:
        if row.get('audit_error'):continue
        key=row['sample_id'];old=next(c for c in failures if c['sample_id']==key)['prediction_geometry']
        baseline=row['baseline'];reproduced+=baseline['status']=='invalid' and baseline['error']==old['error']
        trace=baseline.get('trace') or {};stages[trace.get('stage','not_reproduced')]+=1
        check=trace.get('brep_check') or {}
        found=set(check.get('face',[]))
        for wire in check.get('wires',[]):found.update(wire.get('self',[]));found.update(wire.get('on_face',[]))
        for edge in check.get('edges',[]):
            for field in ('self','on_face','on_wires'):found.update(edge.get(field,[]))
        statuses.update(s for s in found if s!='BRepCheck_NoError')
        for kind,probe in row['probes'].items():
            c=probe_counts.setdefault(kind,{'attempted':0,'valid_after_probe':0,'sample_ids':[],'paired_cd':[]})
            c['attempted']+=1
            if probe['result']['status']=='valid':
                c['valid_after_probe']+=1;c['sample_ids'].append(key);recovered.add(key)
                if 'paired_cd_raw' in probe['result']:c['paired_cd'].append(probe['result']['paired_cd_raw'])
    for value in probe_counts.values():
        cds=value.pop('paired_cd');value['median_reference_cd']=statistics.median(cds) if cds else None
    guard_stats={}
    for row in static:
        for kind in {flag['kind'] for flag in row['flags']}:
            stat=guard_stats.setdefault(kind,{'flagged_invalid':0,'flagged_valid':0})
            stat['flagged_valid' if row['valid_solid'] else 'flagged_invalid']+=1
    report={'schema':'cad-failure-audit-v1','inputs':signature,'cases':len(cases),'failures':len(failures),
            'kernel_failures':len(jobs),'input_failures':len(failures)-len(jobs),'kernel_errors_reproduced':reproduced,
            'primary_errors':[{'task':k[0],'error':k[1],'count':v} for k,v in primary.most_common()],
            'kernel_stages':dict(stages),'brep_status_counts':dict(statuses),'static_guard_counts':guard_stats,
            'repair_probes':probe_counts,'distinct_probe_valid_cases':len(recovered),
            'source_limit_cases':coverage,
            'source_coverage_summary':{'cases':len(coverage),'within_encoder_tokens_and_context':sum(r['within_encoder_token_limit'] and r['within_llm_context'] for r in coverage)},
            'limits':['Diagnostic probes modify private copies only; original predictions, scores, decoder and weights are unchanged.',
                      'Recovered validity is not demonstrated instruction correctness. Arc straightening and forcing add may change design intent.',
                      'Probe counts overlap and must not be added. First-error categories can hide subsequent failures.',
                      'Input-limit prompt counts are hypothetical coverage, not tested model quality beyond the training primitive limit.']}
    write_json(root/'report.json',report);write_json(root/'kernel_audits.json',audits);write_json(root/'static_flags.json',static)
    lines=['# Exact-codebook remaining-failure audit','',f'{len(failures)} failures / {len(cases)} cases: {len(jobs)} kernel failures and {len(failures)-len(jobs)} input failures.','',
           '| Kernel stage | Cases |','| --- | ---: |']
    lines += [f'| {kind} | {n} |' for kind,n in stages.most_common()]
    lines += ['','## Isolated counterfactual probes','','These are not updated benchmark scores.','',
              '| Probe | Attempted | Kernel-valid afterward | Median reference CD |','| --- | ---: | ---: | ---: |']
    lines += [f'| {kind} | {v["attempted"]} | {v["valid_after_probe"]} | {v["median_reference_cd"]} |' for kind,v in probe_counts.items()]
    lines += ['',*report['limits']]
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    write_json(root/'complete.json',{'status':'complete','summary':{k:v for k,v in report.items() if k not in ('source_limit_cases','primary_errors')}})
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_limit_cases','inputs')},indent=2),flush=True)


if __name__=='__main__':main()
