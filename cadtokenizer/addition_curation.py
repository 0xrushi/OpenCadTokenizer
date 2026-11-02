"""Verify explicit claims in existing numeric Add captions, without relabeling.

This is predicate verification, not certification of a unique target program.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

from .common import file_hash, fraction, read_config, require_container, write_json
from .numeric_curation import parse_program, isolated_sem

ROOT = Path('artifacts/addition-curation-v1')
CONTEXT = (
    'Diagnostic coordinate convention: stated two-dimensional positions are '
    'profile centers in the added operation\'s raw sketch grid. Profile lengths '
    'and radii use raw sketch-grid intervals; extrusion heights/depths use axial '
    'grid intervals. They are not millimetres and sketch scale still applies. '
    'Preserve all source operations and append exactly one stated additive solid '
    'or subtractive hole/cutout. Unspecified placement, scale and depth are not '
    'fixed by the caption. Correctness is scored by explicit claims and actual '
    'material change, not exact reproduction of an arbitrary reference program.'
)
RECT_NOUN = r'(?:(?:rectangular|square) )?(?P<kind>prism|block|base|solid|hole|cutout)'
CIRCLE_NOUN = r'(?:(?:circular|cylindrical) )?(?P<kind>hole|cutout|cylinder|circle)'
COORD = r'(?P<location>(?:centered at|at(?: position)?)) \((?P<x>\d+),\s*(?P<y>\d+)\)'
SIZE = r'(?P<dx>\d+)\s*[x×]\s*(?P<dy>\d+)(?:\s*[x×]\s*(?P<dz>\d+))?'
HEIGHT = r'(?:with |and )?(?:a )?(?:height|depth)(?: of)? (?P<height>\d+)(?: units?)?'
TAIL = r'(?: with a cut operation| subtractively)?\.'
PREFIX = r'Add (?:a |an )?(?:new )?'
RECT_PATTERNS = [re.compile(PREFIX+form+TAIL,re.I) for form in (
    SIZE+' '+RECT_NOUN+'(?: '+COORD+')?(?: '+HEIGHT+')?',
    RECT_NOUN+'(?: '+COORD+')? with (?:dimensions |size |a size of )?'+SIZE+r'(?: units?| dimensions| base)?(?: '+HEIGHT+')?',
    RECT_NOUN+' with (?:dimensions |size |a size of )?'+SIZE+r'(?: units?| dimensions| base)?(?: '+COORD+')?(?: '+HEIGHT+')?',
)]
CIRCLE_PATTERNS = [re.compile(PREFIX+form+TAIL,re.I) for form in (
    r'(?P<measure>\d+)[ -]unit (?P<metric>radius|diameter) '+CIRCLE_NOUN+'(?: '+COORD+')?',
    CIRCLE_NOUN+'(?: '+COORD+r')? with (?:a )?(?P<metric>radius|diameter)(?: of)? (?P<measure>\d+)(?: units?)?',
    CIRCLE_NOUN+r' with (?:a )?(?P<metric>radius|diameter)(?: of)? (?P<measure>\d+)(?: units?)?(?: '+COORD+')?',
)]


def caption_claims(instruction):
    for family, patterns in [('rectangle',RECT_PATTERNS),('circle',CIRCLE_PATTERNS)]:
        for pattern in patterns:
            if match := pattern.fullmatch(instruction):
                claims={k:v for k,v in match.groupdict().items() if v is not None}
                claims['family']=family
                return claims
    return None


def profile(sketch):
    """Independent closed profile parser; reject additional loops/faces/curves."""
    circle=re.fullmatch(r'circle,([\d,]+) <curve_end> <loop_end> <face_end> <sketch_end>',sketch)
    if circle:
        values=list(map(int,circle[1].split(',')))
        if len(values)!=8:return None
        a,b,c,d=[values[i:i+2] for i in range(0,8,2)]
        center=[(a[0]+b[0])/2,(c[1]+d[1])/2]
        ra=math.dist(a,b)/2;rb=math.dist(c,d)/2
        if (a[0]!=b[0] or c[1]!=d[1] or (a[1]+b[1])/2!=center[1]
                or (c[0]+d[0])/2!=center[0] or ra<=0 or ra!=rb):return None
        return {'family':'circle','center':center,'radius':ra,'raw_area':math.pi*ra*ra}
    coords=re.findall(r'line,(\d+),(\d+) <curve_end>',sketch)
    if len(coords)!=4:return None
    points=[tuple(map(int,v)) for v in coords]
    expected=' '.join('line,'+str(x)+','+str(y)+' <curve_end>' for x,y in points)+' <loop_end> <face_end> <sketch_end>'
    if expected!=sketch:return None
    xs={p[0] for p in points};ys={p[1] for p in points}
    if len(xs)!=2 or len(ys)!=2 or len(set(points))!=4:return None
    if any((a[0]!=b[0])+(a[1]!=b[1])!=1 for a,b in zip(points,points[1:]+points[:1])):return None
    dims=[max(xs)-min(xs),max(ys)-min(ys)]
    return {'family':'rectangle','center':[(min(xs)+max(xs))/2,(min(ys)+max(ys))/2],
            'dimensions':dims,'raw_area':dims[0]*dims[1]}


def verify(instruction,source,target):
    claims=caption_claims(instruction)
    if claims is None:return {'accepted':False,'reason':'outside_supported_grammar'}
    def reject(reason):return {'accepted':False,'reason':reason,'claims':claims}
    try:before,after=parse_program(source),parse_program(target)
    except ValueError:return reject('unsupported_program')
    if len(after)!=len(before)+1 or after[:-1]!=before:return reject('source_changed_or_not_single_append')
    operation=after[-1];p=operation['fields'];shape=profile(operation['sketch'])
    if shape is None or shape['family']!=claims['family']:return reject('profile_type_or_topology')
    boolean='cut' if claims['kind'].lower() in ('hole','cutout') else 'add'
    if boolean=='add' and re.search(r'cut operation|subtractively',instruction,re.I):return reject('conflicting_boolean_clause')
    if operation['operation']!=boolean:return reject('hole_vs_solid_boolean_mismatch')
    if not 0<=p[0]<p[1]<=63 or not 0<p[14]<=63:return reject('invalid_height_or_scale')
    if claims['family']=='rectangle':
        desired=[int(claims['dx']),int(claims['dy'])]
        if re.search(r'\bsquare\b',instruction,re.I) and desired[0]!=desired[1]:return reject('square_aspect_mismatch')
        if min(desired)<=0 or sorted(desired)!=sorted(shape['dimensions']):return reject('rectangle_dimensions_mismatch')
        if 'dz' in claims and p[1]-p[0]!=int(claims['dz']):return reject('extrusion_height_mismatch')
    else:
        desired=float(claims['measure'])/(2 if claims['metric'].lower()=='diameter' else 1)
        if desired<=0 or not math.isclose(desired,shape['radius'],abs_tol=1e-9):return reject('circle_radius_mismatch')
    if 'height' in claims and p[1]-p[0]!=int(claims['height']):return reject('extrusion_height_mismatch')
    if 'x' in claims and shape['center']!=[int(claims['x']),int(claims['y'])]:return reject('profile_center_mismatch')
    return {'accepted':True,'reason':'explicit_claims_match_declared_convention','claims':claims,
        'profile':shape,'operation_index':len(before),'boolean':boolean,'height_grid':p[1]-p[0],
        'profile_world_unit':(2/63)*(1.4*p[14]/63),'height_normalized':(p[1]-p[0])*2/63,
        'unique_reference_target_certified':False}


def geometry_check(request,output):
    require_container()
    import cadquery as cq
    from .geometry_worker import build_solid
    row=read_config(request);check=verify(row['instruction'],row['source'],row['target'])
    if not check['accepted']:raise ValueError('Caption/program failed verification')
    settings={'boolean_tolerance':1e-5,'minimum_volume':1e-10}
    a,b=[build_solid(row[k],settings) for k in ('source','target')]
    if any(not s.isValid() or len(s.Solids())!=1 for s in (a,b)):raise ValueError('Require valid single source/target solid')
    tool=build_solid(isolated_sem(row['target'],check['operation_index']),settings)
    fields=parse_program(row['target'])[-1]['fields'];z=fields[11:14];norm=math.sqrt(sum(v*v for v in z));z=[v/norm for v in z]
    caps=[];areas=[]
    for face in tool.Faces():
        if face.geomType()=='PLANE' and abs(sum(v*w for v,w in zip(z,face.normalAt().toTuple())))>1-1e-7:
            caps.append(sum(v*w for v,w in zip(z,face.Center().toTuple())));areas.append(face.Area())
    if len(caps)!=2:raise ValueError('Expected two cap faces')
    area=check['profile']['raw_area']*check['profile_world_unit']**2
    if abs(max(caps)-min(caps)-check['height_normalized'])>1e-7:raise ValueError('Cap height mismatch')
    if any(not math.isclose(v,area,rel_tol=1e-6,abs_tol=1e-10) for v in areas):raise ValueError('Cap area mismatch')
    if not math.isclose(tool.Volume(),area*check['height_normalized'],rel_tol=1e-6,abs_tol=1e-10):raise ValueError('Tool volume mismatch')
    added,removed=b.cut(a,tol=1e-5).Volume(),a.cut(b,tol=1e-5).Volume()
    tolerance=max(a.Volume(),b.Volume(),tool.Volume())*1e-6+1e-9
    changed,forbidden=(added,removed) if check['boolean']=='add' else (removed,added)
    if forbidden>tolerance or changed<=tolerance or changed>tool.Volume()+tolerance:raise ValueError('Material monotonicity/nontriviality/upper-bound failed')
    if abs(added-removed-(b.Volume()-a.Volume()))>tolerance:raise ValueError('Material conservation failed')
    directory=Path(output).parent
    for role,shape in [('source',a),('target',b),('tool',tool)]:cq.exporters.export(shape,str(directory/(role+'.step')))
    write_json(output,{'accepted':True,'source_volume':a.Volume(),'target_volume':b.Volume(),
        'tool_volume':tool.Volume(),'tool_cap_area':areas,'tool_cap_height':max(caps)-min(caps),
        'added_material_volume':added,'removed_material_volume':removed,'tolerance':tolerance})


def run_curation():
    require_container();ROOT.mkdir(exist_ok=True,parents=True)
    if (ROOT/'manifest.json').exists():
        frozen=read_config(ROOT/'manifest.json')
        if any(file_hash(p)!=v for p,v in frozen['hashes'].items()):raise ValueError('Frozen provenance changed; use a new version')
        print(json.dumps(read_config(ROOT/'results.json'),indent=2));return
    write_json(ROOT/'state.json',{'status':'running','phase':'read_only_caption_screen'})
    db=sqlite3.connect('file:data/prepared/cad.sqlite?mode=ro',uri=True)
    prior=set();pilot=read_config('artifacts/lossless-sft-pilot-v1/manifest.json')
    for row in pilot['train']:
        comps=db.execute('SELECT a.component,b.component FROM examples e LEFT JOIN geometry a ON a.id=e.source_id JOIN geometry b ON b.id=e.target_id WHERE e.id=?',(row['example_id'],)).fetchone()
        prior.update(c for c in comps if c)
    raw=read_config('data/extracted/cad_editor/dataset/train.json');screen=[];candidates=[]
    query="""SELECT e.id,e.instruction,a.sequence,b.sequence,a.component,b.component FROM examples e JOIN geometry a ON a.id=e.source_id JOIN geometry b ON b.id=e.target_id WHERE e.split='train' AND e.task='editing' AND e.dataset='cad_editor' AND lower(e.instruction) LIKE 'add %' AND e.instruction GLOB '*[0-9]*' ORDER BY e.id"""
    scanned=0
    for identity,instruction,source,target,sc,tc in db.execute(query):
        scanned+=1;check=verify(instruction,source,target)
        screen.append({'example_id':identity,'instruction':instruction,'verification':check})
        if check['reason']=='outside_supported_grammar':continue
        original=raw[int(identity.rsplit(':',1)[1])]
        norm=lambda s:' '.join(s.replace(',',' ').split())
        if original['instruction'].strip()!=instruction or norm(original['original_sequence'])!=norm(source) or norm(original['edited_sequence'])!=norm(target):raise ValueError('Raw/prepared mismatch: '+identity)
        if check['accepted']:candidates.append({'example_id':identity,'instruction':instruction,'source':source,'target':target,
            'coordinate_context':CONTEXT,'verification':check,'source_component':sc,'target_component':tc,
            'parent_training_component_overlap':bool({sc,tc}&prior),'raw_method':original.get('method')})
    db.close();candidates.sort(key=lambda r:(fraction(r['example_id'],744),r['example_id']))
    write_json(ROOT/'caption_screen.json',screen);write_json(ROOT/'semantic_candidates.json',candidates)
    def measure(row):
        directory=ROOT/'geometry'/row['example_id'].replace(':','_');directory.mkdir(exist_ok=True,parents=True)
        req,out=directory/'request.json',directory/'result.json';write_json(req,row)
        try:
            done=subprocess.run([sys.executable,'-m','cadtokenizer.addition_curation','--request',str(req),'--output',str(out)],capture_output=True,text=True,timeout=90)
            if done.returncode or not out.exists():return {'accepted':False,'error':done.stderr[-1200:]}
            return read_config(out)
        except subprocess.TimeoutExpired:return {'accepted':False,'error':'90-second kernel timeout'}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(row,result) in enumerate(zip(candidates,pool.map(measure,candidates)),1):
            row['geometry_verification']=result
            write_json(ROOT/'state.json',{'status':'running','phase':'geometry_checks','done':i,'total':len(candidates)})
    write_json(ROOT/'checked_candidates.json',candidates)
    verified=[r for r in candidates if r['geometry_verification']['accepted']]
    panel=[];used=set()
    for row in verified:
        components={row['source_component'],row['target_component']}
        if row['parent_training_component_overlap'] or None in components or components&used:continue
        panel.append(row);used.update(components)
    for name,rows in [('verified_all.jsonl',verified),('panel.jsonl',panel)]:
        (ROOT/name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    # Preregistered coverage gate, not a claim that 16 yields a definitive study.
    numeric_configs={json.dumps({k:v for k,v in r['verification']['claims'].items() if k not in ('kind','location')},sort_keys=True) for r in panel}
    adequate=len(panel)>=16 and len(numeric_configs)>=8
    results={'status':'complete','numeric_add_captions_scanned':scanned,
        'grammar_covered':sum(r['verification']['reason']!='outside_supported_grammar' for r in screen),
        'screen_reasons':dict(Counter(r['verification']['reason'] for r in screen)),
        'semantic_candidates':len(candidates),'geometry_verified':len(verified),'independent_panel_count':len(panel),
        'independent_numeric_configurations':len(numeric_configs),'data_gate':{'minimum_independent_add_cases':16,'minimum_numeric_configurations':8,'passed':adequate},
        'coordinate_context':CONTEXT,'unique_target_certification':False,
        'training_launched':False,'inference_launched':False,'original_data_modified':False,'held_out_test_used':False,
        'next':'Freeze predicate scoring and profile a diagnostic baseline' if adequate else 'Existing numeric annotations do not supply enough certified independent additions under this bounded verifier; pursue a different annotation source or independent annotation review before training.'}
    write_json(ROOT/'results.json',results)
    files=['data/prepared/cad.sqlite','data/extracted/cad_editor/dataset/train.json',
        'artifacts/lossless-sft-pilot-v1/manifest.json',__file__,'cadtokenizer/numeric_curation.py',
        'cadtokenizer/geometry_worker.py','tests/test_addition_curation.py']
    files += [str(ROOT/n) for n in ['panel.jsonl','verified_all.jsonl','caption_screen.json','checked_candidates.json','results.json']]
    write_json(ROOT/'manifest.json',{'seed':744,'image_id':os.environ.get('CADTOKENIZER_IMAGE_ID'),
        'coordinate_context':CONTEXT,'hashes':{str(p):file_hash(p) for p in files},
        'selection':'All numeric Add captions from prepared TRAIN, full closed grammar, exact append, all stated claims, kernel/material checks, original pilot component exclusion, one hash-ordered representative per new component. No model outcomes.'})
    table='\n'.join(f"| {r['example_id']} | {r['instruction']} | {r['parent_training_component_overlap']} |" for r in verified)
    (ROOT/'README.md').write_text(f'''# Existing numeric addition curation

Scanned **{scanned}** numeric Add captions in existing prepared training data; **{results['grammar_covered']}** fall within the closed grammar. **{len(candidates)}** match all stated claims under the declared convention. **{len(verified)}** pass independent CAD measurements and nontrivial material-change checks. **{len(panel)}** remain in the component-independent panel. Data gate (at least 16 independent additions and eight numeric configurations): **{'passed' if adequate else 'not met'}**.

## Coordinate contract and scoring scope

{CONTEXT}

Original captions and source/target programs are unchanged. The diagnostic context is supplied separately; it does not prove what units or center the caption author intended. Plain `at` and explicit `centered at` both mean profile center under this declared contract; the verifier never tries minimum corner, translation code or boundary point as alternative interpretations to make an example pass. Extra clauses outside the full grammar are rejected. Only explicitly stated predicates are certified: the target is not necessarily unique when scale, depth or placement are absent. Exact target-program equality would be the wrong sole metric for this panel.

The independent parser requires exactly one appended operation and byte/token-equivalent existing operations. Rectangle side lengths may be exchanged; every asserted height, radius/diameter and center must agree. Holes/cutouts require subtraction, solids require addition. Circular profiles must have consistent diameter endpoints; extra loops, arcs and incompatible topology fail. Nonmatching records are uncertified under this contract, not automatically erroneous corpus labels.

In the cached Unsloth image, CadQuery/OCP validates single solids, independently measures the added tool's cap areas and height, checks analytic tool volume, and verifies nonzero material change, correct inclusion direction, an upper bound from tool volume and volume conservation. This is a separate measurement path in the same CAD backend, not a second CAD implementation or human review. Failed probes are preserved. Unit tests exercise wrong dimensions, mislabeled holes, shifted centers, unrelated source edits and unsupported extra clauses.

## Verified examples (including previously trained families)

| ID | Original instruction | Parent training component overlap |
|---|---|---|
{table}

## Readiness and next action

{results['next']}

This is a bounded expansion, not a license for repeated grammar widening until a desired count appears. Do not train on rejected captions or silently correct them. Keep the existing serving router and previous frozen artifacts unchanged. `verified_all.jsonl` retains all CAD-verified candidates, while `panel.jsonl` excludes every original pilot training component and deduplicates families. Neither is an untouched upstream test set; public pretraining overlap is unknown. The small counts do not estimate a dataset-wide error rate.

Reproduce with `CAD_CPU=1 ./scripts/cad.sh addition-curation`. `manifest.json` freezes inputs, code, panel, outcomes and image ID. `caption_screen.json` retains all screen decisions; `checked_candidates.json` includes rejected CAD probes. STEP files for source/target/tool are under `geometry/`. No training, inference, test-set access, remote rental, applied exclusions or relabeling occurred.
''')
    write_json(ROOT/'state.json',{'status':'complete','independent_panel_count':len(panel),'data_gate_passed':adequate})
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();geometry_check(args.request,args.output)
