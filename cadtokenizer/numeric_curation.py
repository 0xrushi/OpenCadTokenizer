"""Conservative, versioned curation of existing numeric height captions.

The caption verifier uses a separate text/SEM parser, not instruction_audit.describe.
Acceptance is conditional on an explicit diagnostic coordinate contract.
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

ROOT = Path('artifacts/numeric-curation-v1')
CONTEXT = (
    "For this diagnostic, first/second refers to the ordered sketch-extrusion "
    "operations in the source SEM. Extrusion height is upper minus lower axial "
    "plane. One height unit is one axial grid interval, equal to 2/63 normalized "
    "geometry units. Change only the specified operation's upper axial plane; "
    "keep its lower plane and every other SEM field unchanged."
)
ENTITY = r'(?:extrusion|extrude|extruded sketch|sketch extrusion|cylinder|prism)'
PREFIX = r'(?P<verb>Increase|Decrease|Reduce|Change|Set) '
SUFFIX = r' (?P<mode>by|to) (?P<amount>\d+) units?\.'
PATTERNS = [re.compile(PREFIX + form + SUFFIX, re.I) for form in (
    rf'(?:the )?(?P<ordinal>first|second) (?P<entity>{ENTITY})(?:\'s)? height',
    rf'(?:the )?height of (?:the )?(?P<ordinal>first|second) (?P<entity>{ENTITY})',
)]


def parse_program(sequence):
    """Independent minimal parser retaining every untouched sketch byte/token."""
    result = []
    chunks = sequence.strip().split('<extrude_end>')
    if chunks[-1].strip():
        raise ValueError('Incomplete extrusion')
    for chunk in chunks[:-1]:
        match = re.fullmatch(r'\s*(.+<sketch_end>)\s+(add|cut|intersect),([\d,\s-]+)\s*', chunk)
        if not match:
            raise ValueError('Unsupported SEM syntax')
        fields = [int(v) for v in match[3].replace(',', ' ').split()]
        if len(fields) != 17:
            raise ValueError('Expected 17 numeric extrusion fields')
        result.append({'sketch': ' '.join(match[1].split()), 'operation': match[2], 'fields': fields})
    if not result:
        raise ValueError('Empty SEM')
    return result


def verify(instruction, source, target):
    match = next((m for p in PATTERNS if (m := p.fullmatch(instruction))), None)
    if match is None:
        return {'accepted': False, 'reason': 'outside_supported_caption_grammar'}
    verb, mode, amount = match['verb'].lower(), match['mode'].lower(), int(match['amount'])
    if amount <= 0 or (mode == 'by' and verb in ('change', 'set')):
        return {'accepted': False, 'reason': 'ambiguous_or_zero_numeric_action'}
    index = 0 if match['ordinal'].lower() == 'first' else 1
    try:
        before, after = parse_program(source), parse_program(target)
    except ValueError as exc:
        return {'accepted': False, 'reason': 'parse_error', 'detail': str(exc)}
    if len(before) != len(after) or index >= len(before):
        return {'accepted': False, 'reason': 'operation_count_or_reference'}
    a, b = before[index], after[index]
    entity = match['entity'].lower()
    # Physical feature nouns are accepted only for filled additive simple profiles.
    if entity in ('cylinder', 'prism'):
        sk = a['sketch']
        circular = re.fullmatch(r'circle,[\d,]+ <curve_end> <loop_end> <face_end> <sketch_end>', sk)
        rectangular = (len(re.findall(r'\bline,', sk)) == 4 and sk.count('<loop_end>') == 1
                       and sk.count('<face_end>') == 1 and 'circle,' not in sk and 'arc,' not in sk)
        if a['operation'] != 'add' or not (circular if entity == 'cylinder' else rectangular):
            return {'accepted': False, 'reason': 'unverified_feature_noun'}
    if any(x != y for i, (x, y) in enumerate(zip(before, after)) if i != index):
        return {'accepted': False, 'reason': 'unrequested_other_operation_changes'}
    p, q = a['fields'], b['fields']
    if a['sketch'] != b['sketch'] or a['operation'] != b['operation'] or p[2:] != q[2:]:
        return {'accepted': False, 'reason': 'unrequested_profile_transform_or_boolean_changes'}
    if p[0] != q[0]:
        return {'accepted': False, 'reason': 'lower_plane_changes_outside_contract'}
    if not (0 <= p[0] < p[1] <= 63 and 0 <= q[0] < q[1] <= 63):
        return {'accepted': False, 'reason': 'invalid_axial_bounds'}
    old, new = p[1] - p[0], q[1] - q[0]
    signed = amount if verb == 'increase' else -amount
    expected = amount if mode == 'to' else old + signed
    if new != expected or new == old or (verb == 'increase' and new <= old) or (verb in ('reduce', 'decrease') and new >= old):
        return {'accepted': False, 'reason': 'numeric_height_contradiction',
                'observed_before': old, 'observed_after': new, 'expected_after': expected}
    if a['operation'] == 'intersect':
        return {'accepted': False, 'reason': 'intersection_not_supported'}
    return {'accepted': True, 'reason': 'matches_explicit_diagnostic_contract', 'operation_index': index,
            'before_height_grid': old, 'after_height_grid': new,
            'before_height_normalized': old * 2 / 63, 'after_height_normalized': new * 2 / 63,
            'boolean': a['operation']}


def isolated_sem(sequence, index):
    chunk = sequence.strip().split('<extrude_end>')[index].strip()
    return re.sub(r'(<sketch_end>\s+)(?:add|cut|intersect),', r'\g<1>add,', chunk) + ' <extrude_end>'


def check_geometry(request, output):
    require_container()
    import cadquery as cq
    from .geometry_worker import build_solid
    row = read_config(request)
    check = verify(row['instruction'], row['source'], row['target'])
    if not check['accepted']:
        raise ValueError('Unverified input to geometry check')
    settings = {'boolean_tolerance': 1e-5, 'minimum_volume': 1e-10}
    shapes = [build_solid(row[k], settings) for k in ('source', 'target')]
    if any(not s.isValid() or len(s.Solids()) != 1 for s in shapes):
        raise ValueError('Require valid, nonempty, single-solid source and target')
    index = check['operation_index']
    isolated = []
    for role in ('source', 'target'):
        shape = build_solid(isolated_sem(row[role], index), settings)
        # Measure separation of planar cap centers from BRep, not from token span.
        fields = parse_program(row[role])[index]['fields']
        z = fields[11:14]; norm = math.sqrt(sum(v*v for v in z)); z = [v/norm for v in z]
        caps = []
        for face in shape.Faces():
            if face.geomType() != 'PLANE':
                continue
            normal = face.normalAt().toTuple()
            if abs(sum(a*b for a,b in zip(z, normal))) > 1 - 1e-7:
                caps.append(sum(a*b for a,b in zip(z, face.Center().toTuple())))
        if len(caps) < 2:
            raise ValueError('Unable to measure two planar caps')
        height = max(caps)-min(caps)
        if abs(height-check[('before' if role == 'source' else 'after')+'_height_normalized']) > 1e-7:
            raise ValueError('Independent BRep cap-height mismatch')
        isolated.append({'cap_height': height, 'volume': shape.Volume()})
    ratio = check['after_height_grid']/check['before_height_grid']
    if not math.isclose(isolated[1]['volume']/isolated[0]['volume'], ratio, rel_tol=1e-6):
        raise ValueError('Isolated tool volume does not follow height ratio')
    a,b = shapes
    added, removed = b.cut(a, tol=1e-5).Volume(), a.cut(b, tol=1e-5).Volume()
    delta = b.Volume()-a.Volume()
    tolerance = max(a.Volume(),b.Volume())*1e-6 + 1e-9
    increasing = check['after_height_grid'] > check['before_height_grid']
    material_increases = increasing == (check['boolean'] == 'add')
    forbidden, expected = (removed,added) if material_increases else (added,removed)
    if forbidden > tolerance or expected <= tolerance or abs((added-removed)-delta) > tolerance:
        raise ValueError('Material difference failed monotonicity/nontriviality/conservation')
    directory = Path(output).parent
    for role,shape in zip(('source','target'),shapes):
        cq.exporters.export(shape, str(directory/(role+'.step')))
    write_json(output, {'accepted': True, 'isolated_tools': isolated, 'added_material_volume': added,
        'removed_material_volume': removed, 'source_volume': a.Volume(), 'target_volume': b.Volume(),
        'single_solid': True, 'tolerance': tolerance})


def run_curation():
    require_container()
    ROOT.mkdir(exist_ok=True, parents=True)
    if (ROOT/'manifest.json').exists():
        frozen = read_config(ROOT/'manifest.json')
        if any(file_hash(path) != digest for path,digest in frozen['hashes'].items()):
            raise ValueError('Frozen curation provenance changed; use a new version/root')
        print(json.dumps(read_config(ROOT/'results.json'),indent=2),flush=True)
        return
    audit = Path('artifacts/numeric-instruction-audit-v1')
    write_json(ROOT/'state.json', {'status': 'running', 'phase': 'read_only_scan'})
    raw = read_config('data/extracted/cad_editor/dataset/train.json')
    pilot = read_config('artifacts/lossless-sft-pilot-v1/manifest.json')
    connection = sqlite3.connect('file:data/prepared/cad.sqlite?mode=ro', uri=True)
    prior_components = set()
    for item in pilot['train']:
        comps = connection.execute('SELECT a.component,b.component FROM examples e LEFT JOIN geometry a ON a.id=e.source_id JOIN geometry b ON b.id=e.target_id WHERE e.id=?', (item['example_id'],)).fetchone()
        prior_components.update(c for c in comps if c)
    reviews, candidates = [], []
    query = '''SELECT e.id,e.instruction,a.sequence,b.sequence,a.component,b.component
        FROM examples e JOIN geometry a ON a.id=e.source_id JOIN geometry b ON b.id=e.target_id
        WHERE e.split='train' AND e.task='editing' AND e.dataset='cad_editor'
        AND lower(e.instruction) LIKE '%height%' AND e.instruction GLOB '*[0-9]*' ORDER BY e.id'''
    for identity,instruction,source,target,sc,tc in connection.execute(query):
        result = verify(instruction,source,target)
        if result['reason'] == 'outside_supported_caption_grammar':
            continue
        original = raw[int(identity.rsplit(':',1)[1])]
        if original['instruction'].strip() != instruction or ' '.join(original['original_sequence'].replace(',', ' ').split()) != ' '.join(source.replace(',', ' ').split()) or ' '.join(original['edited_sequence'].replace(',', ' ').split()) != ' '.join(target.replace(',', ' ').split()):
            raise ValueError('Prepared/raw discrepancy: '+identity)
        record = {'example_id': identity, 'instruction': instruction, 'verification': result}
        reviews.append(record)
        if result['accepted']:
            candidates.append({**record, 'source': source, 'target': target, 'source_component': sc,
                'target_component': tc, 'parent_training_component_overlap': bool({sc,tc}&prior_components),
                'coordinate_context': CONTEXT, 'raw_method': original.get('method')})
    connection.close()
    candidates.sort(key=lambda r:(fraction(r['example_id'],643),r['example_id']))
    write_json(ROOT/'caption_screen.json', reviews)
    write_json(ROOT/'semantic_candidates.json', candidates)
    def measure(row):
        directory = ROOT/'geometry'/row['example_id'].replace(':','_'); directory.mkdir(parents=True,exist_ok=True)
        request,output=directory/'request.json',directory/'result.json'
        write_json(request,row)
        try:
            done=subprocess.run([sys.executable,'-m','cadtokenizer.numeric_curation','--request',str(request),'--output',str(output)],capture_output=True,text=True,timeout=90)
            if done.returncode or not output.exists():
                return {'accepted':False,'error':done.stderr[-1200:]}
            return read_config(output)
        except subprocess.TimeoutExpired:
            return {'accepted':False,'error':'geometry check timed out at 90 seconds'}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index,(row,result) in enumerate(zip(candidates,pool.map(measure,candidates)),1):
            row['geometry_verification']=result
            write_json(ROOT/'state.json',{'status':'running','phase':'independent_geometry_checks','done':index,'total':len(candidates)})
    write_json(ROOT/'checked_candidates.json',candidates)
    # One representative per connected component; no training/evaluation split is
    # manufactured from inverse edits of the same underlying geometry family.
    panel=[]; used=set()
    for row in candidates:
        components={row['source_component'],row['target_component']}
        if (not row['geometry_verification']['accepted'] or row['parent_training_component_overlap']
                or None in components or components&used):
            continue
        used.update(components); panel.append(row)
    with (ROOT/'panel.jsonl').open('w') as handle:
        for row in panel:
            handle.write(json.dumps(row)+'\n')
    write_json(ROOT/'review_set.json', {'source_audit_sha256':file_hash(audit/'reviewed_cases.json'),
        'source_candidates':read_config(audit/'quarantine_candidates.json'),
        'policy':'No original rows excluded or rewritten. Original audit flags remain review candidates; narrow verifier cannot adjudicate all Add/Move captions.',
        'original_cases':read_config(audit/'reviewed_cases.json')})
    stats={'status':'complete','supported_caption_count':len(reviews),
        'screen_reasons':dict(Counter(r['verification']['reason'] for r in reviews)),
        'semantically_matching':len(candidates),
        'geometry_verified':sum(r['geometry_verification']['accepted'] for r in candidates),
        'frozen_panel_count':len(panel),'unique_panel_components':len(used),
        'parent_training_overlap_in_panel':0,'coordinate_context':CONTEXT,
        'limits':'Conditional height-only diagnostic from existing TRAIN data; not an untouched upstream test set or verification of Add/Move instruction following. Captions unchanged; explicit coordinate context changes task framing.',
        'training_started':False,'models_modified':False,'database_modified':False,
        'next':'Run a small frozen diagnostic baseline on the verified panel; only then decide whether a clean-data learnability run is warranted.'}
    write_json(ROOT/'results.json',stats)
    files=['data/prepared/cad.sqlite','data/extracted/cad_editor/dataset/train.json',
           'artifacts/lossless-sft-pilot-v1/manifest.json',str(audit/'reviewed_cases.json'),__file__,
           'cadtokenizer/geometry_worker.py','tests/test_numeric_curation.py',str(ROOT/'panel.jsonl'),str(ROOT/'caption_screen.json'),str(ROOT/'checked_candidates.json')]
    write_json(ROOT/'manifest.json',{'seed':643,'coordinate_context':CONTEXT,'hashes':{str(p):file_hash(p) for p in files},
        'image_id':os.environ.get('CADTOKENIZER_IMAGE_ID'),'selection':'All grammar matches in prepared CAD-Editor train; arithmetic/unchanged-field, CAD geometry and parent-component gates; first hash-ordered verified example per new component. No model outcomes used.'})
    table='\n'.join(f"| {r['example_id']} | {r['instruction']} | {r['verification']['before_height_grid']} → {r['verification']['after_height_grid']} |" for r in panel)
    (ROOT/'README.md').write_text(f'''# Numeric curation v1

Completed a read-only screen of existing training annotations. Of {len(reviews)} captions in the supported height grammar, {len(candidates)} match the explicit coordinate contract and {stats['geometry_verified']} pass independent CAD checks. The frozen panel contains **{len(panel)}** examples, one per source/target component, with no component used in original pilot training. Original captions are preserved. This is a small height-edit diagnostic; it cannot establish Add/Move performance or corpus-wide label accuracy.

## Explicit task context

{CONTEXT}

This context is supplied separately from the original instruction. Acceptance is conditional on this declared convention; it does not recover the caption author's intended physical units. Only an upper-plane edit is permitted. No profile, placement, orientation, Boolean, other-operation or lower-plane changes are accepted. Do not mix these context-augmented results with earlier benchmarks.

## Verification

The verifier uses its own SEM field parser and a closed caption grammar. Adversarial tests check wrong signs/numbers, unrelated changes, unsupported instructions, inverse edits and coordinate preservation. Inside the cached Unsloth image, CadQuery/OCP independently measures isolated extrusion cap separation and volume ratios, then checks full source/target single-solid validity, nonzero material change, monotonicity and volume conservation. Rejected geometry probes are retained. This is an independent measurement path within the same kernel backend, not an independent CAD implementation or human annotation review.

The panel is selected without model scores, from existing train records, excluding all components in the original pilot training manifest. Each component appears once, preventing reverse edits from becoming separate independent cases. Public base-model pretraining overlap is unknown. No upstream test data is accessed. No training examples are created, existing captions rewritten, dataset rows deleted or models trained.

## Frozen panel

| ID | Unchanged original instruction | Height in declared grid units |
|---|---|---|
{table}

## Files and next action

`panel.jsonl` contains original instructions, source/target SEMs, explicit context and geometry evidence. `caption_screen.json` records every supported-caption decision; `checked_candidates.json` includes failures and component overlap. STEP files and isolated measurement results live under `geometry/`. `review_set.json` preserves the earlier 32-case review flags; they remain provisional and have not been applied as an automatic filter. `manifest.json` freezes data, verifier, context, panel and image hashes.

Next: score the unchanged serving editing adapter on this frozen diagnostic, with the same context for every request, before deciding on clean-data training. No baseline inference or further training has been launched in this stage. A height-only panel does not resolve numeric additions; those need separately checked annotations rather than a claim that height curation repairs them.

Reproduce: `CAD_CPU=1 ./scripts/cad.sh numeric-curation`. Preserve this version's evidence; use a new root/version if data, verifier or context changes.
''')
    write_json(ROOT/'state.json',{'status':'complete','phase':'panel_frozen','panel_count':len(panel)})
    print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();check_geometry(args.request,args.output)
