"""Compare the pinned author's parser with the independent local numeric trace."""
import ast
from collections import OrderedDict
import math
from pathlib import Path
import re

from .common import require_container,read_config,write_json,file_hash


def run_parity():
    require_container()
    import numpy as np
    root=Path('artifacts/numeric-instruction-audit-v1')
    sources=read_config(root/'upstream/sources.json')
    source=Path(sources['files']['utils/parse_seq2obj.py']['local_path'])
    if file_hash(source)!=sources['files']['utils/parse_seq2obj.py']['sha256']:raise ValueError('Pinned parser changed')
    tree=ast.parse(source.read_text())
    names={'CADparser','dequantize_verts','find_arc_geometry','angle_from_vector_to_x'}
    definitions=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in definitions}!=names:raise ValueError('Unexpected author parser definitions')
    # Only reviewed pure parser definitions are loaded, not the upstream CLI,
    # file-writing routines, or the separate 70B caption-generation program.
    scope={'np':np,'OrderedDict':OrderedDict,'math':math,'re':re,'SKETCH_R':1.,'EXTRUDE_R':1.,'SCALE_R':1.4,'OFFSET_R':.9}
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(source),'exec'),scope)
    arc_errors=[];original_arc=scope['find_arc_geometry']
    def traced_arc(*args):
        try:return original_arc(*args)
        except Exception as exc:
            arc_errors.append(type(exc).__name__+': '+str(exc));raise
    scope['find_arc_geometry']=traced_arc
    records=[];rejected=[]
    for row in read_config(root/'per_case.json'):
        for role,ops in [('source',row['source_operations']),('target',row['target_operations'])]:
            seq=row[role];parser=scope['CADparser'](6);sketches,extrudes=parser.get_SE(seq)
            errors=[];circle_errors=[]
            for local,ext in zip(ops,extrudes):
                params,scale,offset=parser.parse_extrude(ext)
                errors += [abs(float(x)-y) for x,y in zip(params['value'],local['world_plane_bounds'])]
                errors += [abs(float(x)-y) for x,y in zip(params['T'],local['world_translation'])]
                errors += [abs(float(scale)-local['world_scale'])]
                errors += [abs(float(x)-y) for x,y in zip(offset,local['world_offset'])]
            error_offset=len(arc_errors);parsed=parser.perform(seq)
            if parsed is None:rejected.append({'id':row['example_id'],'role':role,'caught_arc_errors':arc_errors[error_offset:]})
            for local in ops:
                for primitive in local['primitives']:
                    if primitive['type']!='circle':continue
                    cur=['circle']+list(map(str,primitive['raw_coordinates']))
                    p1,p2,p3,p4=parser.dequantize_circle_points(cur,cur,None,None)
                    radius=(np.linalg.norm(p1-p2)+np.linalg.norm(p3-p4))/4.*np.float32(local['world_scale'])
                    circle_errors.append(abs(float(radius)-primitive['world_radius']))
            maximum=max(errors+circle_errors+[0.])
            if maximum>1e-6:raise ValueError('Author/local numeric conversion disagreement')
            records.append({'id':row['example_id'],'role':role,'operation_count':len(ops),
                'circles_checked':len(circle_errors),'max_absolute_difference':maximum})
    result={'checked_sequences':len(records),'circles_checked':sum(r['circles_checked'] for r in records),
        'max_absolute_difference':max(r['max_absolute_difference'] for r in records),'tolerance':1e-6,
        'upstream_commit':sources['commit'],'source_sha256':file_hash(source),'records':records,'full_author_parser_rejections':rejected,
        'scope':'Extrusion dequantization, translation, sketch scale/offset, and effective circle radii. Float32 author versus Float64 local trace. This does not prove identical Boolean/meshing backends or certify natural-language labels.'}
    write_json(root/'parser_parity.json',result)
    print({k:v for k,v in result.items() if k!='records'},flush=True)
