"""Inspect downloaded train-only CAD-Preserve assets; never execute references."""
import ast
import importlib.metadata
import json
import math
import os
from pathlib import Path

from .common import file_hash, read_config, require_container, write_json

ROOT=Path('artifacts/cad-preserve-train-inspection-v1')
DATA=Path('data/external/cad-preserve-train-v0.1.0')


def run_inspection():
    require_container()
    import cadquery as cq
    ROOT.mkdir(exist_ok=True,parents=True)
    if (ROOT/'manifest.json').exists():
        frozen=read_config(ROOT/'manifest.json')
        if any(file_hash(p)!=v for p,v in frozen['hashes'].items()):raise ValueError('Frozen inspection changed')
        print(json.dumps(read_config(ROOT/'results.json'),indent=2));return
    download=read_config(DATA/'download_manifest.json')
    for item in download['files']:
        if file_hash(DATA/item['path'])!=item['sha256']:raise ValueError('Downloaded file hash mismatch')
    records=read_config(DATA/'pilot_records.json')
    assert len(records)==2 and all(r['split']=='train' and r['units']=='mm' for r in records)
    output=[]
    for row in records:
        # Read the static published design specification only. No exec/import of
        # the downloaded reference, generator, graders or paid-model harness.
        tree=ast.parse((DATA/row['reference_file']).read_text())
        assignments=[n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SPEC' for t in n.targets)]
        if len(assignments)!=1:raise ValueError('Expected one literal source specification')
        spec=ast.literal_eval(assignments[0]);target=spec['target']
        if target['kind']!='cylinder':raise ValueError('Pilot only supports cylindrical bore inspection')
        origin=target['origin'];axis=target['axis'];norm=math.sqrt(sum(v*v for v in axis));axis=[v/norm for v in axis]
        def cylinder_radius(shape,expected):
            found=[]
            for face in shape.Faces():
                if face.geomType()!='CYLINDER':continue
                cylinder=face._geomAdaptor().Cylinder();a=cylinder.Axis();d=a.Direction();p=a.Location()
                direction=[d.X(),d.Y(),d.Z()];location=[p.X(),p.Y(),p.Z()]
                delta=[x-y for x,y in zip(location,origin)];projection=sum(x*y for x,y in zip(delta,axis))
                perpendicular=math.sqrt(sum((v-projection*w)**2 for v,w in zip(delta,axis)))
                if abs(sum(v*w for v,w in zip(direction,axis)))>1-1e-7 and perpendicular<1e-6:
                    found.append(cylinder.Radius())
            if not any(abs(r-expected)<1e-6 for r in found):raise ValueError('Published bore diameter/axis absent from BRep')
            return found
        source=cq.importers.importStep(str(DATA/row['input_file'])).val()
        if not source.isValid() or len(source.Solids())!=1:raise ValueError('Invalid source')
        source_radii=cylinder_radius(source,spec['old_value']/2)
        parameters=[]
        for gold,check_path in zip(row['gold_files'],row['check_files']):
            check=read_config(DATA/check_path);shape=cq.importers.importStep(str(DATA/gold)).val()
            if not shape.isValid() or len(shape.Solids())!=1:raise ValueError('Invalid target')
            radii=cylinder_radius(shape,check['value']/2)
            probes=[]
            for probe in check['probes']:
                observed=bool(shape.isInside(cq.Vector(*probe['point']),1e-7))
                if observed!=probe['material']:raise ValueError('Published functional witness failed')
                probes.append({'name':probe['name'],'expected_material':probe['material'],'observed_material':observed})
            a,b=source.BoundingBox(),shape.BoundingBox()
            if max(abs(getattr(a,key)-getattr(b,key)) for key in ['xmin','xmax','ymin','ymax','zmin','zmax'])>1e-6:raise ValueError('Bore resize changed external bounds')
            parameters.append({'value_mm':check['value'],'observed_coaxial_radii_mm':radii,'valid_single_solid':True,
                'volume_mm3':shape.Volume(),'functional_probes':probes,'external_bounds_preserved':True})
        if len(parameters)!=4:raise ValueError('Expected four published parameter cases')
        output.append({'id':row['id'],'family':row['family'],'split':row['split'],'source_volume_mm3':source.Volume(),
            'source_coaxial_radii_mm':source_radii,'parameters':parameters,
            'source_representation_flags':{'fillet':bool(spec.get('edge_radius')),
                'fixed_feature_kinds':sorted({f['kind'] for f in spec['fixed']})}})
    results={'status':'complete','dataset_revision':download['revision'],'downloaded_train_metadata_records':40,
        'inspected_train_sources':len(output),'inspected_targets':sum(len(r['parameters']) for r in output),
        'functional_probes_passed':sum(len(p['functional_probes']) for r in output for p in r['parameters']),
        'units':'mm','cases':output,'reference_programs_executed':False,'model_execution':False,
        'test_or_validation_records_downloaded':False,'cadquery_version':importlib.metadata.version('cadquery'),
        'readiness':'CAD asset compatibility smoke passed; not directly compatible with the current SEM editing model. No training benefit established.',
        'next':'Assess a STEP/CadQuery editing adapter on train-only cases, preserving unit semantics; do not claim lossless conversion of fillets, cones or tori into the current planar-extrusion SEM.'}
    write_json(ROOT/'results.json',results)
    paths=[str(DATA/'download_manifest.json'),str(DATA/'pilot_records.json'),__file__,str(ROOT/'results.json')]
    write_json(ROOT/'manifest.json',{'image_id':os.environ.get('CADTOKENIZER_IMAGE_ID'),
        'hashes':{p:file_hash(p) for p in paths},'downloaded_asset_hashes':download['files']})
    write_json(ROOT/'state.json',{'status':'complete','sources':2,'targets':8})
    print(json.dumps({k:v for k,v in results.items() if k!='cases'},indent=2),flush=True)
