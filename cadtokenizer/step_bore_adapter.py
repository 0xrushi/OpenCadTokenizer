"""Narrow STEP-native vertical bore editor using source surfaces and public task.

No target files, reference specifications or dataset identities enter this API.
Supporting-surface continuation is conditional on the stated independent-cut
feature semantics. This is not a general STEP feature-history reconstruction.
"""
import argparse
import json
import math
from pathlib import Path
import re


def public_request(prompt, value):
    xy=re.search(r'\(x,y\)=\(([\d.]+),\s*([\d.]+)\)',prompt)
    diameter=re.search(r'(?:currently diameter|from diameter) ([\d.]+) mm',prompt)
    if not xy or not diameter or not re.search(r'vertical through[- ](?:port|bore)',prompt):
        raise ValueError('Only an explicit vertical through-bore request is supported')
    value=float(value);old=float(diameter[1])
    if not math.isfinite(value) or value<=0 or old<=0:raise ValueError('Diameters must be finite and positive')
    if 'independent' not in prompt.lower():raise ValueError('Independent-cut continuation must be stated')
    return {'center_xy_mm':list(map(float,xy.groups())),'old_diameter_mm':old,'diameter_mm':value,
            'full_toroidal_passage':bool(re.search(r'full circular ring',prompt,re.I))}


def dot(a,b):return sum(x*y for x,y in zip(a,b))
def minus(a,b):return [x-y for x,y in zip(a,b)]
def plus(a,b):return [x+y for x,y in zip(a,b)]
def times(a,s):return [x*s for x in a]
def xyz(p):return [p.X(),p.Y(),p.Z()]


def inventory(shape):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    result=[]
    for index,face in enumerate(shape.Faces()):
        kind=face.geomType();adaptor=BRepAdaptor_Surface(face.wrapped,True);u0,u1,v0,v1=face.uvBounds()
        normal,point=face.normalAt((u0+u1)/2,(v0+v1)/2);point=list(point.toTuple());normal=list(normal.toTuple())
        bounds=face.BoundingBox()
        item={'index':index,'type':kind,'area_mm2':face.Area(),
            'bounds':[bounds.xmin,bounds.ymin,bounds.zmin,bounds.xmax,bounds.ymax,bounds.zmax]}
        radial=None
        if kind in ('CYLINDER','CONE','TORUS'):
            surface={'CYLINDER':adaptor.Cylinder,'CONE':adaptor.Cone,'TORUS':adaptor.Torus}[kind]()
            origin=xyz(surface.Location());axis=xyz(surface.Axis().Direction())
            delta=minus(point,origin);radial=minus(delta,times(axis,dot(delta,axis)))
            item.update(origin_mm=origin,axis=axis,v_bounds=[v0,v1])
            if kind=='CYLINDER':item['radius_mm']=surface.Radius()
            if kind=='CONE':item.update(reference_radius_mm=surface.RefRadius(),semi_angle_rad=surface.SemiAngle())
            if kind=='TORUS':
                item.update(major_radius_mm=surface.MajorRadius(),minor_radius_mm=surface.MinorRadius())
                length=math.sqrt(dot(radial,radial))
                if length<1e-9:raise ValueError('Degenerate torus frame')
                radial=minus(delta,times(radial,surface.MajorRadius()/length))
            item['concave']=dot(normal,radial)<0
        elif kind=='PLANE':
            plane=adaptor.Plane();item.update(origin_mm=xyz(plane.Location()),normal=xyz(plane.Axis().Direction()))
            item['edge_types']=[edge.geomType() for edge in face.Edges()]
        result.append(item)
    return result


def same_bore(item,request):
    if item['type']!='CYLINDER' or not item['concave']:return False
    return (abs(abs(item['axis'][2])-1)<1e-7
        and math.dist(item['origin_mm'][:2],request['center_xy_mm'])<1e-6
        and abs(item['radius_mm']-request['old_diameter_mm']/2)<1e-6)


def cavity_tool(item):
    import cadquery as cq
    kind=item['type'];origin=item['origin_mm'];axis=item['axis'];v0,v1=item['v_bounds']
    if kind=='CYLINDER':
        return cq.Solid.makeCylinder(item['radius_mm'],v1-v0,cq.Vector(*plus(origin,times(axis,v0))),cq.Vector(*axis))
    if kind=='TORUS':
        return cq.Solid.makeTorus(item['major_radius_mm'],item['minor_radius_mm'],cq.Vector(*origin),cq.Vector(*axis))
    if kind=='CONE':
        angle=item['semi_angle_rad'];r0=item['reference_radius_mm']+v0*math.sin(angle);r1=item['reference_radius_mm']+v1*math.sin(angle)
        if min(r0,r1)<-1e-8:raise ValueError('Cone crosses its apex')
        return cq.Solid.makeCone(max(0,r0),max(0,r1),(v1-v0)*math.cos(angle),
            cq.Vector(*plus(origin,times(axis,v0*math.cos(angle)))),cq.Vector(*axis))
    raise ValueError('Unsupported cavity continuation')


def continued_cylinders(faces):
    """Recover a declared independent straight passage across split wall patches.

    Canonical line coordinates keep grouping invariant to axis direction and the
    arbitrary axial origin used by a STEP supporting surface.
    """
    groups={}
    for face in faces:
        if face['type']!='CYLINDER' or not face.get('concave'):continue
        axis=face['axis']
        if next(v for v in axis if abs(v)>1e-10)<0:axis=times(axis,-1)
        origin=minus(face['origin_mm'],times(axis,dot(face['origin_mm'],axis)))
        endpoints=[plus(face['origin_mm'],times(face['axis'],v)) for v in face['v_bounds']]
        positions=[dot(p,axis) for p in endpoints]
        key=tuple(round(v,7) for v in axis+origin+[face['radius_mm']])
        if key not in groups:
            groups[key]={**face,'axis':axis,'origin_mm':origin,'v_bounds':[min(positions),max(positions)],'source_face_indices':[face['index']]}
        else:
            item=groups[key];item['v_bounds']=[min(item['v_bounds'][0],*positions),max(item['v_bounds'][1],*positions)]
            item['source_face_indices'].append(face['index'])
    return list(groups.values())


def edit(shape,value,*,prompt,preserve_passages=True):
    import cadquery as cq
    request=public_request(prompt,value)
    if not shape.isValid() or len(shape.Solids())!=1:raise ValueError('Require one valid source solid')
    faces=inventory(shape);target=[x for x in faces if same_bore(x,request)]
    if not target:raise ValueError('Requested bore absent from source STEP')
    bounds=shape.BoundingBox();x,y=request['center_xy_mm'];radius=request['old_diameter_mm']/2
    padding=max(bounds.xlen,bounds.ylen,bounds.zlen)*.01+1
    through_origin=cq.Vector(x,y,bounds.zmin-padding)
    if request['diameter_mm']/2>=min(x-bounds.xmin,bounds.xmax-x,y-bounds.ymin,bounds.ymax-y):
        raise ValueError('Requested bore reaches the external side walls')
    new_tool=cq.Solid.makeCylinder(request['diameter_mm']/2,bounds.zlen+2*padding,through_origin)
    # Enlarging a void needs no missing-material reconstruction.
    if request['diameter_mm']>=request['old_diameter_mm']:
        result=shape.cut(new_tool).clean()
        if not result.isValid() or len(result.Solids())!=1:raise ValueError('Edit produced invalid/multiple solids')
        return result,{'request':request,'source_surfaces':faces,'restoration':'none_enlargement_or_identity','continued_cavities':[]}
    # Restoration is confined to an interior vertical column, never an exterior
    # bounding-box replacement. Refuse columns near external rounds/side walls.
    exterior_radii=[f.get('radius_mm',0) for f in faces if f['type']=='CYLINDER' and not f['concave']]
    margin=max(exterior_radii+[0])
    clearance=min(x-bounds.xmin,bounds.xmax-x,y-bounds.ymin,bounds.ymax-y)-radius
    if clearance<=margin+1e-6:raise ValueError('Restoration column too close to unsupported exterior geometry')
    restoration=cq.Solid.makeCylinder(radius,bounds.zlen,cq.Vector(x,y,bounds.zmin))
    continued=[]
    for face in faces:
        if same_bore(face,request):continue
        if face['type']=='PLANE':
            # An internal planar wall meeting the restoration column needs a
            # richer feature model; do not guess an unobserved box cut.
            b=face['bounds'];nearest=[max(b[0],min(x,b[3])),max(b[1],min(y,b[4]))]
            normal=face['normal'];outer=any(abs(abs(normal[i])-1)<1e-7 and any(abs(face['origin_mm'][i]-edge)<1e-6 for edge in [getattr(bounds,('xmin','ymin','zmin')[i]),getattr(bounds,('xmax','ymax','zmax')[i])]) for i in range(3))
            # Circular/annular caps belong to the finite cylinder/cone tools
            # reconstructed above; a nearby rectangular cap/wall does not.
            analytic_cap=bool(face['edge_types']) and all(t=='CIRCLE' for t in face['edge_types'])
            if not outer and not analytic_cap and math.dist(nearest,[x,y])<radius-1e-6:
                raise ValueError('Unmodeled internal planar feature near restoration column')
        if not face.get('concave'):continue
        if face['type'] not in ('CYLINDER','CONE','TORUS'):raise ValueError('Unsupported concave surface')
        if face['type']=='CYLINDER':continue
        if face['type']=='TORUS' and not request['full_toroidal_passage']:
            raise ValueError('Full-ring continuation was not specified')
        if preserve_passages:
            restoration=restoration.cut(cavity_tool(face))
            continued.append(face['index'])
    cylinders=continued_cylinders([f for f in faces if not same_bore(f,request)])
    if preserve_passages:
        for cylinder in cylinders:
            restoration=restoration.cut(cavity_tool(cylinder))
            continued.extend(cylinder['source_face_indices'])
    result=shape.fuse(restoration).cut(new_tool).clean()
    if not result.isValid() or len(result.Solids())!=1:raise ValueError('Edit produced invalid/multiple solids')
    return result,{'request':request,'source_surfaces':faces,'restoration':'interior_old_bore_column_minus_independent_cavities',
                   'continued_cavities':continued,'merged_cylinders':cylinders,'outside_clearance_mm':clearance}


def make_editor(prompt):
    """Bind public task text to the dataset's edit(shape, value) callable contract."""
    def bound_edit(shape,value):
        result,_=edit(shape,value,prompt=prompt)
        return result
    return bound_edit


def worker(request_path,output_path):
    from .common import require_container,read_config,write_json
    require_container()
    import cadquery as cq
    request=read_config(request_path)
    if set(request)!={'source_step','instruction','value_mm'}:raise ValueError('Input-only request schema violation')
    shape=cq.importers.importStep(request['source_step']).val()
    result,evidence=edit(shape,request['value_mm'],prompt=request['instruction'])
    output=Path(output_path);output.parent.mkdir(exist_ok=True,parents=True)
    cq.exporters.export(result,str(output))
    write_json(output.with_suffix('.evidence.json'),evidence)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();worker(args.request,args.output)
