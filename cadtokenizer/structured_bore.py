"""Strict input-only intent schema connecting interpretation to frozen CAD edits."""
import json
import math

from .step_bore_adapter import public_request

KEYS={'operation','units','center_xy_mm','old_diameter_mm','diameter_mm',
      'preserve_other_features','full_toroidal_passage'}
SCHEMA={
    'operation':'resize_vertical_through_bore', 'units':'mm',
    'center_xy_mm':'array of two finite numbers copied from the requested bore center',
    'old_diameter_mm':'positive finite number: original bore diameter',
    'diameter_mm':'positive finite number: requested_value_mm',
    'preserve_other_features':True,
    'full_toroidal_passage':'boolean: true only when the instruction states a full circular toroidal passage',
}


def validate(value):
    if not isinstance(value,dict) or set(value)!=KEYS:raise ValueError('Intent keys do not match the schema')
    if value['operation']!='resize_vertical_through_bore' or value['units']!='mm':raise ValueError('Unsupported operation or unit')
    center=value['center_xy_mm']
    if not isinstance(center,list) or len(center)!=2:raise ValueError('Expected a two-coordinate center')
    def number(x):return type(x) in (int,float) and math.isfinite(x) and abs(x)<=100000
    if not all(number(x) for x in center):raise ValueError('Invalid center coordinates')
    for key in ('old_diameter_mm','diameter_mm'):
        if not number(value[key]) or value[key]<=0:raise ValueError('Invalid diameter')
    if value['preserve_other_features'] is not True:raise ValueError('Preservation must be true')
    if type(value['full_toroidal_passage']) is not bool:raise ValueError('Expected a Boolean full-ring flag')
    return value


def parse(text):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('Duplicate JSON key')
            result[key]=value
        return result
    def invalid_constant(value):raise ValueError('Nonfinite JSON constant')
    return validate(json.loads(text,object_pairs_hook=pairs,parse_constant=invalid_constant))


def baseline(public):
    original=public_request(public['instruction'],public['requested_value_mm'])
    return validate({'operation':'resize_vertical_through_bore','units':'mm',
        'center_xy_mm':original['center_xy_mm'],'old_diameter_mm':original['old_diameter_mm'],
        'diameter_mm':original['diameter_mm'],'preserve_other_features':True,
        'full_toroidal_passage':original['full_toroidal_passage']})


def same_intent(a,b):
    for key in KEYS:
        if key=='center_xy_mm':
            if any(not math.isclose(x,y,rel_tol=0,abs_tol=1e-6) for x,y in zip(a[key],b[key])):return False
        elif key in ('old_diameter_mm','diameter_mm'):
            if not math.isclose(a[key],b[key],rel_tol=0,abs_tol=1e-6):return False
        elif a[key]!=b[key]:return False
    return True


def canonical_instruction(intent):
    """Every predicted slot feeds the executor; no original-text fallback."""
    validate(intent)
    def decimal(x):return f'{x:.10f}'.rstrip('0').rstrip('.')
    x,y=map(decimal,intent['center_xy_mm']);old=decimal(intent['old_diameter_mm'])
    result=f'Resize the vertical through-bore at (x,y)=({x},{y}) from diameter {old} mm to value mm. Preserve all independent cuts.'
    if intent['full_toroidal_passage']:result+=' The toroidal passage is a full circular ring.'
    return result


def execute(shape,intent):
    from .step_bore_adapter import edit
    result,_=edit(shape,intent['diameter_mm'],prompt=canonical_instruction(intent))
    return result


def complete_object(text):
    """Early stop at a root closing brace, without repairing malformed JSON."""
    text=text.lstrip()
    if not text.startswith('{'):return False
    depth=0;quoted=False;escaped=False
    for char in text:
        if quoted:
            if escaped:escaped=False
            elif char=='\\':escaped=True
            elif char=='"':quoted=False
        elif char=='"':quoted=True
        elif char=='{':depth+=1
        elif char=='}':
            depth-=1
            if depth==0:return True
    return False


def model_prompt(public,demos):
    text=('Extract a CAD edit intent from the input. Return exactly one JSON object, no prose or code fences. '
          'Use only this schema and preserve all other geometry. Do not confuse radius with diameter. '
          'requested_value_mm is the desired diameter, not a scale or an offset. '
          'The source summary contains measurements of the original STEP only.\nSchema: '+json.dumps(SCHEMA,separators=(',',':'))+'\n')
    for demo in demos:
        text+='Input: '+json.dumps(demo,separators=(',',':'))+'\nOutput JSON: '+json.dumps(baseline(demo),separators=(',',':'))+'\n\n'
    return text+'Input: '+json.dumps(public,separators=(',',':'))+'\nOutput JSON:'
