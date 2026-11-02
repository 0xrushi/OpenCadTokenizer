import json, sys
sys.path.insert(0, '/workspace')
from pathlib import Path
from cadtokenizer.hybrid_tokenizer import HybridTokenizer
from cadtokenizer.scale_origin_experiment import numerical_context
base = Path('artifacts/tokenizer-scale-origin-codec-v1')
saved = {(r['id'], r['arm']): r for r in json.load(open(base / 'decoded.json'))}
inputs = json.load(open(base / 'decode_inputs.json'))
with numerical_context():
    for v in inputs['variants']:
        if v['arm'] == 'lossless': continue
        tok = HybridTokenizer(v['checkpoint'], v['batch_size'])
        same = diff = 0
        for s in v['streams']:
            ref = saved[(s['id'], s['arm'])]
            try: out, err = tok.decode(Path(s['packet']).read_bytes()), None
            except ValueError as e: out, err = None, str(e)
            ok = (out == ref['sequence']) if ref['error'] is None else (err is not None)
            same += ok; diff += not ok
        print(v['arm'], 'identical', same, 'different', diff, flush=True)
