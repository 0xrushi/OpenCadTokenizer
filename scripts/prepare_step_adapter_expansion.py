"""Freeze and fetch all remaining supported TRAIN bores, without answer code."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path('artifacts/step-bore-expansion-v1')
DATA=Path('data/external/cad-preserve-bore-expansion-v0.1.0')
ORIGINAL=Path('data/external/cad-preserve-train-v0.1.0')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    rows=[json.loads(line) for line in (ORIGINAL/'data/train.jsonl').read_text().splitlines()]
    pilot={r['id'] for r in json.loads((ORIGINAL/'pilot_records.json').read_text())}
    selected=sorted([r for r in rows if r['family'] in ('bore_resize','bore_resize_complex') and r['id'] not in pilot],key=lambda r:r['id'])
    assert len(selected)==18 and all(r['split']=='train' and r['units']=='mm' for r in selected)
    assert len({r['geometry_fingerprint'] for r in selected})==18
    revision=json.loads((ORIGINAL/'download_manifest.json').read_text())['revision']
    selection={'dataset':'harrrshall/cad-preserve','revision':revision,'selection':'All remaining TRAIN parts in the two supported bore families, excluding both development cases. No model/outcome selection.',
        'sources':18,'targets':72,'records':selected,'excluded_development_ids':sorted(pilot),
        'frozen_code':{p:digest(p) for p in ['cadtokenizer/step_bore_adapter.py','cadtokenizer/step_adapter_study.py','tests/test_step_bore_adapter.py']},
        'train_metadata_sha256':digest(ORIGINAL/'data/train.jsonl'),
        'contract':'Unchanged adapter; fixed published per-direction volume tolerance, functional probes, same controls and identity/translation checks. Report failures without fixing on these cases.'}
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'selection.json').exists():
        assert json.loads((ROOT/'selection.json').read_text())==selection,'Frozen selection/code changed'
    else:write(ROOT/'selection.json',selection)
    write(ROOT/'state.json',{'status':'running','phase':'downloading_train_only','sources':18,'targets':72})
    paths=sorted({p for row in selected for p in [row['input_file'],*row['gold_files'],*row['check_files']]})
    def fetch(path):
        p=Path(path)
        assert not p.is_absolute() and '..' not in p.parts and p.parts[0]=='cases'
        assert p.parts[1] in {r['id'] for r in selected}
        assert p.suffix in ('.step','.json')
        url=f'https://huggingface.co/datasets/harrrshall/cad-preserve/resolve/{revision}/{path}'
        output=DATA/path;output.parent.mkdir(parents=True,exist_ok=True)
        if not output.exists():
            request=urllib.request.Request(url,headers={'User-Agent':'CadTokenizer-train-only-adapter-check'})
            with urllib.request.urlopen(request,timeout=60) as response:body=response.read(10*1024*1024+1)
            if len(body)>10*1024*1024:raise ValueError('Unexpected file size')
            tmp=output.with_suffix(output.suffix+'.part');tmp.write_bytes(body);tmp.replace(output)
        return {'path':path,'sha256':digest(output),'bytes':output.stat().st_size,'url':url}
    files=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,item in enumerate(pool.map(fetch,paths),1):
            files.append(item)
            write(ROOT/'state.json',{'status':'running','phase':'downloading_train_only','done':i,'total':len(paths)})
    manifest={'revision':revision,'files':files,'scope':'18 additional training sources, 72 training targets/check files; no reference Python, generator, validation or test assets.'}
    if (DATA/'download_manifest.json').exists():assert json.loads((DATA/'download_manifest.json').read_text())==manifest
    else:write(DATA/'download_manifest.json',manifest)
    write(ROOT/'state.json',{'status':'prepared','sources':18,'targets':72,'files':len(files)})
    print('Frozen 18 new training sources and 72 targets; fetched',len(files),'files. Adapter unchanged.',flush=True)


if __name__=='__main__':main()
