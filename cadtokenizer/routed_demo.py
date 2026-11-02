"""Re-run illustrative saved validation examples, export CAD and render previews."""
from concurrent.futures import ThreadPoolExecutor
import difflib
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

from .common import require_container,read_config,write_json,file_hash

ROOT=Path('artifacts/routed-demo-v1')


def choose_examples():
    gen_root=Path('artifacts/lossless-generation-validation-v1')
    edit_root=Path('artifacts/lossless-editing-validation-v1')
    gm={c['example_id']:c for c in read_config(gen_root/'manifest.json')['cases']}
    em={c['example_id']:c for c in read_config(edit_root/'manifest.json')['cases']}
    gr=read_config(gen_root/'per_case.json')['expert']
    er=read_config(edit_root/'per_case.json')
    source_valid={r['id']:r['prediction_geometry']['status']=='valid' for r in er['source_copy']}
    valid=sorted((r for r in gr if r['chamfer'] is not None),key=lambda r:(r['chamfer'],r['id']))
    median=valid[len(valid)//2]
    exact=sorted((r for r in er['lossless'] if r['exact_sem'] and r['chamfer'] is not None
                  and source_valid[r['id']] and em[r['id']]['instruction'].lower().startswith('remove ')),
                 key=lambda r:(len(em[r['id']]['instruction']),r['id']))[0]
    numeric=next(r for r in er['lossless'] if r['id']=='cad_editor:train:71552')
    if numeric['chamfer'] is None or numeric['exact_sem']:raise ValueError('Expected the documented valid numeric-error illustration')
    return [
        {'name':'generation','selection':'Median recorded shape error among valid generation/reference pairs; illustrative, not randomly sampled.',
         'case':gm[median['id']],'expected':median},
        {'name':'editing_remove','selection':'Shortest recorded exact, valid removal instruction with valid source; deliberate success illustration.',
         'case':em[exact['id']],'expected':exact},
        {'name':'editing_add','selection':'Previously diagnosed addition with correct syntax but wrong numeric parameters; deliberate limitation illustration.',
         'case':em[numeric['id']],'expected':numeric}]


def render_previews(reports):
    import cadquery as cq
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import textwrap
    roles=('source','output','reference')
    colors={'source':'#89a9d0','output':'#62b5ad','reference':'#d5ad76'}
    meshes={};frames={}
    for report in reports:
        verts=[]
        for role,geometry in report['geometry'].items():
            shape=cq.importers.importStep(str(ROOT/report['name']/f'{role}.step')).val()
            points,triangles=shape.tessellate(.002,.5)
            points=np.asarray([p.toTuple() for p in points]);triangles=np.asarray(triangles,dtype=int)
            if len(triangles)>100000:raise ValueError('Demo mesh exceeds visualization budget')
            meshes[(report['name'],role)]=(points,triangles);verts.append(points)
        all_vertices=np.concatenate(verts)
        center=(all_vertices.min(axis=0)+all_vertices.max(axis=0))/2
        radius=max(float(np.ptp(all_vertices,axis=0).max())*.58,.001)
        frames[report['name']]=(center,radius)
    def draw(ax,report,role):
        if (report['name'],role) not in meshes:
            ax.set_axis_off();ax.text2D(.5,.5,'Text-only generation\nNo input CAD',ha='center',va='center',transform=ax.transAxes)
            return
        points,triangles=meshes[(report['name'],role)]
        ax.add_collection3d(Poly3DCollection(points[triangles],facecolor=colors[role],edgecolor='#435268',linewidth=.08,alpha=1.))
        center,radius=frames[report['name']]
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
        ax.set_proj_type('ortho');ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=45);ax.set_axis_off()
    figure=plt.figure(figsize=(15,12),dpi=140)
    for row,report in enumerate(reports):
        for column,role in enumerate(roles):
            ax=figure.add_subplot(len(reports),3,row*3+column+1,projection='3d')
            draw(ax,report,role)
            ax.set_title({'source':'Original CAD','output':'Model output','reference':'Reference target'}[role],fontsize=12,pad=0)
            if column==0:
                ax.text2D(0.,-.06,textwrap.fill(report['instruction'],48),fontsize=9,transform=ax.transAxes)
        for role in report['geometry']:
            one=plt.figure(figsize=(5,5),dpi=140);ax=one.add_subplot(111,projection='3d')
            draw(ax,report,role);ax.set_title(role.capitalize(),fontsize=12)
            one.savefig(ROOT/report['name']/f'{role}.png',bbox_inches='tight',pad_inches=.03,facecolor='white');plt.close(one)
    figure.suptitle('Routed CAD: kernel-validated outputs and reference comparison',fontsize=16)
    figure.subplots_adjust(left=.02,right=.98,top=.94,bottom=.04,wspace=.02,hspace=.2)
    figure.savefig(ROOT/'overview.png',facecolor='white');plt.close(figure)


def run_demo():
    require_container()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from .task_router import TaskExpertGenerator
    from .bench_runtime import KernelRunner
    from .bench_data import code_hash
    ROOT.mkdir(parents=True,exist_ok=True)
    def status(phase,**extra):write_json(ROOT/'state.json',{'status':'running','phase':phase,**extra})
    with (ROOT/'run.lock').open('a') as lock,sdpa_kernel(SDPBackend.MATH):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if (ROOT/'results.json').exists():return read_config(ROOT/'results.json')
            status('freeze');examples=choose_examples()
            descriptor=Path('artifacts/lossless-routed-v2/router.json')
            manifest={'router_sha256':file_hash(descriptor),'code_sha256':code_hash(),
                'image_id':os.environ.get('CADTOKENIZER_IMAGE_ID'),'examples':examples,
                'limits':'Selected known-validation examples for illustration, including a known numeric failure. Not a new accuracy estimate. No targets supplied to inference. SEM coordinates are discretized/normalized, not guaranteed physical millimetres.'}
            if (ROOT/'manifest.json').exists() and read_config(ROOT/'manifest.json')!=manifest:raise ValueError('Demo provenance changed; choose fresh output')
            write_json(ROOT/'manifest.json',manifest)
            torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
            torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
            status('load_router');router=TaskExpertGenerator(descriptor)
            requests=[{'instruction':e['case']['instruction'],'source':e['case']['source']} for e in examples]
            write_json(ROOT/'requests.json',requests)
            with (ROOT/'requests.jsonl').open('w') as stream:
                for item in requests:stream.write(json.dumps(item)+'\n')
            status('routed_inference');started=time.monotonic()
            predictions=router.predict_many(requests)
            batch_events=list(router.last_batch_stats);fallbacks=[]
            for i,(example,prediction) in enumerate(zip(examples,predictions)):
                expected=example['expected']
                if prediction['sequence']!=expected['sequence'] or prediction['generated_ids']!=expected['generated_ids']:
                    # A finite batch-equivalence sample is not a universal guarantee.
                    fallbacks.append(example['case']['example_id'])
                    predictions[i]=router.predict(**requests[i])
                if predictions[i]['sequence']!=expected['sequence'] or predictions[i]['generated_ids']!=expected['generated_ids']:
                    raise ValueError('Serial routed demonstration differs from independent reference')
            inference_seconds=time.monotonic()-started
            write_json(ROOT/'predictions.json',predictions)
            cfg={**read_config('configs/benchmark.json'),'output':str(ROOT/'kernel-cache'),'render':False}
            runner=KernelRunner(cfg);jobs=[]
            for example,prediction in zip(examples,predictions):
                c=example['case'];key=c['example_id']
                jobs.extend([(key,'output',prediction['sequence']),(key,'reference',c['target'])])
                if c['source'] is not None:jobs.append((key,'source',c['source']))
            status('kernel_validation_and_export');geometry={}
            with ThreadPoolExecutor(max_workers=4) as pool:
                for job,result in zip(jobs,pool.map(lambda j:runner.reconstruct(j[2],j[0],j[1]),jobs)):
                    if result['status']!='valid':raise ValueError('Selected demo failed kernel validation: '+str(job[:2]))
                    geometry[job]=result
            reports=[]
            for example,prediction in zip(examples,predictions):
                c=example['case'];directory=ROOT/example['name'];directory.mkdir(exist_ok=True)
                sequences={'output':prediction['sequence'],'reference':c['target']}
                if c['source'] is not None:sequences['source']=c['source']
                record={'name':example['name'],'id':c['example_id'],'instruction':c['instruction'],
                    'task':c['task'],'selection':example['selection'],'exact_reference_program':prediction['sequence']==c['target'],
                    'geometry':{}}
                (directory/'instruction.txt').write_text(c['instruction']+'\n')
                for role,seq in sequences.items():
                    (directory/f'{role}.sem').write_text(seq+'\n')
                    result=geometry[(c['example_id'],role,seq)]
                    for suffix in ('step','stl'):shutil.copy2(result[suffix],directory/f'{role}.{suffix}')
                    record['geometry'][role]={'status':'valid','volume':result['volume'],'solid_count':result['solid_count'],
                        'step':str(directory/f'{role}.step'),'stl':str(directory/f'{role}.stl'),'preview':str(directory/f'{role}.png')}
                diff='\n'.join(difflib.unified_diff(c['target'].split(),prediction['sequence'].split(),fromfile='reference',tofile='output',lineterm=''))
                (directory/'reference-output.diff').write_text(diff+'\n')
                write_json(directory/'report.json',record);reports.append(record)
            status('render_previews');render_previews(reports)
            readme=['# Routed CAD demonstration','','![CAD comparisons](overview.png)','',manifest['limits'],'',
                'All exported inputs, references and outputs passed CadQuery/OCP solid validity, non-empty-solid and positive-volume checks. This does not establish instruction correctness. Each row uses one common camera scale across its source/output/reference; scales may differ across rows. Preview tessellation is for illustration, separate from evaluation metrics.','']
            for r in reports:
                name=r['name']
                readme += ['## '+name,'',r['instruction'],'',r['selection'],'',
                    'Reference program: '+('exact match.' if r['exact_reference_program'] else 'different; inspect the reference comparison and diff.'),'',
                    f'[Output STEP]({name}/output.step) · [Output STL]({name}/output.stl) · [Output preview]({name}/output.png) · [Reference preview]({name}/reference.png)','']
            readme += ['## Reproduce','','`./scripts/cad.sh lossless-demo`','','The saved requests can also be sent through `./scripts/cad.sh lossless-route --requests artifacts/routed-demo-v1/requests.jsonl --output artifacts/demo-replay.json`. Model inference uses the cached Unsloth image.','']
            (ROOT/'README.md').write_text('\n'.join(readme))
            result={'cases':reports,'all_outputs_kernel_valid':True,'inference_seconds':inference_seconds,
                'batch_events':batch_events,'serial_equivalence_fallback_cases':fallbacks,
                'overview':str(ROOT/'overview.png'),'report':str(ROOT/'README.md'),'limits':manifest['limits']}
            write_json(ROOT/'results.json',result)
            with zipfile.ZipFile(ROOT/'demo-files.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
                for path in [ROOT/'README.md',ROOT/'overview.png',ROOT/'results.json',ROOT/'requests.jsonl',*[p for r in reports for p in (ROOT/r['name']).iterdir() if p.is_file()]]:
                    archive.write(path,path.relative_to(ROOT))
            write_json(ROOT/'state.json',{'status':'complete','results':str(ROOT/'results.json')})
        except Exception as exc:
            write_json(ROOT/'state.json',{'status':'failed','error':repr(exc)});raise
