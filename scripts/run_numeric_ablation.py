"""Host-only orchestration; all model profiling/training/evaluation via cad.sh."""
import json
from pathlib import Path
import subprocess


def main():
    state=Path('.runtime/numeric-ablation-supervisor.json')
    stages=[('tests','control','inspect'),('gpu_profile','profile','train'),
        ('smoke_control','control','smoke'),('smoke_weighted','weighted','smoke'),
        ('parent_evaluation','parent','train'),('control','control','train'),
        ('weighted','weighted','train'),('report','report','train')]
    for phase,arm,mode in stages:
        state.write_text(json.dumps({'status':'running','phase':phase})+'\n')
        code=subprocess.run(['./scripts/cad.sh','lossless-numeric-ablation','--numeric-arm',arm,'--pilot-mode',mode]).returncode
        if code:
            state.write_text(json.dumps({'status':'failed','phase':phase,'returncode':code})+'\n');raise SystemExit(code)
    state.write_text(json.dumps({'status':'complete'})+'\n')


if __name__=='__main__':main()
