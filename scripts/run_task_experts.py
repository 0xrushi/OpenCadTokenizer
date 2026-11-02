"""Host-only supervisor; tests, training and inference all execute via cad.sh."""
import json
from pathlib import Path
import subprocess


def main():
    state=Path('.runtime/task-experts-supervisor.json')
    stages=[('tests','joint','inspect'),('smoke','joint','smoke'),
            ('joint','joint','train'),('editing','editing','train'),
            ('generation','generation','train'),('routing_and_report','report','train')]
    for phase,arm,mode in stages:
        state.write_text(json.dumps({'status':'running','phase':phase})+'\n')
        result=subprocess.run(['./scripts/cad.sh','lossless-experts','--expert-arm',arm,'--pilot-mode',mode])
        if result.returncode:
            state.write_text(json.dumps({'status':'failed','phase':phase,'returncode':result.returncode})+'\n')
            raise SystemExit(result.returncode)
    state.write_text(json.dumps({'status':'complete'})+'\n')


if __name__=='__main__':
    main()
