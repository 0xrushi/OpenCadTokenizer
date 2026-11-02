"""Host orchestration only: gate the local training job on a container smoke run."""
import json
from pathlib import Path
import subprocess


def main():
    state = Path('.runtime/lossless-continuation-supervisor.json')
    for mode in ('smoke', 'train'):
        state.write_text(json.dumps({'status':'running','phase':mode})+'\n')
        result = subprocess.run(['./scripts/cad.sh','lossless-continue','--pilot-mode',mode])
        if result.returncode:
            state.write_text(json.dumps({'status':'failed','phase':mode,'returncode':result.returncode})+'\n')
            raise SystemExit(result.returncode)
    state.write_text(json.dumps({'status':'complete'})+'\n')


if __name__ == '__main__':
    main()
