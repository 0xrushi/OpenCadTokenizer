"""Host supervision only; all model tests and profiling execute via cad.sh."""
import json
from pathlib import Path
import subprocess


def main():
    status=Path('.runtime/routed-benchmark-supervisor.json')
    for mode in ('inspect','train'):
        status.write_text(json.dumps({'status':'running','phase':mode})+'\n')
        result=subprocess.run(['./scripts/cad.sh','lossless-router-benchmark','--pilot-mode',mode])
        if result.returncode:
            status.write_text(json.dumps({'status':'failed','phase':mode,'returncode':result.returncode})+'\n')
            raise SystemExit(result.returncode)
    status.write_text(json.dumps({'status':'complete'})+'\n')


if __name__=='__main__':main()
