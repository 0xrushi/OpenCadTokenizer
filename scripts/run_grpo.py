"""Host orchestration only; run all GRPO/model tests inside the cached image."""
import json
from pathlib import Path
import subprocess


def main():
    status=Path('.runtime/lossless-grpo-supervisor.json')
    for mode in ('inspect','smoke','train'):
        status.write_text(json.dumps({'status':'running','phase':mode})+'\n')
        result=subprocess.run(['./scripts/cad.sh','lossless-grpo','--pilot-mode',mode])
        if result.returncode:
            status.write_text(json.dumps({'status':'failed','phase':mode,'returncode':result.returncode})+'\n')
            raise SystemExit(result.returncode)
    status.write_text(json.dumps({'status':'complete'})+'\n')


if __name__=='__main__':
    main()
