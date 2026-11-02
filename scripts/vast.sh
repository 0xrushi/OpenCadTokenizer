#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -f /.dockerenv || -z "${VAST_CONTAINERLABEL:-}" ]]; then
  echo 'This launcher is only for the authorized Vast.ai container. Locally use scripts/cad.sh.' >&2
  exit 1
fi
export CADTOKENIZER_RUNTIME=vastai
export HF_HOME="$ROOT/.cache/huggingface"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 WANDB_DISABLED=true
PYTHON_BIN="${VAST_PYTHON:-/venv/main/bin/python}"
COMMAND=("$PYTHON_BIN" -m cadtokenizer "$@")
if [[ "${CAD_NPROC:-1}" -gt 1 ]]; then
  if [[ "${1:-}" != train-sft ]]; then
    echo 'CAD_NPROC > 1 is supported for train-sft only.' >&2
    exit 1
  fi
  COMMAND=("$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$CAD_NPROC" -m cadtokenizer "$@")
fi
exec "${COMMAND[@]}"
