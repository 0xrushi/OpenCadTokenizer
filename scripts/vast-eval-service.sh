#!/usr/bin/env bash
set -euo pipefail
source /opt/supervisor-scripts/utils/environment.sh
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CADTOKENIZER_RUNTIME=vastai CAD_NPROC=1
export HF_HOME="$ROOT/.cache/huggingface" PYTHONPATH="$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec /venv/main/bin/python scripts/vast_evaluate.py "$@"
