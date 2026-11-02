#!/usr/bin/env bash
set -euo pipefail
source /opt/supervisor-scripts/utils/environment.sh
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CADTOKENIZER_RUNTIME=vastai CAD_NPROC=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT" TOKENIZERS_PARALLELISM=false
exec /venv/main/bin/python scripts/audit_codebook_failures.py \
  --benchmark artifacts/benchmark-vastai-codebook-v2 --output artifacts/codebook-failure-audit-v2 "$@"
