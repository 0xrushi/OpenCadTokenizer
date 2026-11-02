#!/usr/bin/env bash
set -euo pipefail
source /opt/supervisor-scripts/utils/environment.sh
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CAD_NPROC="${CAD_NPROC:-4}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec bash scripts/vast.sh train-sft --config "${CAD_CONFIG:-configs/vastai.json}" "$@"
