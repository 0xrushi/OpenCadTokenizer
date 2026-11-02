#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-configs/train.json}"
"$ROOT/scripts/cad.sh" prepare --config "$CONFIG"
"$ROOT/scripts/cad.sh" train-vq --config "$CONFIG"
"$ROOT/scripts/cad.sh" train-align --config "$CONFIG"
"$ROOT/scripts/cad.sh" encode --config "$CONFIG"
CAD_NPROC="${SFT_NPROC:-1}" "$ROOT/scripts/cad.sh" train-sft --config "$CONFIG"
