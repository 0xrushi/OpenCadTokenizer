#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-configs/train.json}"
BENCHMARK_CONFIG="${2:-configs/benchmark.json}"
cd "$ROOT"
# Host Python reads only JSON configuration; all model/kernel work stays in Docker.
BENCHMARK_ROOT="$(python3 - "$BENCHMARK_CONFIG" <<'PY'
import json,sys
with open(sys.argv[1]) as stream:
    print(json.load(stream)['output'])
PY
)"
if [[ ! -f "$BENCHMARK_ROOT/manifest.json" ]]; then
  CAD_CPU=1 "$ROOT/scripts/cad.sh" benchmark-freeze --config "$CONFIG" --benchmark-config "$BENCHMARK_CONFIG"
fi
RESUME_ARGS=()
if [[ -f "$BENCHMARK_ROOT/predictions.jsonl" || -f "$BENCHMARK_ROOT/predictions.meta.json" ]]; then
  RESUME_ARGS=(--resume-predictions)
fi
"$ROOT/scripts/cad.sh" benchmark-predict --config "$CONFIG" --benchmark-config "$BENCHMARK_CONFIG" "${RESUME_ARGS[@]}"
CAD_CPU=1 "$ROOT/scripts/cad.sh" benchmark-score --config "$CONFIG" --benchmark-config "$BENCHMARK_CONFIG"
if [[ ! -d "$BENCHMARK_ROOT/judging" ]]; then
  CAD_CPU=1 "$ROOT/scripts/cad.sh" benchmark-judging --config "$CONFIG" --benchmark-config "$BENCHMARK_CONFIG"
fi
