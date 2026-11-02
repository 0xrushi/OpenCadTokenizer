#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "${1:-}" == images ]]; then
  exec docker image ls --format '{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}'
fi
if ! docker info >/dev/null 2>&1; then
  echo 'Cannot access Docker. Run this launcher from a terminal with Docker socket access.' >&2
  exit 1
fi
mkdir -p .runtime .cache/huggingface
if [[ -n "${UNSLOTH_IMAGE:-}" ]]; then
  IMAGE_REF="$UNSLOTH_IMAGE"
elif [[ -f .runtime/unsloth-image-id ]]; then
  IMAGE_REF="$(cat .runtime/unsloth-image-id)"
else
  mapfile -t CANDIDATES < <(docker image ls --format '{{.Repository}}:{{.Tag}} {{.ID}}' | awk 'tolower($1) ~ /unsloth/ {print $2}' | sort -u)
  if [[ ${#CANDIDATES[@]} -ne 1 ]]; then
    echo 'Set UNSLOTH_IMAGE to the already-downloaded Unsloth tag or image ID. Available images:' >&2
    docker image ls --format '{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}' >&2
    exit 1
  fi
  IMAGE_REF="${CANDIDATES[0]}"
fi
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE_REF")"
TAGS="$(docker image inspect --format '{{json .RepoTags}}' "$IMAGE_ID")"
if [[ "${TAGS,,}" != *unsloth* ]]; then
  echo "Selected local image has no Unsloth repository tag: $IMAGE_ID ($TAGS)" >&2
  exit 1
fi
printf '%s\n' "$IMAGE_ID" > .runtime/unsloth-image-id
printf '%s\n' "$TAGS" > .runtime/unsloth-image-tags.json
echo "Using cached Unsloth image: $IMAGE_ID ($TAGS)" >&2
GPU_ARGS=(--gpus all)
if [[ "${CAD_CPU:-0}" == 1 ]]; then GPU_ARGS=(); fi
ENV_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then ENV_ARGS+=(-e HF_TOKEN); fi
PYTHON_BIN=python
case "${1:-}" in
  benchmark-score|benchmark-lenient|benchmark-doctor|benchmark-test|benchmark-judging|benchmark-smoke|tokenizer-experiment|tokenizer-overfit|tokenizer-learning-curve|tokenizer-profile|tokenizer-full|tokenizer-regression|tokenizer-repeatability|tokenizer-continuation|tokenizer-heldout|tokenizer-coverage-eval|tokenizer-coverage-hybrid|tokenizer-topk-check|tokenizer-loop-full|tokenizer-shape-audit|tokenizer-origin-ablation|lossless-prepare|lossless-sft|lossless-compare|lossless-editing-eval|lossless-train-recall|lossless-continue|lossless-grpo|lossless-experts|lossless-generation-eval|lossless-router-benchmark|lossless-router-memory-check|lossless-route|lossless-demo|lossless-numeric-ablation|numeric-instruction-audit|numeric-audit-parity|numeric-curation|addition-curation|cad-source-inspect|step-adapter-test|step-edit|step-adapter-expand|structured-bore-prepare|structured-bore-infer|structured-bore-evaluate|step-intent-edit|structured-bore-text-infer|structured-bore-text-evaluate|tokenizer-scale-origin-encode|tokenizer-scale-origin-decode|tokenizer-scale-origin-evaluate|hybrid-encode|hybrid-decode)
    if [[ -f .runtime/eval-venv/image-id ]]; then
      if [[ "$(cat .runtime/eval-venv/image-id)" != "$IMAGE_ID" ]]; then
        echo 'Evaluation environment belongs to another image; see README.md before re-running benchmark-setup.' >&2
        exit 1
      fi
      PYTHON_BIN=/workspace/.runtime/eval-venv/bin/python
    fi
    ;;
esac
COMMAND=("$PYTHON_BIN" -m cadtokenizer "$@")
PORT_ARGS=()
if [[ "${1:-}" == tensorboard ]]; then
  shift
  DASHBOARD_PORT="${CAD_TENSORBOARD_PORT:-6006}"
  if [[ ! "$DASHBOARD_PORT" =~ ^[0-9]{1,5}$ ]] || (( 10#$DASHBOARD_PORT < 1 || 10#$DASHBOARD_PORT > 65535 )); then
    echo 'CAD_TENSORBOARD_PORT must be an integer from 1 to 65535.' >&2
    exit 1
  fi
  GPU_ARGS=()
  PORT_ARGS=(--publish "127.0.0.1:$DASHBOARD_PORT:6006")
  COMMAND=(python -m tensorboard.main --logdir artifacts --host 0.0.0.0 --port 6006 "$@")
fi
if [[ "${CAD_NPROC:-1}" -gt 1 ]]; then
  if [[ "${1:-}" != train-sft ]]; then
    echo 'CAD_NPROC > 1 is supported for train-sft only; other stages use one process.' >&2
    exit 1
  fi
  COMMAND=(torchrun --standalone --nnodes=1 --nproc-per-node="$CAD_NPROC" -m cadtokenizer "$@")
fi
exec docker run --rm --init --pull=never "${GPU_ARGS[@]}" "${PORT_ARGS[@]}" --ipc=host \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$ROOT,target=/workspace" --workdir /workspace \
  -e CADTOKENIZER_CONTAINER=unsloth -e "CADTOKENIZER_IMAGE_ID=$IMAGE_ID" -e PYTHONPATH=/workspace \
  -e HF_HOME=/workspace/.cache/huggingface -e PYTHONUNBUFFERED=1 \
  -e PIP_CACHE_DIR=/workspace/.cache/pip \
  -e TOKENIZERS_PARALLELISM=false -e WANDB_DISABLED=true \
  "${ENV_ARGS[@]}" --entrypoint /bin/bash "$IMAGE_ID" -c 'exec "$@"' _ "${COMMAND[@]}"
