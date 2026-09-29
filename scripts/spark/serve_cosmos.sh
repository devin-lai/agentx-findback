#!/usr/bin/env bash
# Serve a local Cosmos-Reason2 checkpoint with vLLM on loopback for the FindBack HTTP review adapter.
# Usage: serve_cosmos.sh <model_dir> <served_name> <port> [gpu_memory_fraction]
# Example: tmux new-session -d -s agentx-cosmos8b "bash scripts/spark/serve_cosmos.sh ~/models/Cosmos-Reason2-8B nvidia/Cosmos-Reason2-8B 8002 0.3"
# No reasoning parser: FindBack strips <think> blocks itself and asks for JSON-only output.
#
# Optional LoRA adapter, served next to the base model from the same process:
#   COSMOS_LORA_NAME=findback-grounding-v1 COSMOS_LORA_PATH=/path/to/adapter serve_cosmos.sh ...
# Requests naming COSMOS_LORA_NAME use the adapter; requests naming <served_name> use the base.
# A second adapter can be served for a paired experiment with
# COSMOS_LORA_EXTRA_NAME and COSMOS_LORA_EXTRA_PATH. The application still chooses
# only the adapter named in its own AGENTX_COSMOS_REFERENCE_ADAPTER_MODEL setting.
# VLLM_EXTRA passes further vLLM flags, for example VLLM_EXTRA=--quantization=fp8.
set -euo pipefail
MODEL_DIR="$1"; NAME="$2"; PORT="$3"; UTIL="${4:-0.3}"
VLLM="${VLLM_VENV:-$HOME/envs/vllm}"
LORA_FLAGS=()
if [[ -n "${COSMOS_LORA_PATH:-}" ]]; then
  LORA_FLAGS+=(--enable-lora --max-lora-rank "${COSMOS_LORA_RANK:-16}"
    --lora-modules "${COSMOS_LORA_NAME:-findback-grounding}=${COSMOS_LORA_PATH}")
fi
if [[ -n "${COSMOS_LORA_EXTRA_PATH:-}" ]]; then
  if [[ -z "${COSMOS_LORA_PATH:-}" || -z "${COSMOS_LORA_EXTRA_NAME:-}" ]]; then
    echo "A second LoRA needs COSMOS_LORA_PATH and COSMOS_LORA_EXTRA_NAME." >&2
    exit 2
  fi
  LORA_FLAGS+=("${COSMOS_LORA_EXTRA_NAME}=${COSMOS_LORA_EXTRA_PATH}"
    --max-loras 2 --max-cpu-loras 2)
fi
exec "$VLLM/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name "$NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --max-model-len 16384 \
  --limit-mm-per-prompt '{"image":8}' \
  --gpu-memory-utilization "$UTIL" \
  --max-num-seqs 4 \
  --dtype bfloat16 \
  "${LORA_FLAGS[@]}" \
  ${VLLM_EXTRA:-}
