#!/usr/bin/env bash
# Start the two loopback model services in sequence on the approved Spark runtime.
# vLLM profiles available memory during startup; overlapping model loads can make
# the second engine attribute the first engine's allocations to itself and fail.
set -euo pipefail
cd "$(dirname "$0")/../.."

NEMOTRON_DIR="${NEMOTRON_DIR:-$HOME/models/Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
COSMOS_DIR="${COSMOS_DIR:-$HOME/models/Cosmos-Reason2-8B}"
COSMOS_SERVED_NAME="${COSMOS_SERVED_NAME:-nvidia/Cosmos-Reason2-8B}"
COSMOS_LORA_PATH="${COSMOS_LORA_PATH:-}"
COSMOS_LORA_NAME="${COSMOS_LORA_NAME:-findback-grounding}"
NEMOTRON_GPU_FRACTION="${NEMOTRON_GPU_FRACTION:-0.3}"
COSMOS_GPU_FRACTION="${COSMOS_GPU_FRACTION:-0.3}"
WAIT_SECONDS="${AGENTX_MODEL_START_WAIT_SECONDS:-420}"
LOG_DIR="${AGENTX_MODEL_LOG_DIR:-$HOME/agentx-logs}"

for program in tmux curl python3; do
  command -v "$program" >/dev/null || { echo "$program is required." >&2; exit 1; }
done
test -x "${VLLM_VENV:-$HOME/envs/vllm}/bin/vllm" || {
  echo "The approved vLLM environment is missing." >&2; exit 1;
}
test -f "$NEMOTRON_DIR/config.json" && test -f "$COSMOS_DIR/config.json" || {
  echo "One or both local model checkpoints are missing." >&2; exit 1;
}
if [[ -n "$COSMOS_LORA_PATH" && ! -f "$COSMOS_LORA_PATH/adapter_config.json" ]]; then
  echo "The selected Cosmos adapter is missing." >&2
  exit 1
fi
mkdir -p "$LOG_DIR"

serves_model() {
  local port="$1" name="$2"
  curl --fail --silent --show-error --max-time 3 "http://127.0.0.1:$port/v1/models" 2>/dev/null |
    python3 -c 'import json, sys
try:
    models = json.load(sys.stdin).get("data", [])
    raise SystemExit(0 if any(isinstance(model, dict) and model.get("id") == sys.argv[1] for model in models) else 1)
except (ValueError, AttributeError):
    raise SystemExit(1)' "$name" 2>/dev/null
}

wait_for_model() {
  local session="$1" port="$2" name="$3" log="$4"
  local deadline=$((SECONDS + WAIT_SECONDS))
  until serves_model "$port" "$name"; do
    if ! tmux has-session -t "=$session" 2>/dev/null; then
      echo "$session exited before serving $name. Inspect $log." >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "$session did not serve $name within ${WAIT_SECONDS}s. Inspect $log." >&2
      return 1
    fi
    sleep 3
  done
  echo "$session ready: $name on 127.0.0.1:$port"
}

if ! serves_model 8000 nemotron; then
  if ! tmux has-session -t '=agentx-nemotron' 2>/dev/null; then
    printf -v nemotron_cmd 'bash scripts/spark/serve_nemotron.sh %q > %q 2>&1' \
      "$NEMOTRON_GPU_FRACTION" "$LOG_DIR/nemotron-current.log"
    tmux new-session -d -s agentx-nemotron -c "$PWD" "$nemotron_cmd"
  fi
  wait_for_model agentx-nemotron 8000 nemotron "$LOG_DIR/nemotron-current.log"
else
  echo 'Nemotron ready: nemotron on 127.0.0.1:8000'
fi

if ! serves_model 8002 "$COSMOS_SERVED_NAME"; then
  if ! tmux has-session -t '=agentx-cosmos8b' 2>/dev/null; then
    printf -v cosmos_cmd 'COSMOS_LORA_PATH=%q COSMOS_LORA_NAME=%q bash scripts/spark/serve_cosmos.sh %q %q 8002 %q > %q 2>&1' \
      "$COSMOS_LORA_PATH" "$COSMOS_LORA_NAME" "$COSMOS_DIR" "$COSMOS_SERVED_NAME" \
      "$COSMOS_GPU_FRACTION" "$LOG_DIR/cosmos8b-current.log"
    tmux new-session -d -s agentx-cosmos8b -c "$PWD" "$cosmos_cmd"
  fi
  wait_for_model agentx-cosmos8b 8002 "$COSMOS_SERVED_NAME" "$LOG_DIR/cosmos8b-current.log"
else
  echo "Cosmos ready: $COSMOS_SERVED_NAME on 127.0.0.1:8002"
fi
if [[ -n "$COSMOS_LORA_PATH" ]]; then
  if ! serves_model 8002 "$COSMOS_LORA_NAME"; then
    echo "Cosmos does not serve adapter $COSMOS_LORA_NAME. Restart only its project session with the adapter configured." >&2
    exit 1
  fi
  echo "Cosmos adapter ready: $COSMOS_LORA_NAME on 127.0.0.1:8002"
fi
