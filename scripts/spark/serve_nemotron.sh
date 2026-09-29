#!/usr/bin/env bash
# Serve NVIDIA Nemotron 3.5 Lightning 30B-A3B (NVFP4) with vLLM on loopback as the FindBack planner.
# Usage: tmux new-session -d -s agentx-nemotron "bash scripts/spark/serve_nemotron.sh 0.3"
set -euo pipefail
# FlashInfer JIT checks for nvcc. The approved CUDA toolkit is installed here, but
# non-interactive SSH/tmux shells may omit its bin directory from PATH.
if [[ -x /usr/local/cuda/bin/nvcc ]]; then
  export PATH="/usr/local/cuda/bin:$PATH"
fi
MODEL_DIR="${NEMOTRON_DIR:-$HOME/models/Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
VLLM="${VLLM_VENV:-$HOME/envs/vllm}"
# The application sends thinking_token_budget for typed object selection
# (AGENTX_PLANNER_SELECTION_THINKING_BUDGET). vLLM rejects that field with HTTP 400 unless a
# reasoning parser is configured, and the application then falls back to the deterministic
# resolver for every indirect description. Default to this model family's parser; set
# NEMOTRON_REASONING_PARSER=none to serve without one.
NEMOTRON_REASONING_PARSER="${NEMOTRON_REASONING_PARSER:-nemotron_v3}"
NEMOTRON_EXTRA_FLAGS=()
if [[ "$NEMOTRON_REASONING_PARSER" != "none" ]]; then
  NEMOTRON_EXTRA_FLAGS+=(--reasoning-parser "$NEMOTRON_REASONING_PARSER")
fi
# NVIDIA's Nemotron 3.5 vLLM recipe uses qwen3_coder for OpenAI-style tool calls.
# The FindBack planner sends no tools, so its JSON response contract is unchanged.
# Set NEMOTRON_TOOL_PARSER=none to restore the previous serving profile.
NEMOTRON_TOOL_PARSER="${NEMOTRON_TOOL_PARSER:-qwen3_coder}"
if [[ "$NEMOTRON_TOOL_PARSER" != "none" ]]; then
  NEMOTRON_EXTRA_FLAGS+=(--enable-auto-tool-choice --tool-call-parser "$NEMOTRON_TOOL_PARSER")
fi
exec "$VLLM/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name nemotron \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization "${1:-0.3}" \
  --max-num-seqs 8 \
  "${NEMOTRON_EXTRA_FLAGS[@]}" \
  --trust-remote-code
