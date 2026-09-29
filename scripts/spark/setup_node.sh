#!/usr/bin/env bash
# Prepare the FindBack application environment on the assigned DGX Spark node (aarch64, CUDA 13).
# User-level only: no sudo, no driver or system changes. Re-runnable.
#
# The node reaches only the Aliyun PyPI mirror, npmmirror and ModelScope. PyPI, HuggingFace,
# GitHub and astral.sh time out, so:
#   - Python packages come from mirrors.aliyun.com (CUDA-enabled aarch64 torch wheels included);
#   - Cosmos checkpoints come from ModelScope (nv-community/Cosmos-Reason2-*);
#   - small HuggingFace weights (RT-DETR, DINOv2) are relayed from a workstation into
#     ~/.cache/huggingface/hub and used with HF_HUB_OFFLINE=1;
#   - ffmpeg comes from the imageio-ffmpeg wheel (static binary with libx264).
set -euo pipefail
INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
VENV="${AGENTX_VENV:-$HOME/envs/agentx}"
REPO="${AGENTX_REPO:-$HOME/agentx/AgentX}"

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q -U pip uv -i "$INDEX"
fi
cd "$REPO"
UV_INDEX_URL="$INDEX" UV_HTTP_TIMEOUT=300 "$VENV/bin/uv" pip install --python "$VENV/bin/python" \
  -e ".[vision,ffmpeg]" pytest pytest-cov ruff mypy
ln -sfn "$VENV" "$REPO/.venv"
FFMPEG="$("$VENV/bin/python" -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')"
mkdir -p "$HOME/.local/bin"
ln -sf "$FFMPEG" "$HOME/.local/bin/ffmpeg"
ln -sf "$FFMPEG" "$VENV/bin/ffmpeg"
"$VENV/bin/python" -c 'import torch;print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")'
echo "Environment ready at $VENV. Relay cached weights, then run: .venv/bin/python scripts/ops/preflight.py --require-cuda"
