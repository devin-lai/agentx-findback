#!/usr/bin/env bash
# Download Cosmos-Reason2 checkpoints from ModelScope into ~/models (resumable, ~45 MB/s on the node).
# Nemotron 3.5 Lightning NVFP4 is already provided on the node under ~/models by the organizer scripts.
set -euo pipefail
MSC="${MODELSCOPE_VENV:-$HOME/envs/msc}"
if [ ! -x "$MSC/bin/python" ]; then
  python3 -m venv "$MSC"
  "$MSC/bin/pip" install -q -U pip modelscope -i "${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
fi
"$MSC/bin/python" - "$@" <<'PY'
import sys, time
from modelscope import snapshot_download
sizes = sys.argv[1:] or ["8B"]
for size in sizes:
    t = time.time()
    path = snapshot_download(f"nv-community/Cosmos-Reason2-{size}", local_dir=f"{__import__('os').environ['HOME']}/models/Cosmos-Reason2-{size}", max_workers=8)
    print("downloaded", path, round(time.time() - t, 1), "s", flush=True)
PY
