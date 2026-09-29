"""Real pinned-weight inference smoke check; not an object-accuracy benchmark."""

import argparse
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from agentx.config import Settings
from agentx.vision.rtdetr import RTDetrDetector

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--require-cuda", action="store_true")
parser.add_argument(
    "--identity",
    action="store_true",
    help="Also load the DINOv2 identity matcher with a synthetic registration crop.",
)
args = parser.parse_args()
settings = Settings()
if args.identity:
    settings.identity_enabled = True
if args.require_cuda:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Use the approved Spark-compatible PyTorch environment; do not change host drivers."
        )
    settings.model_device = "cuda"
started = time.monotonic()
with tempfile.TemporaryDirectory(prefix="agentx-smoke-") as directory:
    if settings.identity_enabled:
        settings.data_dir = Path(directory)
        patch = np.full((64, 90, 3), (181, 103, 43), dtype=np.uint8)
        cv2.putText(patch, "REMOTE", (7, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
        cv2.imwrite(str(Path(directory) / "reference.png"), patch)
    detector = RTDetrDetector(
        [
            {
                "id": "remote-smoke",
                "name": "Smoke remote",
                "label": "remote",
                "registered_at_ms": 0,
                "reference_path": "reference.png",
            }
        ],
        settings,
        fps=5,
    )
    result = detector.detect(np.zeros((360, 640, 3), dtype=np.uint8), 0)
print(
    json.dumps(
        {
            "provenance": detector.provenance,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "smoke_input": "blank frame; validates execution only",
            "detections": [
                {
                    "reason": d.reason,
                    "score": d.score,
                    "identity": d.identity,
                    "box": d.box.model_dump() if d.box else None,
                }
                for d in result
            ],
        },
        indent=2,
    )
)
