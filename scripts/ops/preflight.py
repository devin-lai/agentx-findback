"""Read-only deployment checks. Never installs packages or changes host configuration."""

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path


def inspect(root: Path, require_cuda: bool) -> dict:
    disk = shutil.disk_usage(root)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            from agentx.vision.video import ffmpeg_executable

            ffmpeg = ffmpeg_executable()
        except ImportError:
            ffmpeg = None
    checks = {
        "python_supported": (3, 12) <= sys.version_info[:2] < (3, 14),
        "ffmpeg_available": bool(ffmpeg),
        "disk_at_least_20_percent_free": disk.free / disk.total >= 0.2,
        "frontend_built": (root / "web/dist/index.html").is_file(),
        "app_installed": importlib.util.find_spec("agentx") is not None,
    }
    report = {
        "checks": checks,
        "disk_free_gib": round(disk.free / 1024**3, 1),
        "ffmpeg": ffmpeg,
        "cuda_test_requested": require_cuda,
    }
    if require_cuda:
        checks["cuda_tensor_execution"] = False
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            tensor = torch.ones((32, 32), device="cuda")
            value = (tensor @ tensor).sum().item()
            torch.cuda.synchronize()
            checks["cuda_tensor_execution"] = value == 32768
            report["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "capability": list(torch.cuda.get_device_capability(0)),
            }
        except (ImportError, RuntimeError, OSError):
            report["gpu_error"] = (
                "CUDA execution failed. Use the organizer-approved environment; do not change host drivers."
            )
    report["ready"] = all(checks.values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    result = inspect(Path(__file__).resolve().parents[2], args.require_cuda)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ready"] else 1)


if __name__ == "__main__":
    main()
