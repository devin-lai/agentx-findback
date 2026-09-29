"""Compare CUDA mask reduction with the former CPU scan on generated masks.

This measures mask postprocessing, not model inference or recognition accuracy.
Both arms receive the same device-resident logits; warmups are excluded and arm
order alternates. A mismatch fails the run instead of reporting a speedup.
"""

import argparse
import hashlib
import json
import statistics
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from agentx.vision.masks import mask_box, mask_geometry


def reference(logits):
    arrays = [(logits[i] > 0).cpu().numpy() for i in range(len(logits))]
    boxes = [mask_box(mask) for mask in arrays]
    overlaps = set()
    for i, j in combinations(range(len(arrays)), 2):
        union = np.count_nonzero(arrays[i] | arrays[j])
        if union and np.count_nonzero(arrays[i] & arrays[j]) / union >= 0.85:
            overlaps.add((i, j))
    return boxes, overlaps


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior reports: choose a new output path.")
    if args.repeats < 2 or not torch.cuda.is_available():
        parser.error("At least two repetitions and a CUDA device are required.")
    report = {
        "scope": "Generated mask postprocessing only; not end-to-end latency or visual accuracy.",
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "repeats": args.repeats,
        "source_sha256": hashlib.sha256(
            Path(__file__).resolve().parents[2].joinpath("src/agentx/vision/masks.py").read_bytes()
        ).hexdigest(),
        "cases": [],
    }
    arms = {"cpu_scan": reference, "cuda_reduction": lambda logits: mask_geometry(logits > 0)}
    for height, width in [(480, 640), (1080, 1920), (2160, 3840)]:
        for count in (1, 4):
            logits = torch.full((count, height, width), -1.0, device="cuda")
            logits[:, height // 5 : height * 4 // 5, width // 5 : width * 4 // 5] = 1
            if count > 1:
                logits[-1] = -1  # empty target, with three overlapping positive masks
            times: dict[str, list[float]] = {name: [] for name in arms}
            expected = reference(logits)
            for iteration in range(args.repeats + 4):
                order = list(arms) if iteration % 2 else list(reversed(arms))
                for name in order:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    result = arms[name](logits)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    if result != expected:
                        raise RuntimeError(f"Geometry mismatch: {height}x{width}, {count}, {name}")
                    if iteration >= 4:
                        times[name].append(elapsed * 1000)
            medians = {name: statistics.median(values) for name, values in times.items()}
            row = {
                "height": height,
                "width": width,
                "objects": count,
                "exact_geometry": True,
                "median_ms": medians,
                "speedup": medians["cpu_scan"] / medians["cuda_reduction"],
                "timings_ms": times,
            }
            report["cases"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "timings_ms"}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
