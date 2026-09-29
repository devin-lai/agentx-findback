"""Bounded-memory GPU soak with ten registered objects and a fixed synthetic frame.

Samples have 5 FPS source timestamps but run unpaced: this does not establish
sustained 5 FPS wall-clock processing or perception accuracy.
It never contacts external inference services. A low available-memory guard stops
the experiment so it does not crowd the existing project model servers.
"""

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import cv2
import numpy as np

from agentx.config import Settings
from agentx.vision.sam2 import Sam2Detector


def available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot inspect available memory on this Linux host.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--objects", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 200:
        raise SystemExit("Use at least 200 samples for a soak check.")
    frame = np.random.default_rng(37).integers(30, 45, (480, 800, 3), dtype=np.uint8)
    descriptors = []
    for index in range(args.objects):
        x, y = 35 + (index % 5) * 155, 75 + (index // 5) * 220
        color = ((73 * index + 50) % 255, (131 * index + 70) % 255, (37 * index + 100) % 255)
        cv2.rectangle(frame, (x, y), (x + 80, y + 65), color, -1)
        cv2.putText(
            frame, str(index), (x + 20, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        descriptors.append(
            dict(
                id=str(index),
                registered_at_ms=0,
                box=dict(x1=x / 800, y1=y / 480, x2=(x + 80) / 800, y2=(y + 65) / 480),
            )
        )
    detector = Sam2Detector(descriptors, Settings(), Path("unused-registration-source.mp4"))
    report = dict(
        status="running",
        scope="Synthetic fixed-frame resource soak; no accuracy claim",
        samples_requested=args.samples,
        objects=args.objects,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        provenance=detector.provenance,
        checkpoints=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for index in range(args.samples):
        result = detector.detect(frame, index * 200)
        if len(result) != args.objects:
            raise RuntimeError("An active object was dropped from the observation inventory.")
        if index % 100 == 0 or index == args.samples - 1:
            available = available_bytes()
            checkpoint = dict(
                sample=index,
                seconds=time.monotonic() - started,
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                available_bytes=available,
                cuda_allocated_bytes=detector.torch.cuda.memory_allocated(),
                cuda_reserved_bytes=detector.torch.cuda.memory_reserved(),
                frame_cache=len(detector.session.processed_frames),
                object_cache=max(
                    len(c["non_cond_frame_outputs"])
                    for c in detector.session.output_dict_per_obj.values()
                ),
                visible=sum(d.box is not None and not d.reason for d in result),
            )
            report["checkpoints"].append(checkpoint)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(checkpoint), flush=True)
            if available < 8 * 1024**3:
                report["status"] = "stopped_low_available_memory"
                break
            if checkpoint["frame_cache"] > detector.window + 2:
                report["status"] = "failed_unbounded_cache"
                break
    else:
        report["status"] = "complete"
    report["seconds"] = time.monotonic() - started
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
