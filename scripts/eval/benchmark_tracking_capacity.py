"""Measure SAM 2 detector capacity by target count and sampling cadence.

Uses a fixed, generated 1080p image, not an accuracy corpus. Every sample runs the
real configured detector, including presence/appearance checks. File decoding,
database writes, HTTP, and concurrent model requests are outside this benchmark.
Unpaced trials report measured service throughput and a *simulated* serial queue.
--paced instead schedules actual arrivals at the requested rate without dropping
frames, and reports measured completion lag. Warmup is excluded in both modes.
"""

import argparse
import gc
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from agentx.config import Settings
from agentx.vision.sam2 import Sam2Detector


def summarize(values):
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "p95": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "max": ordered[-1],
    }


def queue_lags(service_seconds, fps):
    """Completion minus arrival for a lossless, initially empty serial queue."""
    completed = 0.0
    lags = []
    for index, service in enumerate(service_seconds):
        arrival = index / fps
        completed = max(arrival, completed) + service
        lags.append(completed - arrival)
    return lags


def fixture(count, width=1920, height=1080):
    # Same image for every count; only registered prompts change. Unregistered
    # shapes remain in frame so count cannot change the image encoder's input.
    frame = np.random.default_rng(37).integers(30, 45, (height, width, 3), dtype=np.uint8)
    objects = []
    for index in range(10):
        x, y = 80 + index % 5 * 370, 160 + index // 5 * 500
        color = ((73 * index + 50) % 255, (131 * index + 70) % 255, (37 * index + 100) % 255)
        cv2.rectangle(frame, (x, y), (x + 190, y + 145), color, -1)
        cv2.putText(
            frame, str(index), (x + 50, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4
        )
        if index < count:
            objects.append(
                dict(
                    id=str(index),
                    registered_at_ms=0,
                    box=dict(
                        x1=x / width, y1=y / height, x2=(x + 190) / width, y2=(y + 145) / height
                    ),
                )
            )
    return frame, objects


def available_bytes():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux memory telemetry is required.")


def trial(settings, count, fps, warmup, samples, paced):
    frame, objects = fixture(count)
    init_started = time.perf_counter()
    detector = Sam2Detector(objects, settings, Path("unused-registration-source.mp4"))
    torch = detector.torch
    if detector.device != "cuda":
        raise RuntimeError("This benchmark requires the approved CUDA runtime.")
    torch.cuda.synchronize()
    initialization = time.perf_counter() - init_started
    services, measured_lags, visible = [], [], []
    measured_start = None
    for index in range(warmup + samples):
        if index == warmup:
            torch.cuda.reset_peak_memory_stats()
            measured_start = time.perf_counter()
        deadline = None
        if measured_start is not None and paced:
            deadline = measured_start + (index - warmup) / fps
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = detector.detect(frame, round(index * 1000 / fps))
        torch.cuda.synchronize()
        completed = time.perf_counter()
        if len(result) != count:
            raise RuntimeError("The detector dropped a registered target from the inventory.")
        if index >= warmup:
            services.append(completed - start)
            visible.append(sum(d.box is not None and not d.reason for d in result))
            if deadline is not None:
                measured_lags.append(completed - deadline)
        if index % 32 == 0 and available_bytes() < 8 * 1024**3:
            raise RuntimeError("Stopped: less than 8 GiB available host memory.")
        if len(detector.session.processed_frames) > detector.window + 2:
            raise RuntimeError("The tracking frame cache exceeded its bound.")
    elapsed = time.perf_counter() - measured_start
    lags = measured_lags if paced else queue_lags(services, fps)
    row = dict(
        objects=count,
        sampling_fps=fps,
        samples=samples,
        warmup_samples=warmup,
        source_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
        initialization_seconds=initialization,
        service_fps=samples / sum(services),
        service_ms=summarize([x * 1000 for x in services]),
        measured_wall_seconds=elapsed,
        processed_per_wall_second=samples / elapsed,
        queue_kind="measured paced arrivals" if paced else "simulated serial arrivals",
        completion_lag_ms=summarize([x * 1000 for x in lags]),
        final_completion_lag_ms=lags[-1] * 1000,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cuda_allocated_bytes=torch.cuda.memory_allocated(),
        frame_cache=len(detector.session.processed_frames),
        visible_count_range=[min(visible), max(visible)],
        service_seconds=services,
        completion_lags_seconds=lags,
        provenance=detector.provenance,
    )
    del detector
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    parser.add_argument("--fps", type=float, nargs="+", default=[1, 2, 5])
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--paced", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve receipts: choose a new output path.")
    if (
        args.samples < 32
        or args.warmup < 32
        or args.rounds < 1
        or any(n < 1 or n > 10 for n in args.objects)
        or any(fps not in (1, 2, 5) for fps in args.fps)
    ):
        parser.error("Use 1–10 objects, 1/2/5 FPS, >=32 warmup and measured samples, >=1 round.")
    settings = Settings()
    root = Path(__file__).resolve().parents[2]
    report = dict(
        status="running",
        mode="paced" if args.paced else "unpaced",
        scope=__doc__,
        resolution=[1920, 1080],
        parameters=vars(args) | {"output": str(args.output)},
        source_hashes={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__).resolve(), *sorted((root / "src/agentx/vision").glob("*.py"))]
        },
        trials=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    cases = [(n, fps) for n in args.objects for fps in args.fps]
    save()
    try:
        for repetition in range(args.rounds):
            for count, fps in cases if repetition % 2 == 0 else reversed(cases):
                row = trial(settings, count, fps, args.warmup, args.samples, args.paced)
                row["round"] = repetition + 1
                report["trials"].append(row)
                save()
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in row.items()
                            if k not in ("service_seconds", "completion_lags_seconds", "provenance")
                        }
                    ),
                    flush=True,
                )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    report["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
