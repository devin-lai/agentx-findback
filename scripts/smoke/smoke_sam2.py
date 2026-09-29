"""GPU integration check for late registration and bounded multi-object tracking.

Uses the generated fixture; this checks adapter invariants, not real-world accuracy.
The known raw-model occlusion failure is reported separately by evaluate.py.
"""

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

from agentx.config import Settings
from agentx.services.demo import create_fixture
from agentx.vision.sam2 import Sam2Detector
from agentx.vision.video import frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="agentx-sam2-smoke-") as directory:
        source = Path(directory) / "fixture.mp4"
        objects = create_fixture(source)
        descriptors = [
            dict(id=o.name, registered_at_ms=0 if index == 0 else 3733, box=o.box.model_dump())
            for index, o in enumerate(objects)
        ]
        started = time.monotonic()
        detector = Sam2Detector(descriptors, Settings(), source)
        rows = []
        for at_ms, frame in frames(source, 5):
            result = detector.detect(frame, at_ms)
            rows.append(
                dict(
                    at_ms=at_ms,
                    detections=[
                        dict(
                            object_id=d.object_id,
                            box=d.box.model_dump() if d.box else None,
                            score=d.score,
                            reason=d.reason,
                            track_id=d.track_id,
                        )
                        for d in result
                    ],
                )
            )
        checks = {
            "no_future_registration": all(
                all(d["object_id"] != "Blue remote" for d in r["detections"])
                for r in rows
                if r["at_ms"] < 3733
            ),
            "both_objects_after_registration": all(
                {d["object_id"] for d in r["detections"]} == {"Red toolkit", "Blue remote"}
                for r in rows
                if r["at_ms"] >= 3800
            ),
            "stable_remote_after_registration": all(
                any(
                    d["object_id"] == "Blue remote" and d["box"] and not d["reason"]
                    for d in r["detections"]
                )
                for r in rows
                if r["at_ms"] >= 3800
            ),
            "distinct_object_ids": all(
                len({d["track_id"] for d in r["detections"]}) == len(r["detections"]) for r in rows
            ),
            "bounded_frame_cache": len(detector.session.processed_frames) <= detector.window + 3,
        }
        report = dict(
            passed=all(checks.values()),
            checks=checks,
            seconds=time.monotonic() - started,
            provenance=detector.provenance,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            rows=rows,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)
        if not report["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
