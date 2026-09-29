"""CUDA integration of companion isolation and a later explicit registration.

Uses previously inspected cup-20 research footage. The later reference is an
RT-DETR candidate in the registration frame, not a later ground-truth annotation.
This checks causal bookkeeping, not new real-world accuracy.
"""

import argparse
import json
from pathlib import Path

from agentx.config import Settings
from agentx.vision.distractors import propose_distractors
from agentx.vision.sam2 import Sam2Detector
from agentx.vision.video import frames, read_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("artifacts/datasets/lasot-prepared/cup-20")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = Settings(sam2_distractor_guard=True, identity_enabled=True)
    source = args.dataset / "cup-20.mp4"
    reference = json.loads((args.dataset / "labels.json").read_text())["labels"][0]["box"]
    primary = dict(id="primary", label="cup", registered_at_ms=0, box=reference)
    _, at_registration = read_frame(source, 2000)
    candidates = propose_distractors(at_registration, [primary], settings, "cuda")
    if not candidates:
        raise ValueError("The frozen smoke input must contain a candidate.")
    later = {**candidates[0], "id": "later-confirmed", "registered_at_ms": 2000}
    detector = Sam2Detector([primary, later], settings, source)
    rows = []
    for at_ms, frame in frames(source, 5, end_ms=6000):
        detected = detector.detect(frame, at_ms)
        rows.append(
            dict(
                at_ms=at_ms,
                ids=[d.object_id for d in detected],
                reasons={d.object_id: d.reason for d in detected},
            )
        )
    checks = dict(
        no_internal_inventory=all(set(r["ids"]) <= {"primary", "later-confirmed"} for r in rows),
        no_future_registration=all(
            "later-confirmed" not in r["ids"] for r in rows if r["at_ms"] < 2000
        ),
        later_reference_active=all(
            "later-confirmed" in r["ids"] for r in rows if r["at_ms"] >= 2000
        ),
        duplicate_companion_retired=bool(detector.collisions.retired),
        no_false_collision=not detector.collisions.quarantined,
        companion_cap=len(detector.collisions.auxiliary) <= settings.sam2_max_distractors,
        bounded_cache=len(detector.session.processed_frames) <= settings.sam2_memory_frames + 3,
    )
    report = dict(
        passed=all(checks.values()),
        checks=checks,
        rows=rows,
        provenance=detector.provenance,
        retired=sorted(detector.collisions.retired),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
