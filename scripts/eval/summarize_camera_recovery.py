"""Publishable summary of the camera-recovery comparison.

Keeps counts, denominators, per-clip camera evidence and receipts; keeps no research frames.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def cohort(comparison: dict, candidate: dict) -> dict:
    camera = {clip["sequence"]: clip["provenance"].get("camera") for clip in candidate["clips"]}
    return {
        "clips": [
            {
                "sequence": clip["sequence"],
                "truth_visible": clip["baseline"]["truth_visible"],
                "truth_absent": clip["baseline"]["truth_absent"],
                "legacy_correct": clip["baseline"]["correct_at_iou_0_5"],
                "compensated_correct": clip["candidate"]["correct_at_iou_0_5"],
                "gained_correct": clip["gained_correct"],
                "lost_correct": clip["lost_correct"],
                "legacy_false_visible": clip["baseline"]["false_visible_when_absent"],
                "compensated_false_visible": clip["candidate"]["false_visible_when_absent"],
                "legacy_abstentions": clip["baseline"]["abstention_reasons"],
                "compensated_abstentions": clip["candidate"]["abstention_reasons"],
                "camera": camera.get(clip["sequence"]),
            }
            for clip in comparison["clips"]
        ],
        "legacy": comparison["baseline"],
        "compensated": comparison["candidate"],
        "legacy_index_seconds": round(comparison["baseline_index_seconds"], 1),
        "compensated_index_seconds": round(comparison["candidate_index_seconds"], 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--cohorts", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_bytes())
    cohorts, receipts = {}, {}
    for name in args.cohorts:
        paths = {key: args.directory / f"{name}-{key}.json" for key in ("legacy", "compensated")}
        paths["compare"] = args.directory / f"compare-{name}.json"
        reports = {key: json.loads(path.read_bytes()) for key, path in paths.items()}
        for path in paths.values():
            receipts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if reports["compare"].get("status") != "complete":
            raise SystemExit(f"{name}: comparison is not complete.")
        cohorts[name] = cohort(reports["compare"], reports["compensated"])
    args.output.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "scope": (
                    "Paired camera-evidence comparison on public LaSOT research footage: the "
                    "released latching guard against per-frame camera-pose tracking. Correlated "
                    "research frames, not original recordings, a user study, or proof of "
                    "physical identity. Region naming is not scored here; the annotation zone is "
                    "the third of the current frame the truth box falls in, which is the naive "
                    "pixel reading compensation is designed to disagree with."
                ),
                "summarized_at": datetime.now(UTC).isoformat(),
                "frozen_at": freeze["frozen_at"],
                "change": freeze["change"],
                "arms": freeze["arms"],
                "scoring": freeze["scoring"],
                "backend": freeze["backend"],
                "runtime": freeze["runtime"],
                "cohorts": cohorts,
                "freeze_sha256": hashlib.sha256(args.freeze.read_bytes()).hexdigest(),
                "report_sha256": receipts,
            },
            indent=2,
        )
        + "\n"
    )
    for name, data in cohorts.items():
        legacy, new = data["legacy"], data["compensated"]
        print(
            f"{name:13s} {legacy['correct_at_iou_0_5']:5d} -> {new['correct_at_iou_0_5']:5d}"
            f" / {legacy['truth_visible']:5d} visible"
            f" | precision {legacy['precision_at_iou_0_5']:.4f} -> {new['precision_at_iou_0_5']:.4f}"
            f" | absent {legacy['truth_absent']:3d}"
            f" false-visible {legacy['false_visible_when_absent']} -> {new['false_visible_when_absent']}"
        )


if __name__ == "__main__":
    main()
