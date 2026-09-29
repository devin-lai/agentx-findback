"""Compare complete end-to-end tracking reports on exactly matched evidence."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluate_tracking import summarize


def compare(baseline, candidate, *, require_identical_predictions=False):
    for report in (baseline, candidate):
        if report.get("status") != "complete" or report.get("failed_clips", 0):
            raise ValueError("Both evaluation reports must be complete without failed clips.")
        if len({c["sequence"] for c in report["clips"]}) != len(report["clips"]):
            raise ValueError("Duplicate evaluation sequence.")
        if not report["clips"] or any(
            c["status"] != "complete" or not c["rows"] for c in report["clips"]
        ):
            raise ValueError("Every evaluation clip must be complete and nonempty.")
    original = {c["sequence"]: c for c in baseline["clips"]}
    changed = {c["sequence"]: c for c in candidate["clips"]}
    if original.keys() != changed.keys():
        raise ValueError("Both variants must contain the same sequences.")
    clips = []
    for name, first in original.items():
        second = changed[name]
        for field in ("video_sha256", "labels_sha256"):
            if first[field] != second[field]:
                raise ValueError("Both variants must use the same source video and labels.")
        a, b = first["rows"], second["rows"]
        if [r["at_ms"] for r in a] != [r["at_ms"] for r in b]:
            raise ValueError("Both variants must score identical timestamps.")
        for old, new in zip(a, b, strict=True):
            if any(old[k] != new[k] for k in ("truth_visible", "truth_box", "truth_zone")):
                raise ValueError("Ground truth differs between paired rows.")
        prediction_fields = (
            "predicted_visible",
            "predicted_box",
            "predicted_zone",
            "reason",
            "detector_score",
            "identity_score",
        )
        if require_identical_predictions and any(
            key not in row for row in [*a, *b] for key in prediction_fields
        ):
            raise ValueError("Exact comparison needs every prediction field in both reports.")
        changed_rows = sum(
            any(x.get(key) != y.get(key) for key in prediction_fields)
            for x, y in zip(a, b, strict=True)
        )
        if require_identical_predictions and changed_rows:
            raise ValueError(f"{name}: {changed_rows} prediction rows differ.")

        def correct(row):
            return row["truth_visible"] and row["predicted_visible"] and row["iou"] >= 0.5

        clips.append(
            dict(
                sequence=name,
                baseline=summarize(a),
                candidate=summarize(b),
                gained_correct=sum(
                    not correct(x) and correct(y) for x, y in zip(a, b, strict=True)
                ),
                lost_correct=sum(correct(x) and not correct(y) for x, y in zip(a, b, strict=True)),
                identical_prediction_rows=len(a) - changed_rows,
                changed_prediction_rows=changed_rows,
            )
        )
    return dict(
        status="complete",
        clips=clips,
        baseline=summarize([r for c in original.values() for r in c["rows"]]),
        candidate=summarize([r for c in changed.values() for r in c["rows"]]),
        baseline_index_seconds=sum(c["index_seconds"] for c in original.values()),
        candidate_index_seconds=sum(c["index_seconds"] for c in changed.values()),
        identical_prediction_rows=sum(c["identical_prediction_rows"] for c in clips),
        changed_prediction_rows=sum(c["changed_prediction_rows"] for c in clips),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-identical-predictions",
        action="store_true",
        help="Fail if any box, visibility, region, reason or confidence changes.",
    )
    args = parser.parse_args()
    result = compare(
        json.loads(args.baseline.read_text()),
        json.loads(args.candidate.read_text()),
        require_identical_predictions=args.require_identical_predictions,
    )
    result["report_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.baseline, args.candidate)
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
