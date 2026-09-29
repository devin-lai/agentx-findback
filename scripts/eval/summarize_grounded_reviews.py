"""Summarize complete Cosmos diagnostics, pairing identical windows for ablations."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

COUNTS = (
    "frame_requests",
    "available_predictions",
    "truth_visible",
    "truth_absent",
    "positive_predictions",
    "correct_presence",
    "correct_point_in_box",
    "false_visible_when_absent",
    "outside_target_box_when_visible",
)


def aggregate(windows):
    counts = {key: sum(w["summary"][key] for w in windows) for key in COUNTS}
    counts["point_recall"] = (
        counts["correct_point_in_box"] / counts["truth_visible"]
        if counts["truth_visible"]
        else None
    )
    counts["point_precision"] = (
        counts["correct_point_in_box"] / counts["positive_predictions"]
        if counts["positive_predictions"]
        else None
    )
    return {
        **counts,
        "windows": len(windows),
        "failed_windows": sum(w["status"] != "complete" for w in windows),
        "median_window_seconds": statistics.median(w["elapsed_seconds"] for w in windows)
        if windows
        else None,
        "max_window_seconds": max((w["elapsed_seconds"] for w in windows), default=None),
    }


def summarize_report(report):
    if report["status"] != "complete":
        raise ValueError("Only complete reports can be summarized.")
    windows = report["windows"]
    expected = {
        (w["sequence"], w["cutoff_ms"], w["reference"]) for w in report["protocol"]["windows"]
    }
    actual = {(w["sequence"], w["cutoff_ms"], w["reference"]) for w in windows}
    if expected != actual or len(actual) != len(windows):
        raise ValueError("Missing, unexpected or duplicate windows.")
    without = {(w["sequence"], w["cutoff_ms"]): w for w in windows if not w["reference"]}
    with_ref = {(w["sequence"], w["cutoff_ms"]): w for w in windows if w["reference"]}
    paired_with, paired_without = [], []
    for key, control in without.items():
        if key not in with_ref:
            raise ValueError("A no-reference window lacks its matching reference window.")
        treatment = with_ref[key]

        def truth(w):
            return [(r["at_ms"], r["truth_visible"], r["truth_box"]) for r in w["rows"]]

        if truth(control) != truth(treatment):
            raise ValueError("Paired windows have different timestamps or annotations.")
        paired_with.append(treatment)
        paired_without.append(control)
    return {
        "scope": report["scope"],
        "model": report["model"],
        "protocol_sha256": report["protocol_sha256"],
        "unique_source_frames": report["unique_source_frames"],
        "full_reference": aggregate(list(with_ref.values())),
        "paired": {
            "with_reference": aggregate(paired_with),
            "without_reference": aggregate(paired_without),
        },
        "by_sequence": {
            seq: aggregate([w for w in with_ref.values() if w["sequence"] == seq])
            for seq in sorted({w["sequence"] for w in windows})
        },
        "failures": [
            {key: w[key] for key in ("sequence", "cutoff_ms", "reference", "error_type")}
            for w in windows
            if w["status"] != "complete"
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    result = summarize_report(json.loads(raw))
    result["report_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for seq, metrics in result["by_sequence"].items():
        print(
            seq,
            metrics["correct_point_in_box"],
            "/",
            metrics["truth_visible"],
            "outside",
            metrics["outside_target_box_when_visible"],
            "failed windows",
            metrics["failed_windows"],
        )


if __name__ == "__main__":
    main()
