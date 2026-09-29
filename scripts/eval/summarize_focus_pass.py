"""Publishable summary of the Cosmos focus-pass comparison.

Keeps counts, denominators and receipts; keeps no research frames, prompts or model replies.
"""

import argparse
import hashlib
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path


def arm_summary(report: dict) -> dict:
    errors, shifts, statuses = [], [], {"confirmed": 0, "unconfirmed": 0, "unavailable": 0}
    failed = [
        {"sequence": w["sequence"], "cutoff_ms": w["cutoff_ms"], "error": w.get("error_type")}
        for w in report["windows"]
        if w.get("status") != "complete"
    ]
    for window in report["windows"]:
        if not window.get("reference"):
            continue
        for row in window.get("rows", []):
            if row["truth_visible"] and row["predicted_visible"] is True:
                if row.get("center_error_box_diagonals") is not None:
                    errors.append(row["center_error_box_diagonals"])
        for frame in (window.get("result") or {}).get("frame_reports", []):
            focus = frame.get("focus")
            if not focus:
                continue
            statuses[focus["status"]] = statuses.get(focus["status"], 0) + 1
            if focus.get("x") is not None and focus.get("coarse_x") is not None:
                shifts.append(
                    ((focus["x"] - focus["coarse_x"]) ** 2 + (focus["y"] - focus["coarse_y"]) ** 2)
                    ** 0.5
                )
    return {
        "focus_pass": report.get("focus_pass"),
        "focus_window": report.get("focus_window"),
        "summary": report["summary"],
        "failed_windows": failed,
        "model_calls": sum(len(w.get("calls", [])) for w in report["windows"]),
        "total_inference_seconds": round(
            sum(w.get("elapsed_seconds", 0) for w in report["windows"]), 1
        ),
        "median_window_seconds": round(
            statistics.median([w.get("elapsed_seconds", 0) for w in report["windows"]]), 2
        ),
        "centre_error_box_diagonals": {
            "samples": len(errors),
            "median": round(statistics.median(errors), 4) if errors else None,
            "mean": round(statistics.fmean(errors), 4) if errors else None,
            "within_half_diagonal": round(sum(e <= 0.5 for e in errors) / len(errors), 4)
            if errors
            else None,
        },
        "focus_requests": statuses,
        "point_shift_frame_units": {
            "samples": len(shifts),
            "median": round(statistics.median(shifts), 5) if shifts else None,
            "mean": round(statistics.fmean(shifts), 5) if shifts else None,
            "above_0_02": round(sum(s > 0.02 for s in shifts) / len(shifts), 4) if shifts else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="LABEL=PATH for each complete arm report")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms, receipts, protocols = {}, {}, set()
    for entry in args.reports:
        label, _, path = entry.partition("=")
        raw = Path(path).read_bytes()
        report = json.loads(raw)
        if report.get("status") != "complete":
            raise SystemExit(f"{label}: only complete reports can be summarized.")
        protocols.add(report["protocol_sha256"])
        arms[label] = arm_summary(report)
        receipts[f"{label}.json"] = hashlib.sha256(raw).hexdigest()
    if len(protocols) != 1:
        raise SystemExit("Arms must share one frozen protocol; refusing an unpaired summary.")
    first = json.loads(Path(args.reports[0].partition("=")[2]).read_bytes())
    args.output.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "scope": (
                    "Paired three-arm Cosmos focus-pass comparison on the frozen "
                    "grounding protocol. Point-in-box on correlated public research frames; not "
                    "IoU, physical identity, general accuracy or a user study."
                ),
                "summarized_at": datetime.now(UTC).isoformat(),
                "protocol_sha256": protocols.pop(),
                "model": first["model"],
                "max_edge": first["max_edge"],
                "platform": first["platform"],
                "arms": arms,
                "report_sha256": receipts,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps({label: a["summary"]["with_reference"] for label, a in arms.items()}, indent=2)
    )


if __name__ == "__main__":
    main()
