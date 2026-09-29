"""Summarize grounding reports from several VLM arms on identical frozen windows.

Recomputes every count from per-frame rows (never from report summaries), checks that all arms
cover the same windows, frames and labels, and reports paired gains/losses against a control arm.

`--consensus A B` adds an offline two-model panel: a frame counts as a positive claim only when
both models say the target is present and their points lie within `--agree` (normalized
Euclidean distance); the claim uses model A's point. Either model saying "absent" or the two
points disagreeing makes the frame "not claimed". This needs no new inference: it replays the
saved per-frame predictions.

`--presence-veto A B` accepts A's point only when B also says the target is present; it does
not require the points to agree. This measures a base model's presence decision paired with an
adapter's localization, without silently treating an arbitrary distance as agreement.
"""

import argparse
import json
import math
import statistics
from pathlib import Path


def rows_by_key(report: dict) -> dict:
    out = {}
    for window in report["windows"]:
        if not window["reference"]:
            continue
        for row in window["rows"]:
            out[(window["sequence"], window["cutoff_ms"], row["at_ms"])] = row
    return out


def score(rows: list[dict]) -> dict:
    visible = sum(r["truth_visible"] for r in rows)
    positive = sum(r["predicted_visible"] is True for r in rows)
    correct = sum(r["point_inside_target_box"] for r in rows)
    absent = len(rows) - visible
    false_visible = sum(r["predicted_visible"] is True and not r["truth_visible"] for r in rows)
    return {
        "requests": len(rows),
        "visible": visible,
        "absent": absent,
        "available": sum(r["available"] for r in rows),
        "positive_claims": positive,
        "correct_point_in_box": correct,
        "recall": round(correct / visible, 4) if visible else None,
        "precision": round(correct / positive, 4) if positive else None,
        "false_visible": false_visible,
        "wrong_point_on_visible": sum(
            r["predicted_visible"] is True
            and r["truth_visible"]
            and not r["point_inside_target_box"]
            for r in rows
        ),
        "missed_visible": sum(
            r["truth_visible"] and r["predicted_visible"] is not True for r in rows
        ),
    }


def window_seconds(report: dict) -> dict:
    values = [w["elapsed_seconds"] for w in report["windows"] if w["reference"]]
    return {
        "median_window_seconds": round(statistics.median(values), 3),
        "max_window_seconds": round(max(values), 3),
        "failed_windows": sum(
            w["status"] != "complete" for w in report["windows"] if w["reference"]
        ),
    }


def consensus(a: dict, b: dict, agree: float) -> list[dict]:
    rows = []
    for key, ra in a.items():
        rb = b[key]
        both = ra["predicted_visible"] is True and rb["predicted_visible"] is True
        close = (
            both
            and None not in (ra["x"], ra["y"], rb["x"], rb["y"])
            and math.hypot(ra["x"] - rb["x"], ra["y"] - rb["y"]) <= agree
        )
        rows.append(
            {
                **ra,
                "predicted_visible": True if close else False,
                "point_inside_target_box": bool(close and ra["point_inside_target_box"]),
                "available": ra["available"] and rb["available"],
            }
        )
    return rows


def presence_veto(point_model: dict, presence_model: dict) -> list[dict]:
    rows = []
    for key, point in point_model.items():
        veto = presence_model[key]
        claimed = point["predicted_visible"] is True and veto["predicted_visible"] is True
        rows.append(
            {
                **point,
                "predicted_visible": claimed,
                "point_inside_target_box": bool(claimed and point["point_inside_target_box"]),
                "available": point["available"] and veto["available"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="label=path/to/report.json")
    parser.add_argument("--control", required=True)
    parser.add_argument("--consensus", nargs=2, action="append", default=[])
    parser.add_argument("--presence-veto", nargs=2, action="append", default=[])
    parser.add_argument("--agree", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = {}
    for spec in args.arm:
        label, path = spec.split("=", 1)
        reports[label] = json.loads(Path(path).read_text())
    keyed = {label: rows_by_key(report) for label, report in reports.items()}
    base_keys = set(keyed[args.control])
    for label, rows in keyed.items():
        if set(rows) != base_keys:
            raise SystemExit(f"{label} does not cover the control arm's frames.")
        for key, row in rows.items():
            truth = keyed[args.control][key]
            if (row["truth_visible"], row["truth_box"]) != (
                truth["truth_visible"],
                truth["truth_box"],
            ):
                raise SystemExit(f"{label} has different labels at {key}.")
    summary = {
        "frames": len(base_keys),
        "arms": {},
        "paired_vs_control": {},
        "consensus": {},
        "presence_veto": {},
    }
    for label, rows in keyed.items():
        summary["arms"][label] = {
            "model": reports[label].get("model"),
            **score(list(rows.values())),
            **window_seconds(reports[label]),
            "per_sequence": {
                seq: score([r for k, r in rows.items() if k[0] == seq])["correct_point_in_box"]
                for seq in sorted({k[0] for k in rows})
            },
        }
        if label != args.control:
            control = keyed[args.control]
            summary["paired_vs_control"][label] = {
                "gained": sum(
                    rows[k]["point_inside_target_box"] and not control[k]["point_inside_target_box"]
                    for k in base_keys
                ),
                "lost": sum(
                    control[k]["point_inside_target_box"] and not rows[k]["point_inside_target_box"]
                    for k in base_keys
                ),
            }
    for a, b in args.consensus:
        panel = consensus(keyed[a], keyed[b], args.agree)
        summary["consensus"][f"{a}+{b}"] = {"agree_distance": args.agree, **score(panel)}
    for a, b in args.presence_veto:
        panel = presence_veto(keyed[a], keyed[b])
        summary["presence_veto"][f"{a}_point+{b}_presence"] = score(panel)
    text = json.dumps(summary, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
