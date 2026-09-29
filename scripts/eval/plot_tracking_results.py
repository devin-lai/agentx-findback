"""Render a paired tracking comparison directly from retained evaluation receipts."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--labels", nargs=2, default=["RT-DETR + DINOv2", "SAM2 + presence guard"])
    parser.add_argument("--title", default="From detector boxes to temporal object memory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(p.read_text()) for p in (args.baseline, args.candidate)]
    for report in reports:
        if report["status"] != "complete" or report.get("failed_clips", 0):
            raise ValueError(
                "Only completed reports without omitted failures can make this comparison."
            )
    keyed = [{c["sequence"]: c for c in r["clips"]} for r in reports]
    if keyed[0].keys() != keyed[1].keys():
        raise ValueError("Clip sets differ.")
    sequences = list(keyed[0])
    for sequence in sequences:
        a, b = keyed[0][sequence], keyed[1][sequence]
        if any(a[key] != b[key] for key in ("video_sha256", "labels_sha256")):
            raise ValueError("Source or labels differ.")
        if [(r["at_ms"], r["truth_visible"], r["truth_box"]) for r in a["rows"]] != [
            (r["at_ms"], r["truth_visible"], r["truth_box"]) for r in b["rows"]
        ]:
            raise ValueError("Sampling or truth differs.")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(11, 7), facecolor="#f8faf8")
    ax = fig.add_axes([0.15, 0.28, 0.58, 0.50], facecolor="#f8faf8")
    colors = ["#6f8495", "#16856c"]
    ys = np.arange(len(sequences))
    width = 0.32
    for i, (data, label, color) in enumerate(zip(keyed, args.labels, colors, strict=True)):
        values = [100 * data[s]["summary"]["recall_at_iou_0_5"] for s in sequences]
        ypos = ys + (i - 0.5) * width
        ax.barh(ypos, values, height=width, color=color, label=label)
        for y, sequence, value in zip(ypos, sequences, values, strict=True):
            summary = data[sequence]["summary"]
            ax.text(
                value + 1,
                y,
                f"{summary['correct_at_iou_0_5']}/{summary['truth_visible']}  ({value:.1f}%)",
                ha="left",
                va="center",
                fontsize=9,
            )
    ax.set_yticks(ys, sequences)
    ax.invert_yaxis()
    ax.set_xlim(0, 130)
    ax.set_xticks([0, 25, 50, 75, 100], [f"{v}%" for v in [0, 25, 50, 75, 100]])
    ax.grid(axis="x", alpha=0.16)
    ax.set_axisbelow(True)
    ax.set_xlabel("Visible-target recall at IoU ≥ 0.5 (higher is better)", labelpad=12)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.01), frameon=False, ncol=2)
    fig.text(
        0.07,
        0.94,
        args.title,
        fontsize=20,
        weight="bold",
        color="#153e32",
    )
    fig.text(
        0.07,
        0.895,
        f"Paired evaluation · {len(sequences)} public LaSOT clips · Same labels, registration and sampling",
        fontsize=11,
        color="#50635b",
    )
    stats = fig.add_axes([0.8, 0.28, 0.18, 0.56])
    stats.axis("off")
    stats.text(0, 1, "AGGREGATE", fontsize=10, color="#50635b")
    for i, report in enumerate(reports):
        summary = report["summary"]
        y = 0.84 - i * 0.42
        stats.text(
            0,
            y,
            f"{100 * summary['recall_at_iou_0_5']:.1f}%",
            fontsize=27,
            weight="bold",
            color=colors[i],
        )
        stats.text(0, y - 0.09, "visible-target recall", fontsize=10)
        stats.text(
            0, y - 0.18, f"Precision: {100 * summary['precision_at_iou_0_5']:.1f}%", fontsize=10
        )
        stats.text(
            0,
            y - 0.26,
            f"False visible: {summary['false_visible_when_absent']}/{summary['truth_absent']} absent"
            if summary["truth_absent"]
            else "No absent-target samples",
            fontsize=10,
        )
    fig.text(
        0.07,
        0.14,
        "Camera-change abstentions and missed detections remain in the denominator.",
        fontsize=10,
        color="#50635b",
    )
    fig.text(
        0.07,
        0.10,
        f"{reports[0]['summary']['truth_absent']} labeled-absent samples. Frames within each clip are correlated."
        if reports[0]["summary"]["truth_absent"]
        else "No absent-target samples: this comparison cannot validate disappearance behavior. Frames are correlated.",
        fontsize=10,
        color="#50635b",
    )
    fig.text(
        0.07,
        0.06,
        "Research stress test; no new desk-recording or user-trial claim. Pretraining overlap has not been audited.",
        fontsize=9,
        color="#50635b",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["svg", "png"]:
        fig.savefig(args.output.with_suffix("." + ext), dpi=180, facecolor=fig.get_facecolor())
    receipt = dict(
        inputs={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.baseline, args.candidate)
        },
        summaries={
            label: report["summary"] for label, report in zip(args.labels, reports, strict=True)
        },
        scope="Descriptive paired public-video comparison; no confidence intervals or independence assumption.",
    )
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
