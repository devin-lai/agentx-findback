"""Render the camera-evidence comparison as a before/after dumbbell, from retained receipts.

One row per clip, one dot per arm. Clips the change does not touch show a single dot, which is
the point: the gain comes from four clips whose camera actually moves, and nothing else regresses.
"""

import argparse
import hashlib
import json
from pathlib import Path

# One hue, two shades: an ordinal ramp validated against this surface (light end 2.25:1).
BEFORE, AFTER = "#63b8a0", "#0f6e56"
SURFACE, INK, MUTED, GRID = "#f8faf8", "#153e32", "#50635b", "#d5e0da"


def load(directory: Path, cohorts: list[str]) -> tuple[list[dict], dict]:
    clips, totals = [], {"legacy": [], "compensated": []}
    for cohort in cohorts:
        comparison = json.loads((directory / f"compare-{cohort}.json").read_text())
        if comparison.get("status") != "complete":
            raise SystemExit(f"{cohort}: comparison is not complete.")
        for clip in comparison["clips"]:
            clips.append(
                {
                    "sequence": clip["sequence"],
                    "cohort": cohort,
                    "visible": clip["baseline"]["truth_visible"],
                    "legacy": clip["baseline"]["correct_at_iou_0_5"],
                    "candidate": clip["candidate"]["correct_at_iou_0_5"],
                    "lost": clip["lost_correct"],
                }
            )
        for arm, key in (("legacy", "baseline"), ("compensated", "candidate")):
            totals[arm].append(comparison[key])
    return clips, totals


def aggregate(summaries: list[dict]) -> dict:
    keys = ("correct_at_iou_0_5", "truth_visible", "truth_absent", "false_visible_when_absent")
    total = {k: sum(s[k] for s in summaries) for k in keys}
    total["recall"] = total["correct_at_iou_0_5"] / total["truth_visible"]
    return total


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--cohorts", nargs="+", default=["inspected", "holdout", "absence", "confirmation"]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clips, totals = load(args.directory, args.cohorts)
    before, after = aggregate(totals["legacy"]), aggregate(totals["compensated"])
    # Largest change first; the reader should meet the four clips that moved immediately.
    clips.sort(
        key=lambda c: (c["legacy"] - c["candidate"], -c["candidate"] / c["visible"]),
    )

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
    fig = plt.figure(figsize=(12, 8.2), facecolor=SURFACE)
    ax = fig.add_axes([0.11, 0.22, 0.60, 0.60], facecolor=SURFACE)
    for index, clip in enumerate(clips):
        low = 100 * clip["legacy"] / clip["visible"]
        high = 100 * clip["candidate"] / clip["visible"]
        moved = clip["candidate"] != clip["legacy"]
        if moved:
            ax.plot([low, high], [index, index], color=AFTER, linewidth=2, zorder=2)
            ax.scatter(
                [low], [index], s=90, color=BEFORE, zorder=3, edgecolors=SURFACE, linewidths=2
            )
        ax.scatter([high], [index], s=90, color=AFTER, zorder=4, edgecolors=SURFACE, linewidths=2)
        # Direct labels only where something changed; a number on every row is noise.
        if moved:
            ax.text(
                high + 2.4,
                index,
                f"{clip['legacy']} → {clip['candidate']} of {clip['visible']}",
                ha="left",
                va="center",
                fontsize=9,
                color=INK,
            )
    ax.set_yticks(range(len(clips)), [c["sequence"] for c in clips], fontsize=9)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.invert_yaxis()
    ax.set_xlim(-3, 132)
    ax.set_ylim(len(clips) - 0.4, -0.6)
    ax.set_xticks([0, 25, 50, 75, 100], [f"{v}%" for v in (0, 25, 50, 75, 100)], color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("Visible-target recall at IoU ≥ 0.5", labelpad=12, color=MUTED)

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=9,
            markerfacecolor=colour,
            markeredgecolor=SURFACE,
            label=label,
        )
        for colour, label in (
            (BEFORE, "Released guard — abstain for the rest of the recording"),
            (AFTER, "Camera-pose tracking — keep observing"),
        )
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(-0.005, 1.015),
        frameon=False,
        ncol=1,
        labelcolor=INK,
        handletextpad=0.4,
    )
    fig.text(
        0.055,
        0.945,
        "A knocked camera no longer erases the recording",
        fontsize=21,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.905,
        f"Both arms in one session · {len(clips)} public LaSOT clips · identical sources, "
        "labels, checkpoints and sample times",
        fontsize=10.5,
        color=MUTED,
    )

    panel = fig.add_axes([0.755, 0.22, 0.22, 0.60])
    panel.axis("off")
    panel.text(0, 1.0, "AGGREGATE", fontsize=9.5, color=MUTED)
    for i, (label, data, colour) in enumerate(
        (("Released guard", before, BEFORE), ("Camera-pose tracking", after, AFTER))
    ):
        y = 0.90 - i * 0.40
        panel.text(0, y, f"{100 * data['recall']:.1f}%", fontsize=30, weight="bold", color=colour)
        panel.text(0, y - 0.085, label, fontsize=10, color=INK)
        panel.text(
            0,
            y - 0.15,
            f"{data['correct_at_iou_0_5']:,} of {data['truth_visible']:,} visible samples",
            fontsize=9.5,
            color=MUTED,
        )
        panel.text(
            0,
            y - 0.21,
            f"{data['false_visible_when_absent']} of {data['truth_absent']} absent samples "
            "called visible",
            fontsize=9.5,
            color=MUTED,
        )
    gained = sum(max(0, c["candidate"] - c["legacy"]) for c in clips)
    panel.text(
        0,
        0.07,
        f"+{gained} newly correct\n{sum(c['lost'] for c in clips)} lost",
        fontsize=10.5,
        color=INK,
        linespacing=1.5,
    )

    unchanged = sum(c["candidate"] == c["legacy"] for c in clips)
    for i, note in enumerate(
        (
            "Every abstention stays in the recall denominator. "
            f"{unchanged} of the {len(clips)} clips are identical in both arms.",
            "Correlated frames from public research footage: not original desk recordings, a user "
            "trial, or proof of physical identity.",
            "Region naming is not scored here: the annotation names the third of the current "
            "frame, which is exactly what a recovered region disagrees with.",
        )
    ):
        fig.text(0.055, 0.115 - i * 0.032, note, fontsize=9, color=MUTED)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, facecolor=SURFACE)
    fig.savefig(args.output.with_suffix(".svg"), facecolor=SURFACE)
    receipt = {
        "inputs": {
            f"compare-{c}.json": hashlib.sha256(
                (args.directory / f"compare-{c}.json").read_bytes()
            ).hexdigest()
            for c in args.cohorts
        },
        "clips": len(clips),
        "legacy": before,
        "compensated": after,
        "palette": {"before": BEFORE, "after": AFTER, "surface": SURFACE},
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"figure": str(args.output), "clips": len(clips)}))


if __name__ == "__main__":
    main()
