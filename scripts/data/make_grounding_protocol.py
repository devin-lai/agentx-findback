"""Build a frozen reference-crop grounding protocol from prepared clips, using labels only.

Window rule (unchanged from the prospective bottle cohort): for each clip, cutoffs at 7 s,
floor(duration / 2) s and floor(last frame timestamp / 1000) s, plus the integer-second cutoff
of at least 7 s whose eight 1 FPS timestamps contain the most externally labelled absent frames
(earliest on ties, and only when that count is positive). Cutoffs are deduplicated. No image,
description or model output is read; the evaluator re-checks every label and video hash.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

WINDOW_FRAMES = 8
FIRST_CUTOFF_MS = 7000


def window_times(cutoff_ms: int) -> list[int]:
    """The eight 1 FPS timestamps a grounded review samples before this cutoff."""
    start = max(0, cutoff_ms - 7000)
    return [t for t in range(start, cutoff_ms + 1, 1000)][:WINDOW_FRAMES]


def cutoffs(labels: dict) -> list[int]:
    fps = labels["assigned_fps"]
    rows = labels["labels"]
    last_ms = rows[-1]["at_ms"]
    duration_s = len(rows) / fps
    chosen = [FIRST_CUTOFF_MS, int(duration_s // 2) * 1000, (last_ms // 1000) * 1000]

    def absent(at_ms: int) -> bool:
        index = round(at_ms * fps / 1000)
        return index < len(rows) and not rows[index]["visible"]

    best, best_count = None, 0
    for cutoff in range(FIRST_CUTOFF_MS, (last_ms // 1000) * 1000 + 1, 1000):
        count = sum(absent(t) for t in window_times(cutoff))
        if count > best_count:
            best, best_count = cutoff, count
    if best is not None:
        chosen.append(best)
    return sorted({c for c in chosen if FIRST_CUTOFF_MS <= c <= last_ms})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--selection-plan", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("A frozen protocol already exists at this path.")
    inputs, windows = {}, []
    for sequence in args.sequences:
        raw = (args.dataset / sequence / "labels.json").read_bytes()
        labels = json.loads(raw)
        chosen = cutoffs(labels)
        inputs[sequence] = {
            "labels_sha256": hashlib.sha256(raw).hexdigest(),
            "video_sha256": labels["video_sha256"],
            "category": labels["category"],
            "cutoffs_ms": chosen,
            "split": "held_out_never_prepared",
        }
        windows += [{"sequence": sequence, "cutoff_ms": c, "reference": True} for c in chosen]
    protocol = {
        "name": args.name,
        "frozen_at": datetime.now(UTC).isoformat(),
        "scope": args.scope,
        "sampling": __doc__.split("\n\n")[1].replace("\n", " "),
        "comparison": "Every model arm runs the identical windows, reference crops and scorer in one session.",
        "inputs": inputs,
        "windows": windows,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if args.selection_plan:
        protocol["selection_plan_sha256"] = hashlib.sha256(
            args.selection_plan.read_bytes()
        ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({"windows": len(windows), "sequences": len(inputs)}))


if __name__ == "__main__":
    main()
