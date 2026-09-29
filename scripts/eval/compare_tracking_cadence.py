"""Validate paired cadence receipts and summarize identical evidence samples."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluate_tracking import summarize


def compare(report: dict) -> dict:
    if report.get("status") != "complete":
        raise ValueError("Cadence experiment is not complete.")
    groups: dict[float, dict[str, dict]] = {}
    for clip in report["clips"]:
        group = groups.setdefault(clip["tracking_fps"], {})
        if clip["sequence"] in group:
            raise ValueError("Duplicate sequence/cadence result.")
        group[clip["sequence"]] = clip
    baseline = groups[min(groups)]
    results = []
    for cadence, clips in sorted(groups.items()):
        if set(clips) != set(baseline):
            raise ValueError("Variants must include the same clips.")
        for sequence, clip in clips.items():
            original = baseline[sequence]
            for field in ("video_sha256", "labels_sha256"):
                if clip[field] != original[field]:
                    raise ValueError("Variants must use the same source and labels.")
            if [r["at_ms"] for r in clip["rows"]] != [r["at_ms"] for r in original["rows"]]:
                raise ValueError("Variants must use identical evidence timestamps.")
            if not clip["rows"]:
                raise ValueError("An empty clip cannot be scored.")
        results.append(
            dict(
                tracking_fps=cadence,
                seconds=sum(c["seconds"] for c in clips.values()),
                inferred_frames=sum(c["inferred_frames"] for c in clips.values()),
                summary=summarize([r for c in clips.values() for r in c["rows"]]),
                clips=[
                    dict(sequence=name, summary=summarize(c["rows"])) for name, c in clips.items()
                ],
            )
        )
    return dict(status="complete", evidence_fps=report["evidence_fps"], variants=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    result = compare(json.loads(raw))
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
