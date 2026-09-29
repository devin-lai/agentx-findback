"""Compare complete grounding runs on identical source frames and annotations."""

import argparse
import hashlib
import json
from pathlib import Path

from summarize_grounded_reviews import aggregate, summarize_report


def compare_reports(reports, *, allow_subset=False):
    if len(reports) < 2:
        raise ValueError("At least two reports are required.")
    inventories = {}
    for name, report in reports.items():
        summarize_report(report)  # Reject partial, duplicate and unpaired reports.
        inventories[name] = {
            (w["sequence"], w["cutoff_ms"], w["reference"]): w for w in report["windows"]
        }
    common = set.intersection(*(set(windows) for windows in inventories.values()))
    if not common:
        raise ValueError("Reports have no matching windows.")
    if not allow_subset and any(set(windows) != common for windows in inventories.values()):
        raise ValueError("Different windows: explicitly enable subset comparison.")
    for key in sorted(common):
        source = truth = None
        for name, windows in inventories.items():
            provenance = reports[name]["protocol"]["inputs"][key[0]]
            candidate_source = (provenance["video_sha256"], provenance["labels_sha256"])
            candidate_truth = [
                (r["at_ms"], r["truth_visible"], r["truth_box"]) for r in windows[key]["rows"]
            ]
            if source is not None and source != candidate_source:
                raise ValueError(f"Source/label hashes differ for {key}.")
            if truth is not None and truth != candidate_truth:
                raise ValueError(f"Frame timestamps or truth differ for {key}.")
            source, truth = candidate_source, candidate_truth
    return {
        "scope": "Matched-frame comparison; inspected development data, not held-out accuracy.",
        "subset_explicitly_allowed": allow_subset,
        "matched_windows": [list(key) for key in sorted(common)],
        "unique_source_frames": len(
            {
                (key[0], row["at_ms"])
                for key in common
                for row in next(iter(inventories.values()))[key]["rows"]
            }
        ),
        "variants": {
            name: {
                "metrics": aggregate([windows[key] for key in sorted(common)]),
                "excluded_windows": [list(key) for key in sorted(set(windows) - common)],
            }
            for name, windows in inventories.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="LABEL=PATH for each complete report")
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing comparison.")
    reports, receipts = {}, {}
    for entry in args.reports:
        name, separator, raw_path = entry.partition("=")
        if not separator or not name or name in reports:
            parser.error("Each report needs a distinct nonempty LABEL=PATH.")
        path = Path(raw_path)
        raw = path.read_bytes()
        reports[name] = json.loads(raw)
        receipts[name] = {"report": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    result = compare_reports(reports, allow_subset=args.allow_subset)
    result["inputs"] = receipts
    result["comparator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for name, variant in result["variants"].items():
        metrics = variant["metrics"]
        print(name, json.dumps(metrics))


if __name__ == "__main__":
    main()
