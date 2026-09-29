"""Freeze same-category, different-recording negatives from an existing grounding protocol.

Each source keeps its frozen cutoffs and uses the next source's registration crop, wrapping at
the end. Only sequence identities and protocol bytes are read; no image, annotation or model
output is inspected. Run before inference on these source/reference pairs.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-protocol", required=True, type=Path)
    parser.add_argument("--selection-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("A frozen cross-reference protocol already exists at this path.")
    raw = args.source_protocol.read_bytes()
    source = json.loads(raw)
    sequences = list(source["inputs"])
    if len(sequences) < 2 or len(set(sequences)) != len(sequences):
        raise ValueError("Use at least two distinct source sequences.")
    by_category: dict[str, list[str]] = {}
    for name in sequences:
        by_category.setdefault(source["inputs"][name]["category"], []).append(name)
    if any(len(names) < 2 for names in by_category.values()):
        raise ValueError("Every category needs at least two recordings for identity negatives.")
    by_source = {
        name: names[(index + 1) % len(names)]
        for names in by_category.values()
        for index, name in enumerate(names)
    }
    windows = []
    for case in source["windows"]:
        if not case["reference"] or "reference_sequence" in case:
            raise ValueError("The source protocol must contain ordinary reference-crop windows.")
        sequence = case["sequence"]
        if sequence not in by_source:
            raise ValueError(f"Unknown source sequence: {sequence}")
        windows.append(
            {
                **case,
                "reference_sequence": by_source[sequence],
                "truth": "different_recording_absent",
            }
        )
    result = {
        "name": "Cross-recording identity negatives for " + ", ".join(by_category),
        "frozen_at": datetime.now(UTC).isoformat(),
        "scope": "Different LaSOT recordings of the same category are used as a proxy for different physical targets. Source hashes prove different files, not distinct object identity; inspect scenes. This tests identity mismatch, not same-scene disappearance.",
        "comparison": source["comparison"],
        "truth": "different_recording_absent",
        "source_protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "selection_plan_sha256": hashlib.sha256(args.selection_plan.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": source["inputs"],
        "windows": windows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"sequences": len(sequences), "windows": len(windows)}))


if __name__ == "__main__":
    main()
