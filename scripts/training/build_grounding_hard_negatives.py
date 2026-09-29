"""Rebalance an existing GOT-10k SFT set with same-class cross-recording negatives.

The input must be an immutable output of build_grounding_sft.py. Existing positive,
annotated-absence and crop-negative rows retain their exact production-encoded images.
For each new hard negative, the reference is from one GOT sequence and the full
query frame contains a labeled visible object of the *same named class* in another
sequence. Separate recordings are only a proxy for different physical identities;
the resulting labels can contain errors and must be judged on separate footage.

The output links to the input image store and is research-only: GOT-10k's
CC BY-NC-SA 4.0 media and these derived images must not be released with source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def class_name(row: dict) -> str:
    target = row.get("target", "")
    if not target.startswith("Registered "):
        raise ValueError(f"Unexpected target phrase: {target!r}")
    return re.sub(r"\s+", " ", target.removeprefix("Registered ").strip().casefold())


def negative_answer(name: str) -> str:
    return json.dumps(
        {
            "present": False,
            "x": None,
            "y": None,
            "note": f"The {name} from the reference is not visible in this frame.",
        }
    )


def choose_rows(
    source_rows: list[dict],
    *,
    positive: int,
    annotated_absent: int,
    crop_absent: int,
    hard_absent: int,
    exclude_classes: set[str],
    seed: int,
) -> tuple[list[dict], dict]:
    """Return shuffled rows plus counts, refusing undersized or malformed source pools."""
    rng = random.Random(seed)
    pools: dict[str, list[dict]] = defaultdict(list)
    by_class: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in source_rows:
        name = class_name(row)
        if name in exclude_classes:
            continue
        kind, present = row.get("kind"), row.get("present")
        if kind == "same_sequence" and present is True:
            pools["positive"].append(row)
            by_class[name][row["sequence"]].append(row)
        elif kind == "same_sequence" and present is False:
            pools["annotated_absent"].append(row)
        elif kind == "crop_negative" and present is False:
            pools["crop_absent"].append(row)
    needs = {
        "positive": positive,
        "annotated_absent": annotated_absent,
        "crop_absent": crop_absent,
    }
    for kind, count in needs.items():
        if count < 0 or len(pools[kind]) < count:
            raise ValueError(f"Need {count} {kind} rows; found {len(pools[kind])}.")
    if hard_absent < 0:
        raise ValueError("hard_absent must be nonnegative")
    selected = [row.copy() for kind, count in needs.items() for row in rng.sample(pools[kind], count)]

    eligible = {name: seqs for name, seqs in by_class.items() if len(seqs) >= 2}
    if hard_absent and not eligible:
        raise ValueError("No named class has visible targets in two sequences.")
    names = sorted(eligible)
    rng.shuffle(names)
    used: set[tuple[str, str]] = set()
    hard_classes = Counter()
    hard_rows = []
    attempts = 0
    max_attempts = max(1000, hard_absent * 100)
    while len(hard_rows) < hard_absent and attempts < max_attempts:
        name = names[attempts % len(names)]
        attempts += 1
        seqs = eligible[name]
        src_seq, query_seq = rng.sample(sorted(seqs), 2)
        source = rng.choice(seqs[src_seq])
        query = rng.choice(seqs[query_seq])
        pair = (source["reference"], query["image"])
        if pair in used:
            continue
        used.add(pair)
        display_name = source["target"].removeprefix("Registered ")
        hard_rows.append(
            {
                "sequence": f"{src_seq}->{query_seq}",
                "source_sequence": src_seq,
                "query_sequence": query_seq,
                "frame": query.get("frame"),
                "kind": "same_class_cross_sequence",
                "class_name": name,
                "target": source["target"],
                "reference": source["reference"],
                "image": query["image"],
                "answer": negative_answer(display_name),
                "present": False,
            }
        )
        hard_classes[name] += 1
    if len(hard_rows) != hard_absent:
        raise ValueError(f"Only {len(hard_rows)} unique hard pairs for {hard_absent} requested.")
    selected.extend(hard_rows)
    rng.shuffle(selected)
    stats = {
        "counts": needs | {"same_class_cross_sequence": hard_absent},
        "total": len(selected),
        "hard_negative_classes": len(hard_classes),
        "hard_negative_top_classes": hard_classes.most_common(20),
        "source_pool_counts": {kind: len(rows) for kind, rows in pools.items()},
    }
    assert len(selected) == sum(needs.values()) + hard_absent
    return selected, stats


def validate_images(rows: list[dict], source: Path) -> dict:
    paths = sorted({row[key] for row in rows for key in ("reference", "image")})
    digest = hashlib.sha256()
    for rel in paths:
        path = (source / rel).resolve()
        if not path.is_relative_to((source / "images").resolve()) or not path.is_file():
            raise ValueError(f"Missing or escaped image: {rel}")
        actual = sha256(path)
        if path.stem != actual:
            raise ValueError(f"Image digest mismatch: {rel}")
        digest.update(f"{rel}\t{actual}\n".encode())
    return {"count": len(paths), "path_and_sha256_digest": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--positive", type=int, default=1500)
    parser.add_argument("--annotated-absent", type=int, default=200)
    parser.add_argument("--crop-absent", type=int, default=400)
    parser.add_argument("--hard-absent", type=int, default=900)
    parser.add_argument("--exclude-class", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit("Refusing to overwrite an existing sample set.")
    manifest_path, samples_path = source / "manifest.json", source / "samples.jsonl"
    if not manifest_path.is_file() or not samples_path.is_file() or not (source / "images").is_dir():
        raise SystemExit("The source must contain manifest.json, samples.jsonl and images/.")
    source_manifest = json.loads(manifest_path.read_text())
    if source_manifest["samples_sha256"] != sha256(samples_path):
        raise SystemExit("Source sample file does not match its frozen manifest.")
    source_rows = [json.loads(line) for line in samples_path.read_text().splitlines()]
    excluded = {s.strip().casefold() for s in args.exclude_class}
    rows, stats = choose_rows(
        source_rows,
        positive=args.positive,
        annotated_absent=args.annotated_absent,
        crop_absent=args.crop_absent,
        hard_absent=args.hard_absent,
        exclude_classes=excluded,
        seed=args.seed,
    )
    image_validation = validate_images(rows, source)
    output.mkdir(parents=True)
    (output / "images").symlink_to(os.path.relpath(source / "images", output), target_is_directory=True)
    with (output / "samples.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    manifest = {
        "schema_version": 1,
        "purpose": "GOT-10k research SFT with same named-class cross-recording proxy negatives",
        "source_manifest_sha256": sha256(manifest_path),
        "source_samples_sha256": sha256(samples_path),
        "source_image_validation": image_validation,
        "source_path": str(source),
        "source_license": source_manifest.get("source"),
        "args": {k: str(v) for k, v in vars(args).items()},
        "excluded_classes": sorted(excluded),
        "stats": stats,
        "script_sha256": sha256(Path(__file__)),
        "samples_sha256": sha256(output / "samples.jsonl"),
        "limitations": [
            "Different GOT-10k recordings are not verified distinct physical identities.",
            "The hard negatives are a training proxy and can contain mislabeled identities.",
            "The output images link to a research dataset and must not be published with source.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
