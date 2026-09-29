"""Build video-disjoint, appearance-matched GOT-10k proxy negatives for SFT.

The source GOT-10k annotations establish that the query object's class is visible.
Different canonical source-video IDs reduce same-recording label contamination; they
still do not establish different physical identities. This is private research data.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_video(url: str) -> str | None:
    parsed = urlparse(url.strip())
    host = parsed.hostname.lower() if parsed.hostname else ""
    path = parsed.path.strip("/")
    if not host or not path:
        return None
    if host in {"youtu.be", "www.youtu.be"}:
        return "youtube:" + path.split("/", 1)[0]
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if path == "watch":
            value = parse_qs(parsed.query).get("v", [])
            return "youtube:" + value[0] if value else None
        if path.startswith(("embed/", "shorts/")):
            return "youtube:" + path.split("/", 1)[1].split("/", 1)[0]
    return host + "/" + path


def name_from_target(row: dict) -> str:
    target = row["target"]
    if not target.startswith("Registered "):
        raise ValueError(f"Unexpected target: {target!r}")
    return re.sub(r"\s+", " ", target.removeprefix("Registered ").strip().casefold())


def absent_answer(display_name: str) -> str:
    return json.dumps(
        {
            "present": False,
            "x": None,
            "y": None,
            "note": f"The {display_name} from the reference is not visible in this frame.",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--got-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hard-absent", type=int, default=900)
    parser.add_argument("--min-similarity", type=float, default=0.50)
    parser.add_argument("--min-reference-side", type=int, default=96)
    parser.add_argument("--min-query-box-side", type=int, default=64)
    parser.add_argument("--max-per-class", type=int, default=6)
    parser.add_argument("--exclude-class", action="append", default=["guitar"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source, candidate = args.source.resolve(), args.candidate.resolve()
    got_root, output = args.got_root.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit("Refusing to overwrite an existing filtered set.")
    source_manifest = json.loads((source / "manifest.json").read_text())
    candidate_manifest = json.loads((candidate / "manifest.json").read_text())
    if sha256(source / "samples.jsonl") != source_manifest["samples_sha256"]:
        raise ValueError("Source samples changed.")
    if sha256(candidate / "samples.jsonl") != candidate_manifest["samples_sha256"]:
        raise ValueError("Candidate samples changed.")
    with np.load(args.embeddings, allow_pickle=False) as data:
        embeddings = dict(zip(data["paths"].tolist(), data["vectors"], strict=True))
    rows = [json.loads(s) for s in (source / "samples.jsonl").read_text().splitlines()]
    candidates = [json.loads(s) for s in (candidate / "samples.jsonl").read_text().splitlines()]
    excluded = {name.strip().casefold() for name in args.exclude_class}
    by_class: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    refs: dict[str, str] = {}
    for row in rows:
        if row["kind"] != "same_sequence" or row["present"] is not True:
            continue
        name = name_from_target(row)
        if name in excluded:
            continue
        by_class[name][row["sequence"]].append(row)
        previous = refs.setdefault(row["sequence"], row["reference"])
        if previous != row["reference"]:
            raise ValueError(f"Multiple reference crops for {row['sequence']}")
    from PIL import Image

    # Tiny reference crops can make apparent DINO similarity mostly upsampled blur.
    # A tiny query target makes an absent label too easy to learn from the full frame.
    for name in list(by_class):
        for seq, visible_rows in list(by_class[name].items()):
            with Image.open(source / refs[seq]) as reference_image:
                if min(reference_image.size) < args.min_reference_side:
                    del by_class[name][seq]
                    continue
            boxes = np.loadtxt(got_root / seq / "groundtruth.txt", delimiter=",", ndmin=2)
            kept = [
                row
                for row in visible_rows
                if min(boxes[row["frame"] - 1][2:]) >= args.min_query_box_side
            ]
            if kept:
                by_class[name][seq] = kept
            else:
                del by_class[name][seq]
        if not by_class[name]:
            del by_class[name]
    video_keys: dict[str, str] = {}
    metadata_hash = hashlib.sha256()
    for name, seqs in sorted(by_class.items()):
        for seq in sorted(seqs):
            path = got_root / seq / "meta_info.ini"
            if not path.is_file():
                raise ValueError(f"Missing GOT metadata: {seq}")
            parser_ = configparser.ConfigParser()
            parser_.read_string(path.read_text(errors="replace"))
            meta = parser_["METAINFO"]
            if re.sub(r"\s+", " ", meta.get("object_class", "").strip().casefold()) != name:
                raise ValueError(f"GOT metadata class differs from training row: {seq}")
            key = canonical_video(meta.get("url", ""))
            if key:
                video_keys[seq] = key
            metadata_hash.update(f"{seq}\t{sha256(path)}\n".encode())
    # Keep only visually similar, distinct-source-video *ordered* sequence pairs.
    edges: dict[str, list[tuple[float, str, str]]] = {}
    for name, seqs in by_class.items():
        ordered = []
        for src_seq in seqs:
            if src_seq not in video_keys:
                continue
            for query_seq in seqs:
                if query_seq == src_seq or query_seq not in video_keys:
                    continue
                if video_keys[src_seq] == video_keys[query_seq]:
                    continue
                src_ref, query_ref = refs[src_seq], refs[query_seq]
                if src_ref == query_ref:
                    continue
                score = float(embeddings[src_ref] @ embeddings[query_ref])
                if args.min_similarity <= score < 0.9999:
                    ordered.append((score, src_seq, query_seq))
        ordered.sort(key=lambda e: (-e[0], e[1], e[2]))
        if ordered:
            edges[name] = ordered
    capacity = sum(min(args.max_per_class, len(items)) for items in edges.values())
    if capacity < args.hard_absent:
        raise ValueError(f"Only {capacity} eligible class-capped ordered pairs for {args.hard_absent} requested.")

    rng = random.Random(args.seed)
    classes = sorted(edges)
    rng.shuffle(classes)
    selected = []
    class_counts = Counter()
    while len(selected) < args.hard_absent:
        advanced = False
        for name in classes:
            if len(selected) == args.hard_absent:
                break
            if class_counts[name] >= args.max_per_class or not edges[name]:
                continue
            score, src_seq, query_seq = edges[name].pop(0)
            source_row = by_class[name][src_seq][0]
            query_row = rng.choice(by_class[name][query_seq])
            display_name = source_row["target"].removeprefix("Registered ")
            selected.append(
                {
                    "sequence": f"{src_seq}->{query_seq}",
                    "source_sequence": src_seq,
                    "query_sequence": query_seq,
                    "source_video_sha256": hashlib.sha256(video_keys[src_seq].encode()).hexdigest(),
                    "query_video_sha256": hashlib.sha256(video_keys[query_seq].encode()).hexdigest(),
                    "frame": query_row["frame"],
                    "kind": "same_class_cross_video_appearance",
                    "class_name": name,
                    "target": source_row["target"],
                    "reference": refs[src_seq],
                    "image": query_row["image"],
                    "query_reference": refs[query_seq],
                    "appearance_similarity": round(score, 6),
                    "answer": absent_answer(display_name),
                    "present": False,
                }
            )
            class_counts[name] += 1
            advanced = True
        if not advanced:
            raise ValueError("Candidate pool exhausted before reaching hard-negative target.")
    rng.shuffle(selected)
    hard_iter = iter(selected)
    final = [
        next(hard_iter) if row["kind"] == "same_class_cross_sequence" else row
        for row in candidates
    ]
    if len(final) != 3000 or next(hard_iter, None) is not None:
        raise AssertionError("Unexpected training set size or hard-negative slot count.")
    if any(r["source_video_sha256"] == r["query_video_sha256"] for r in selected):
        raise AssertionError("A source-video collision survived selection.")
    output.mkdir(parents=True)
    (output / "images").symlink_to(os.path.relpath(source / "images", output), target_is_directory=True)
    with (output / "samples.jsonl").open("w") as stream:
        for row in final:
            stream.write(json.dumps(row) + "\n")
    scores = np.asarray([r["appearance_similarity"] for r in selected], dtype=np.float64)
    manifest = {
        "schema_version": 1,
        "purpose": "GOT-10k research SFT with same-class, distinct-source-video, appearance-matched proxy negatives",
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "candidate_manifest_sha256": sha256(candidate / "manifest.json"),
        "embeddings_sha256": sha256(args.embeddings),
        "metadata_path_sha256_digest": metadata_hash.hexdigest(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "stats": {
            "rows": len(final),
            "hard_negatives": len(selected),
            "hard_negative_classes": len(class_counts),
            "max_per_class": max(class_counts.values()),
            "candidate_capacity_after_class_cap": capacity,
            "same_upstream_video_pairs": 0,
            "min_similarity": round(float(scores.min()), 6),
            "median_similarity": round(float(np.median(scores)), 6),
            "p90_similarity": round(float(np.percentile(scores, 90)), 6),
        },
        "script_sha256": sha256(Path(__file__)),
        "samples_sha256": sha256(output / "samples.jsonl"),
        "limits": [
            "Different upstream source videos and same named class are still proxy negative labels, not verified physical identity.",
            "DINOv2 similarity is a training-data filter, not a calibrated deployed identity score.",
            "Research media and derived rows are excluded from source releases under the GOT-10k license.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["stats"]), flush=True)


if __name__ == "__main__":
    main()
