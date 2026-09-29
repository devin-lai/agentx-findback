"""Replace same-class proxy negatives with visually closest different-sequence pairs.

This is a training-data selector, not an identity verifier. DINOv2 ranks registration
crops of the same named GOT-10k class. The query is still a labeled-visible frame
from another sequence. All source imagery remains private research data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def summary(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": round(float(a.min()), 5),
        "p10": round(float(np.percentile(a, 10)), 5),
        "median": round(float(np.median(a)), 5),
        "p90": round(float(np.percentile(a, 90)), 5),
        "max": round(float(a.max()), 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-similarity", type=float, default=0.0)
    args = parser.parse_args()
    source, candidate, output = args.source.resolve(), args.candidate.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit("Refusing to overwrite an existing appearance-selected set.")
    source_manifest = json.loads((source / "manifest.json").read_text())
    candidate_manifest = json.loads((candidate / "manifest.json").read_text())
    if sha256(source / "samples.jsonl") != source_manifest["samples_sha256"]:
        raise ValueError("Source SFT samples changed.")
    if sha256(candidate / "samples.jsonl") != candidate_manifest["samples_sha256"]:
        raise ValueError("Candidate samples changed.")
    source_rows = load_rows(source / "samples.jsonl")
    candidate_rows = load_rows(candidate / "samples.jsonl")
    hard_rows = [r for r in candidate_rows if r["kind"] == "same_class_cross_sequence"]
    if not hard_rows:
        raise ValueError("Candidate set contains no same-class cross-sequence rows.")

    by_class: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    reference_by_sequence: dict[str, str] = {}
    for row in source_rows:
        if row["kind"] != "same_sequence" or row["present"] is not True:
            continue
        name = row["target"].removeprefix("Registered ").strip().casefold()
        by_class[name][row["sequence"]].append(row)
        previous = reference_by_sequence.setdefault(row["sequence"], row["reference"])
        if previous != row["reference"]:
            raise ValueError(f"Inconsistent reference crop for {row['sequence']}")

    classes = {r["class_name"] for r in hard_rows}
    reference_paths = sorted(
        {r["reference"] for r in hard_rows}
        | {
            reference_by_sequence[seq]
            for name in classes
            for seq in by_class[name]
        }
    )
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    torch.cuda.set_per_process_memory_fraction(0.08)
    processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).to("cuda").eval()
    features = []
    for start in range(0, len(reference_paths), args.batch_size):
        paths = reference_paths[start : start + args.batch_size]
        images = [Image.open(source / p).convert("RGB") for p in paths]
        inputs = processor(images=images, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state.float()
            vectors = torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=-1)
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
        features.append(vectors.cpu().numpy())
        for image in images:
            image.close()
        if start % 480 == 0:
            print(json.dumps({"embedded": min(start + len(paths), len(reference_paths)), "total": len(reference_paths)}), flush=True)
    matrix = np.concatenate(features, axis=0)
    embedding_by_path = dict(zip(reference_paths, matrix, strict=True))
    rng = random.Random(args.seed)
    used: set[tuple[str, str]] = set()
    chosen = []
    old_scores = []
    new_scores = []
    for row in candidate_rows:
        if row["kind"] != "same_class_cross_sequence":
            chosen.append(row)
            continue
        name, src_seq = row["class_name"], row["source_sequence"]
        src_ref = row["reference"]
        old_ref = reference_by_sequence[row["query_sequence"]]
        old_scores.append(float(embedding_by_path[src_ref] @ embedding_by_path[old_ref]))
        ranked = []
        for query_seq, query_rows in by_class[name].items():
            if query_seq == src_seq:
                continue
            query_ref = reference_by_sequence[query_seq]
            if query_ref == src_ref:
                continue
            score = float(embedding_by_path[src_ref] @ embedding_by_path[query_ref])
            if score < args.min_similarity or score >= 0.9999:
                continue
            ranked.append((score, query_seq, query_ref, query_rows))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        replacement = None
        for score, query_seq, query_ref, query_rows in ranked:
            options = query_rows.copy()
            rng.shuffle(options)
            for query in options:
                pair = (src_ref, query["image"])
                if pair in used:
                    continue
                used.add(pair)
                replacement = {
                    **row,
                    "sequence": f"{src_seq}->{query_seq}",
                    "query_sequence": query_seq,
                    "frame": query["frame"],
                    "image": query["image"],
                    "query_reference": query_ref,
                    "appearance_similarity": round(score, 6),
                }
                new_scores.append(score)
                break
            if replacement:
                break
        if replacement is None:
            raise ValueError(f"No visually ranked unique query for {name}/{src_seq}")
        chosen.append(replacement)
    if len(new_scores) != len(hard_rows):
        raise AssertionError("The hard-negative count changed.")
    output.mkdir(parents=True)
    (output / "images").symlink_to(os.path.relpath(source / "images", output), target_is_directory=True)
    with (output / "samples.jsonl").open("w") as stream:
        for row in chosen:
            stream.write(json.dumps(row) + "\n")
    np.savez_compressed(output / "reference_embeddings.npz", paths=np.array(reference_paths), vectors=matrix)
    model_path = Path(args.model).expanduser()
    if model_path.is_dir():
        weights = model_path / "model.safetensors"
    else:
        cache_base = Path.home() / f".cache/huggingface/hub/models--{args.model.replace('/', '--')}"
        snapshots = sorted((cache_base / "snapshots").glob("*"))
        weights = snapshots[0] / "model.safetensors" if len(snapshots) == 1 else None
    manifest = {
        "schema_version": 1,
        "purpose": "Appearance-ranked same-class proxy negatives for GOT-10k research SFT",
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "candidate_manifest_sha256": sha256(candidate / "manifest.json"),
        "candidate_samples_sha256": sha256(candidate / "samples.jsonl"),
        "model": args.model,
        "model_weights_sha256": sha256(weights) if weights and weights.is_file() else None,
        "model_method": "DINOv2 CLS plus mean-patch L2-normalized cosine of the two sequences' registration crops; ranking only, not an identity label",
        "embeddings_sha256": sha256(output / "reference_embeddings.npz"),
        "reference_crops_embedded": len(reference_paths),
        "old_candidate_similarity": summary(old_scores),
        "appearance_selected_similarity": summary(new_scores),
        "args": {k: str(v) for k, v in vars(args).items()},
        "script_sha256": sha256(Path(__file__)),
        "samples_sha256": sha256(output / "samples.jsonl"),
        "limitations": [
            "Same named class and high DINOv2 similarity do not prove that two recordings contain different physical objects.",
            "The ranking model can emphasize background, pose or species and needs a visual spot-check.",
            "The derived images and embeddings stay private under GOT-10k's research-data license.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(chosen), "hard": len(new_scores), "old": manifest["old_candidate_similarity"], "new": manifest["appearance_selected_similarity"]}), flush=True)


if __name__ == "__main__":
    main()
