"""Build reference-crop localization fine-tuning samples from GOT-10k training sequences.

Every sample is rendered with the application's own grounding prompt builders and image
encoders, so an adapter trained on it sees the bytes the deployed review sends:

* reference: the target's crop from one clear frame of the same sequence, encoded like a
  registration crop (JPEG quality 90 at native crop size);
* frame: another frame of the sequence through `bounded_jpeg(frame, 768)`;
* target answer: the integer 0..1000 contract, `{"present", "x", "y", "note"}`.

Labels come only from GOT-10k annotations. A frame is present when `absence == 0` and
`cover >= 2`; absent when `absence == 1` or `cover == 0`; `cover == 1` (a sliver visible) is
skipped as ambiguous. The point is the annotated box centre. Sequences whose class names
overlap the evaluation categories (books, cups, bottles and similar tableware) are excluded, so
the LaSOT evaluation categories stay unseen, and so are the `person` and `object part` roots
(people, faces, hands): the application never identifies people.

GOT-10k is licensed CC BY-NC-SA 4.0 for research; samples stay in local research artifacts and
are never published with the source.
"""

import argparse
import configparser
import hashlib
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np

from agentx.agents.skills import SkillRegistry
from agentx.vision.video import bounded_jpeg, jpeg

# Whole words only: suffix matching would wrongly drop "african elephant" or "pelican".
EXCLUDE_TERMS = (
    "book",
    "books",
    "notebook",
    "cup",
    "cups",
    "teacup",
    "mug",
    "bottle",
    "bottles",
    "glass",
    "glasses",
    "wineglass",
    "goblet",
    "tumbler",
    "flask",
    "jar",
    "vase",
    "can",
    "carton",
    "tableware",
)
# FindBack never identifies people, and its localization prompt says so. Training the reviewer to
# re-find a particular person, face or hand from a crop would be person re-identification and would
# contradict that instruction, so whole root classes are removed, not just words.
EXCLUDE_ROOTS = ("person", "object part")
MAX_EDGE = 768


def excluded(meta: dict) -> str | None:
    """Why a sequence is left out: a person/body-part root, or an evaluation-category term."""
    root = meta.get("root_class", "").strip().lower()
    if root in EXCLUDE_ROOTS:
        return f"root:{root}"
    names = " ".join(meta.get(k, "") for k in ("object_class", "major_class", "root_class"))
    words = set(re.findall(r"[a-z]+", names.lower()))
    for term in EXCLUDE_TERMS:
        if term in words:
            return term
    return None


def read_meta(path: Path) -> dict:
    parser = configparser.ConfigParser()
    parser.read_string(path.read_text(errors="replace"))
    return dict(parser["METAINFO"]) if "METAINFO" in parser else {}


def read_labels(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.loadtxt(directory / "groundtruth.txt", delimiter=",", ndmin=2)
    absence = np.loadtxt(directory / "absence.label", dtype=int, ndmin=1)
    cover = np.loadtxt(directory / "cover.label", dtype=int, ndmin=1)
    if not len(boxes) == len(absence) == len(cover):
        raise ValueError(f"Annotation lengths differ in {directory.name}")
    return boxes, absence, cover


def state(index: int, absence: np.ndarray, cover: np.ndarray) -> bool | None:
    """True present, False absent, None ambiguous (skip)."""
    if absence[index] == 1 or cover[index] == 0:
        return False
    if cover[index] == 1:
        return None
    return True


def clear_reference(index, boxes, absence, cover, width, height) -> bool:
    x, y, w, h = boxes[index]
    area = w * h / (width * height)
    return bool(
        absence[index] == 0
        and cover[index] == 8
        and w >= 16
        and h >= 16
        and 0.002 <= area <= 0.5
        and x >= 0
        and y >= 0
        and x + w <= width
        and y + h <= height
    )


def answer(present: bool, box, width: int, height: int, name: str) -> str:
    if not present:
        return json.dumps(
            {
                "present": False,
                "x": None,
                "y": None,
                "note": f"The {name} from the reference is not visible in this frame.",
            }
        )
    x, y, w, h = box
    cx = min(1000, max(0, round((x + w / 2) / width * 1000)))
    cy = min(1000, max(0, round((y + h / 2) / height * 1000)))
    return json.dumps(
        {
            "present": True,
            "x": cx,
            "y": cy,
            "note": f"The {name} from the reference is visible.",
        }
    )


def crop_without_target(image: np.ndarray, box, rng: random.Random, tries: int = 30):
    """The largest of a 60%, 50% or 40% window of the same scene that keeps a margin away from
    the target box, so the target is out of view while the scene stays the same."""
    height, width = image.shape[:2]
    x, y, w, h = box
    margin_x, margin_y = max(12.0, 0.15 * w), max(12.0, 0.15 * h)
    left, top = x - margin_x, y - margin_y
    right, bottom = x + w + margin_x, y + h + margin_y
    for share in (0.6, 0.5, 0.4):
        cw, ch = int(share * width), int(share * height)
        for _ in range(tries):
            cx, cy = rng.randint(0, width - cw), rng.randint(0, height - ch)
            if cx + cw <= left or cx >= right or cy + ch <= top or cy >= bottom:
                return image[cy : cy + ch, cx : cx + cw]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--got-root", required=True, type=Path, help="GOT-10k/train directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-sequence", type=int, default=2)
    parser.add_argument("--cross-negative-fraction", type=float, default=0.08)
    parser.add_argument(
        "--crop-negatives",
        type=int,
        default=0,
        help="Per sequence: crops of a visible frame that exclude the target (same scene, "
        "target out of view), labelled absent",
    )
    parser.add_argument(
        "--all-absences",
        action="store_true",
        help="Use every extracted annotated absence, not at most one per sequence",
    )
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument(
        "--min-gap", type=int, default=10, help="Frames between reference and query"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing sample set.")
    rng = random.Random(args.seed)
    images_dir = args.output / "images"
    images_dir.mkdir(parents=True)
    skill, trace = SkillRegistry().load("review-visual-evidence")
    sequences = sorted(p for p in args.got_root.iterdir() if p.is_dir())
    rng.shuffle(sequences)
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]
    stats = {"sequences_seen": 0, "excluded": {}, "no_reference": 0, "samples": 0, "present": 0}
    pool: list[tuple[Path, dict]] = []
    rows = []

    def frame_path(directory: Path, index: int) -> Path:
        return directory / f"{index + 1:08d}.jpg"

    def write(image: np.ndarray, *, bounded: bool) -> str:
        data = bounded_jpeg(image, MAX_EDGE) if bounded else jpeg(image)
        digest = hashlib.sha256(data).hexdigest()
        path = images_dir / f"{digest[:2]}/{digest}.jpg"
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return str(path.relative_to(args.output))

    for directory in sequences:
        stats["sequences_seen"] += 1
        meta_path = directory / "meta_info.ini"
        if not meta_path.exists():
            continue
        meta = read_meta(meta_path)
        term = excluded(meta)
        if term:
            stats["excluded"][term] = stats["excluded"].get(term, 0) + 1
            continue
        try:
            boxes, absence, cover = read_labels(directory)
        except (OSError, ValueError):
            continue
        available = [i for i in range(len(boxes)) if frame_path(directory, i).exists()]
        if len(available) < 3:
            continue
        first = cv2.imread(str(frame_path(directory, available[0])))
        if first is None:
            continue
        height, width = first.shape[:2]
        references = [
            i for i in available if clear_reference(i, boxes, absence, cover, width, height)
        ]
        if not references:
            stats["no_reference"] += 1
            continue
        ref_index = references[0]
        ref_image = cv2.imread(str(frame_path(directory, ref_index)))
        x, y, w, h = (int(round(v)) for v in boxes[ref_index])
        crop = ref_image[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        name = meta.get("object_class", "object").strip() or "object"
        record = {"directory": directory, "ref": write(crop, bounded=False), "name": name}
        pool.append((directory, record | {"root": meta.get("root_class", "")}))
        queries = [
            i
            for i in available
            if abs(i - ref_index) >= args.min_gap and state(i, absence, cover) is not None
        ]
        absent_queries = [i for i in queries if state(i, absence, cover) is False]
        chosen = rng.sample(queries, min(args.per_sequence, len(queries)))
        if args.all_absences:
            chosen = [i for i in chosen if i not in absent_queries] + absent_queries
        elif absent_queries and not any(i in absent_queries for i in chosen):
            # Keep annotated same-scene absences: they are the realistic negatives.
            chosen[-1:] = [rng.choice(absent_queries)]
        visible_queries = [i for i in queries if state(i, absence, cover)]
        for index in rng.sample(visible_queries, min(args.crop_negatives, len(visible_queries))):
            image = cv2.imread(str(frame_path(directory, index)))
            if image is None or image.shape[:2] != (height, width):
                continue
            cropped = crop_without_target(image, boxes[index], rng)
            if cropped is None:
                continue
            rows.append(
                {
                    "sequence": directory.name,
                    "frame": index + 1,
                    "kind": "crop_negative",
                    "target": f"Registered {name}",
                    "reference": record["ref"],
                    "image": write(cropped, bounded=True),
                    "answer": answer(False, None, 1, 1, name),
                    "present": False,
                }
            )
        for index in chosen:
            image = cv2.imread(str(frame_path(directory, index)))
            if image is None or image.shape[:2] != (height, width):
                continue
            present = bool(state(index, absence, cover))
            rows.append(
                {
                    "sequence": directory.name,
                    "frame": index + 1,
                    "kind": "same_sequence",
                    "target": f"Registered {name}",
                    "reference": record["ref"],
                    "image": write(image, bounded=True),
                    "answer": answer(present, boxes[index], width, height, name),
                    "present": present,
                }
            )
    # Cross-sequence negatives: a reference from one sequence, a frame from another sequence of
    # the same root class. The exact registered object is not in that video.
    by_root: dict[str, list[dict]] = {}
    for _, record in pool:
        by_root.setdefault(record["root"], []).append(record)
    wanted = int(len(rows) * args.cross_negative_fraction)
    candidates = [r for r in pool if len(by_root.get(r[1]["root"], [])) > 1]
    for _ in range(wanted):
        if not candidates:
            break
        _, source = rng.choice(candidates)
        other = rng.choice([r for r in by_root[source["root"]] if r is not source])
        directory = other["directory"]
        frames_there = sorted(directory.glob("*.jpg"))
        image = cv2.imread(str(rng.choice(frames_there)))
        if image is None:
            continue
        rows.append(
            {
                "sequence": f"{source['directory'].name}->{directory.name}",
                "frame": None,
                "kind": "cross_sequence",
                "target": f"Registered {source['name']}",
                "reference": source["ref"],
                "image": write(image, bounded=True),
                "answer": answer(False, None, 1, 1, source["name"]),
                "present": False,
            }
        )
    rng.shuffle(rows)
    stats["samples"] = len(rows)
    stats["present"] = sum(r["present"] for r in rows)
    stats["cross_sequence"] = sum(r["kind"] == "cross_sequence" for r in rows)
    stats["crop_negative"] = sum(r["kind"] == "crop_negative" for r in rows)
    with (args.output / "samples.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    manifest = {
        "source": "GOT-10k train (CC BY-NC-SA 4.0), research use only",
        "rules": __doc__.split("\n\n")[3].replace("\n", " "),
        "exclude_terms": EXCLUDE_TERMS,
        "exclude_roots": EXCLUDE_ROOTS,
        "max_edge": MAX_EDGE,
        "skill": trace.model_dump(),
        "skill_body_sha256": hashlib.sha256(skill.encode()).hexdigest(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "stats": stats,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "samples_sha256": hashlib.sha256((args.output / "samples.jsonl").read_bytes()).hexdigest(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
