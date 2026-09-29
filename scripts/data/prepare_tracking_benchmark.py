"""Convert selected LaSOT image sequences and independent labels into AgentX clips.

The assigned playback rate is explicit: these archives contain ordered JPEGs,
not the original video's presentation timestamps. Ground truth is copied from
the dataset, never inferred by AgentX or a language model.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

import av
import cv2
import numpy as np
from pydantic import model_validator

from agentx.domain.contracts import Box, default_regions, region_for


class AnnotationBox(Box):
    """Keep tiny positive external labels, independent of the UI registration minimum."""

    @model_validator(mode="after")
    def nonempty(self) -> "AnnotationBox":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("An external annotation must have positive clipped area.")
        return self


def normalized_box(raw, width, height):
    x, y, w, h = map(float, raw)
    return AnnotationBox(
        x1=max(0, min(1, x / width)),
        y1=max(0, min(1, y / height)),
        x2=max(0, min(1, (x + w) / width)),
        y2=max(0, min(1, (y + h) / height)),
    )


def prepare(archive: Path, sequence: str, output: Path, fps: int, max_frames: int) -> dict:
    category = sequence.rsplit("-", 1)[0]
    if category not in {"book", "cup", "bottle", "electricfan", "guitar"}:
        raise ValueError(
            "This adapter maps only book/cup/bottle/electricfan/guitar. "
            "LaSOT 'mouse' means an animal, not a computer mouse."
        )
    with ZipFile(archive) as source:
        prefix = sequence + "/"
        annotations = {
            name: source.read(prefix + name)
            for name in [
                "groundtruth.txt",
                "full_occlusion.txt",
                "out_of_view.txt",
                "nlp.txt",
            ]
        }
        boxes = np.loadtxt(annotations["groundtruth.txt"].decode().splitlines(), delimiter=",")
        occluded = np.fromstring(annotations["full_occlusion.txt"].decode(), sep=",", dtype=int)
        outside = np.fromstring(annotations["out_of_view.txt"].decode(), sep=",", dtype=int)
        images = sorted(
            n for n in source.namelist() if n.startswith(prefix + "img/") and n.endswith(".jpg")
        )
        if not (len(boxes) == len(occluded) == len(outside) == len(images)):
            raise ValueError(f"Annotation/image count mismatch in {sequence}")
        count = min(len(images), max_frames)
        if occluded[0] or outside[0]:
            raise ValueError("The first frame must contain a visible registration target.")
        output.mkdir(parents=True, exist_ok=True)
        for name, content in annotations.items():
            (output / name).write_bytes(content)
        first = cv2.imdecode(np.frombuffer(source.read(images[0]), np.uint8), cv2.IMREAD_COLOR)
        height, width = first.shape[:2]
        if width % 2 or height % 2:
            raise ValueError(
                "This converter preserves source dimensions; H.264 requires even dimensions."
            )
        labels = []
        image_hash = hashlib.sha256()
        clip = output / f"{sequence}.mp4"
        with av.open(str(clip), "w") as video:
            stream = video.add_stream("libx264", rate=fps)
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            stream.options = {"crf": "18", "preset": "fast"}
            for index, name in enumerate(images[:count]):
                raw = source.read(name)
                image_hash.update(raw)
                image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if image.shape[:2] != (height, width):
                    raise ValueError("Image dimensions changed within the sequence.")
                frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                frame.pts, frame.time_base = index, Fraction(1, fps)
                for packet in stream.encode(frame):
                    video.mux(packet)
                present = not bool(occluded[index] or outside[index])
                box = normalized_box(boxes[index], width, height) if present else None
                labels.append(
                    {
                        "frame": index + 1,
                        "at_ms": round(index * 1000 / fps),
                        "visible": present,
                        "full_occlusion": bool(occluded[index]),
                        "out_of_view": bool(outside[index]),
                        "box": box.model_dump() if box else None,
                        "zone": region_for(box, default_regions()) if box else None,
                    }
                )
            for packet in stream.encode():
                video.mux(packet)
    registration = {"name": sequence, "label": category, "at_ms": 0, "box": labels[0]["box"]}
    cases = []
    latest = None
    for index, label in enumerate(labels):
        if label["visible"]:
            latest = label
        if index % fps or index == 0:
            continue
        cases.append(
            {
                "question": f"Where is {sequence}?",
                "at_ms": label["at_ms"],
                "expected_object": sequence,
                "expected_status": "visible"
                if label["visible"]
                else "last_seen"
                if latest
                else "not_observed",
                "expected_zone": latest["zone"] if latest else None,
                "expected_last_seen_ms": latest["at_ms"] if latest else None,
                "tolerance_ms": 250,
            }
        )
    with clip.open("rb") as stream:
        clip_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    data = {
        "schema_version": "1",
        "sequence": sequence,
        "category": category,
        "source": "LaSOT CVPR 2019; public tracking research footage and upstream annotations",
        "description": annotations["nlp.txt"].decode().strip(),
        "assigned_fps": fps,
        "frames": len(labels),
        "original_frame_count": len(images),
        "width": width,
        "height": height,
        "source_images_sha256": image_hash.hexdigest(),
        "video_sha256": clip_sha,
        "annotation_sha256": {n: hashlib.sha256(b).hexdigest() for n, b in annotations.items()},
        "coordinate_convention": "Upstream [x,y,width,height] divided by image dimensions and clipped to frame bounds.",
        "visibility_convention": "visible iff both upstream full_occlusion and out_of_view flags are zero; partial occlusion remains visible.",
        "timing_note": "Ordered source JPEGs encoded at the explicitly assigned rate; not recovered source presentation timestamps.",
        "labels": labels,
    }
    (output / "labels.json").write_text(json.dumps(data, indent=2) + "\n")
    manifest = {
        "name": f"LaSOT {sequence} public research stress test",
        "provenance": data["source"] + ". " + data["timing_note"],
        "clips": [{"path": clip.name, "objects": [registration], "cases": cases}],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {k: v for k, v in data.items() if k != "labels"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, default=Path("artifacts/datasets/lasot-raw"))
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/datasets/lasot-prepared"))
    parser.add_argument("--fps", type=int, default=30, choices=range(1, 61))
    parser.add_argument("--max-frames", type=int, default=1800)
    args = parser.parse_args()
    if args.max_frames < 2:
        raise SystemExit("At least two frames are required.")
    for seq in args.sequences:
        category = seq.rsplit("-", 1)[0]
        if not seq.replace("-", "").isalnum() or seq.count("-") != 1:
            raise SystemExit("Expected a sequence name such as book-2.")
        print(
            json.dumps(
                prepare(
                    args.archives / f"{category}.zip",
                    seq,
                    args.output / seq,
                    args.fps,
                    args.max_frames,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
