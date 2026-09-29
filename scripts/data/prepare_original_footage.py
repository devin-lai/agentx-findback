"""Build private contact sheets and a blank annotation draft for original desk footage.

This tool does not infer labels or contact a model. Keep its output under ignored
``artifacts/``: contact sheets retain source video pixels, including any private detail.
The draft deliberately fails ``evaluate_real_clips.py`` preflight until a human assigns
splits, registration boxes and questions from the footage.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from agentx.domain.contracts import timestamp
from agentx.vision.video import frames, probe

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
TILE_WIDTH = 480
TILE_HEIGHT = 288
COLUMNS = 4
ROWS = 4


def contact_sheet(samples: list[tuple[int, np.ndarray]], output: Path) -> None:
    """Render 16 upright source frames with timestamps and the default region thirds."""
    sheet = np.full((ROWS * TILE_HEIGHT, COLUMNS * TILE_WIDTH, 3), 30, dtype=np.uint8)
    for position, (at_ms, frame) in enumerate(samples):
        y0 = (position // COLUMNS) * TILE_HEIGHT
        x0 = (position % COLUMNS) * TILE_WIDTH
        image_height, image_width = frame.shape[:2]
        scale = min(TILE_WIDTH / image_width, (TILE_HEIGHT - 18) / image_height)
        width, height = round(image_width * scale), round(image_height * scale)
        image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        left = x0 + (TILE_WIDTH - width) // 2
        top = y0 + 18 + (TILE_HEIGHT - 18 - height) // 2
        sheet[top : top + height, left : left + width] = image
        for third in (1, 2):
            x = left + round(width * third / 3)
            cv2.line(sheet, (x, top), (x, top + height - 1), (0, 230, 230), 1)
        cv2.putText(
            sheet,
            timestamp(at_ms),
            (x0 + 8, y0 + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(output), sheet):
        raise OSError(f"Could not write contact sheet: {output}")


def prepare(clips_dir: Path, output_dir: Path, *, sample_fps: float = 2.0) -> dict:
    if not 0 < sample_fps <= 5:
        raise ValueError("sample_fps must be above zero and at most five.")
    root = clips_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("clips_dir must be a directory.")
    clips = sorted(path for path in root.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES)
    if not clips:
        raise ValueError("No supported video clips found in clips_dir.")
    if output_dir.exists():
        raise FileExistsError(
            "Choose a new output directory; do not overwrite an annotation draft."
        )

    staged = []
    for path in clips:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Clip must be a regular file in clips_dir: {path.name}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        info = probe(path)
        staged.append((path, digest, info))

    output_dir.mkdir(parents=True)
    draft = {
        "tolerance_ms": 400,
        "clips": [],
    }
    receipt = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "Private human-label intake; no model requests or truth labels.",
        "sample_fps": sample_fps,
        "clips": [],
    }
    for path, digest, info in staged:
        clip_dir = output_dir / f"clip-{len(draft['clips']) + 1:02d}"
        clip_dir.mkdir()
        page = []
        sheets = []
        sample_count = 0
        for at_ms, image in frames(path, sample_fps=sample_fps):
            page.append((at_ms, image))
            sample_count += 1
            if len(page) == COLUMNS * ROWS:
                destination = clip_dir / f"sheet-{len(sheets) + 1:02d}.png"
                contact_sheet(page, destination)
                sheets.append(str(destination.relative_to(output_dir)))
                page = []
        if page:
            destination = clip_dir / f"sheet-{len(sheets) + 1:02d}.png"
            contact_sheet(page, destination)
            sheets.append(str(destination.relative_to(output_dir)))
        if not sample_count:
            raise ValueError(f"Clip yielded no timestamped frames: {path.name}")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"Source clip changed during intake: {path.name}")
        draft["clips"].append(
            {
                "file": path.name,
                "sha256": digest,
                "split": "SET_DEVELOPMENT_OR_EVALUATION",
                "objects": [],
                "questions": [],
            }
        )
        receipt["clips"].append(
            {
                "file": path.name,
                "sha256": digest,
                "bytes": path.stat().st_size,
                "duration_ms": info.duration_ms,
                "width": info.width,
                "height": info.height,
                "source_fps": info.fps,
                "samples": sample_count,
                "sheets": sheets,
            }
        )
    (output_dir / "annotation-draft.json").write_text(json.dumps(draft, indent=2) + "\n")
    (output_dir / "intake.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    args = parser.parse_args()
    receipt = prepare(args.clips, args.output, sample_fps=args.sample_fps)
    print(
        json.dumps(
            {
                "status": "draft_needs_human_labels",
                "clips": len(receipt["clips"]),
                "sheets": sum(len(clip["sheets"]) for clip in receipt["clips"]),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
