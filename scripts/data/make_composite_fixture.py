"""Compose a labeled test clip from real object photos: real textures, scripted motion.

The output is still not a real recording (motion, lighting and occlusion are synthetic), but the
objects are real photographs, so learned detectors and identity embeddings see realistic
appearance, including look-alike objects of one category. It writes composite.mp4, a manifest for
scripts/eval/evaluate.py and per-frame labels for later per-frame scoring. No image is committed.

Example:
  python scripts/data/make_composite_fixture.py --output artifacts/composite \\
    --object "Black remote=remote=objects/remote-black.jpg" \\
    --object "White short remote=remote=objects/remote-white-short.jpg" \\
    --object "White long remote=remote=objects/remote-white-long.jpg"
"""

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np

WIDTH, HEIGHT = 960, 540
# A camera-motion clip is composed on a larger canvas so a pan reveals real scene, not black
# borders. The viewport starts centred, so the first frame matches the fixed-camera fixture.
CANVAS_SCALE = 1.5


def cutout(path: Path, height: int) -> tuple[np.ndarray, np.ndarray]:
    """BGR cutout and alpha mask from a photo on a near-white background."""
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"Cannot read {path}")
    # Flood the connected near-white background from the corners so white object parts survive.
    h, w = image.shape[:2]
    flood = np.zeros((h + 2, w + 2), np.uint8)
    work = image.copy()
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        cv2.floodFill(
            work,
            flood,
            seed,
            (0, 0, 0),
            (14, 14, 14),
            (14, 14, 14),
            cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8) | 4,
        )
    mask = (flood[1:-1, 1:-1] == 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    ys, xs = np.nonzero(mask)
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    image, mask = image[y1:y2, x1:x2], mask[y1:y2, x1:x2]
    scale = height / image.shape[0]
    size = (max(8, round(image.shape[1] * scale)), height)
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return image, mask


def desk(rng: np.random.Generator, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    grain = (
        np.sin(ys / 9 + 3 * np.sin(xs / 140)) * 6
        + np.sin(ys / 31 + xs / 400) * 5
        + rng.normal(0, 3, (height, width))
    )
    base = np.stack([70 + grain, 105 + grain * 1.2, 150 + grain * 1.4], axis=-1)
    vignette = 1 - 0.25 * (((xs - width / 2) / width) ** 2 + ((ys - height / 2) / height) ** 2)
    frame = np.clip(base * vignette[..., None], 0, 255).astype(np.uint8)
    dx, dy = (width - WIDTH) // 2, (height - HEIGHT) // 2
    cv2.rectangle(frame, (40 + dx, 30 + dy), (300 + dx, 110 + dy), (210, 220, 225), -1)  # notebook
    cv2.line(frame, (60 + dx, 60 + dy), (280 + dx, 62 + dy), (90, 90, 90), 2)
    cv2.line(frame, (60 + dx, 84 + dy), (240 + dx, 86 + dy), (90, 90, 90), 2)
    cv2.line(frame, (620 + dx, 40 + dy), (900 + dx, 90 + dy), (30, 30, 30), 5)  # a pen
    if (dx, dy) != (0, 0):
        # Texture the revealed margin so a panned view still has something to match against.
        cv2.rectangle(frame, (dx - 90, dy + 250), (dx - 20, dy + 430), (120, 150, 190), -1)
        cv2.rectangle(
            frame, (dx + WIDTH + 25, dy + 120), (dx + WIDTH + 110, dy + 300), (60, 90, 130), -1
        )
        for i in range(0, height, 44):
            cv2.line(frame, (0, i), (width, i + 9), (95, 125, 165), 1)
    return frame


def viewport(t: float, motion: bool, width: int, height: int) -> tuple[float, float, float]:
    """Camera offset and zoom into the canvas at time t: (left, top, scale).

    Without motion the viewport never leaves its registered pose. With motion the camera is
    knocked between 10 and 12 seconds and stays there, which is what a bumped tripod does.
    """
    left, top = (width - WIDTH) / 2, (height - HEIGHT) / 2
    if not motion or t < 10:
        return left, top, 1.0
    progress = min(1.0, (t - 10) / 2)
    eased = progress * progress * (3 - 2 * progress)
    return left + 168 * eased, top - 96 * eased, 1 + 0.14 * eased


def crop_view(canvas: np.ndarray, view: tuple[float, float, float]) -> np.ndarray:
    left, top, scale = view
    w, h = WIDTH / scale, HEIGHT / scale
    matrix = np.array([[scale, 0, -left * scale], [0, scale, -top * scale]], dtype=np.float32)
    del w, h
    return cv2.warpAffine(
        canvas, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )


def view_box(
    box: list[float], view: tuple[float, float, float], width: int, height: int
) -> list[float]:
    """Map a canvas-normalized box into the frame the moved camera produced."""
    left, top, scale = view
    x1, y1, x2, y2 = box
    return [
        (x1 * width - left) * scale / WIDTH,
        (y1 * height - top) * scale / HEIGHT,
        (x2 * width - left) * scale / WIDTH,
        (y2 * height - top) * scale / HEIGHT,
    ]


def paste(frame: np.ndarray, image: np.ndarray, mask: np.ndarray, x: int, y: int) -> None:
    h, w = image.shape[:2]
    region = frame[y : y + h, x : x + w]
    alpha = mask[..., None]
    region[:] = (region * (1 - alpha) + image * alpha).astype(np.uint8)


def mover_position(t: float) -> tuple[int, int] | None:
    """Left 0-6 s, glide 6-7 s, right 7-12 s, hidden 12-15 s, center 15-20 s."""
    if t < 6:
        return 90, 120
    if t < 7:
        return round(90 + (700 - 90) * (t - 6)), 120
    if t < 12:
        return 700, 120
    if t < 15:
        return None
    return 400, 120


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--object", action="append", required=True, help='"Name=label=path"; first one moves'
    )
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--camera-motion",
        action="store_true",
        help="Knock the camera between 10 and 12 seconds and leave it there. Objects do not move.",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    specs = []
    for raw in args.object:
        name, label, path = raw.split("=", 2)
        specs.append((name.strip(), label.strip().lower(), Path(path)))
    cut = [cutout(path, 200 if i == 0 else 170) for i, (_, _, path) in enumerate(specs)]
    static_positions = [(430, 340), (760, 340), (120, 330)]
    canvas_w = round(WIDTH * CANVAS_SCALE) if args.camera_motion else WIDTH
    canvas_h = round(HEIGHT * CANVAS_SCALE) if args.camera_motion else HEIGHT
    offset_x, offset_y = (canvas_w - WIDTH) // 2, (canvas_h - HEIGHT) // 2
    background = desk(rng, canvas_w, canvas_h)
    args.output.mkdir(parents=True, exist_ok=True)
    video = args.output / "composite.mp4"
    labels = []
    with av.open(str(video), mode="w") as container:
        stream = container.add_stream("libx264", rate=args.fps)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        stream.options = {"crf": "20", "preset": "fast"}
        for index in range(args.seconds * args.fps):
            t = index / args.fps
            frame = background.copy()
            boxes = {}
            for i, ((name, _, _), (image, mask)) in enumerate(zip(specs, cut, strict=True)):
                if i == 0:
                    position = mover_position(t)
                else:
                    position = static_positions[i - 1]
                if position is None:
                    continue
                jitter = rng.integers(-1, 2, 2)
                x = int(position[0] + jitter[0]) + offset_x
                y = int(position[1] + jitter[1]) + offset_y
                paste(frame, image, mask, x, y)
                h, w = image.shape[:2]
                boxes[name] = [x / canvas_w, y / canvas_h, (x + w) / canvas_w, (y + h) / canvas_h]
            if 12 <= t < 15:  # a sheet of paper slid over the right area
                cv2.rectangle(
                    frame,
                    (660 + offset_x, 90 + offset_y),
                    (900 + offset_x, 330 + offset_y),
                    (235, 238, 240),
                    -1,
                )
                cv2.line(
                    frame,
                    (690 + offset_x, 150 + offset_y),
                    (870 + offset_x, 150 + offset_y),
                    (150, 150, 150),
                    1,
                )
            view = viewport(t, args.camera_motion, canvas_w, canvas_h)
            if args.camera_motion:
                frame = crop_view(frame, view)
                boxes = {n: view_box(b, view, canvas_w, canvas_h) for n, b in boxes.items()}
            gain = 1 + rng.normal(0, 0.01)
            frame = np.clip(
                frame.astype(np.float32) * gain + rng.normal(0, 1.5, frame.shape), 0, 255
            )
            frame = frame.astype(np.uint8)
            labels.append(
                {
                    "at_ms": round(t * 1000),
                    "boxes": boxes,
                    "camera_moved": bool(args.camera_motion and t >= 10),
                }
            )
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="bgr24")):
                container.mux(packet)
            if index == 0:
                cv2.imwrite(str(args.output / "first-frame.png"), frame)
        for packet in stream.encode():
            container.mux(packet)
    first = labels[0]["boxes"]
    mover = specs[0][0]
    objects = [
        {
            "name": name,
            "label": label,
            "at_ms": 0,
            "box": dict(zip("x1 y1 x2 y2".split(), first[name], strict=True)),
        }
        for name, label, _ in specs
    ]
    cases = [
        {
            "question": f"Where is {mover}?",
            "at_ms": 3000,
            "expected_status": "visible",
            "expected_zone": "left",
        },
        {
            "question": f"Where is {mover}?",
            "at_ms": 9000,
            "expected_status": "visible",
            "expected_zone": "right",
        },
        {
            "question": f"Where was {mover} last seen?",
            "at_ms": 13500,
            "expected_intent": "last_seen",
            "expected_status": "last_seen",
            "expected_zone": "right",
            "expected_last_seen_ms": 11900,
            "tolerance_ms": 600,
        },
        {
            "question": f"Where is {mover}?",
            "at_ms": 18000,
            "expected_status": "visible",
            "expected_zone": "center",
        },
        {
            "question": f"Show the history of {mover}",
            "at_ms": 18000,
            "expected_intent": "history",
            "expected_status": "visible",
            "expected_zone": "center",
        },
        {"question": "Where is my passport?", "at_ms": 5000, "expect_match": False},
    ]
    zones = {1: "center", 2: "right", 3: "left"}
    for i, (name, _, _) in enumerate(specs[1:], start=1):
        cases.append(
            {
                "question": f"Where is {name}?",
                "at_ms": 10000,
                "expected_status": "visible",
                "expected_zone": zones[i],
            }
        )
        cases.append(
            {
                "question": f"Where is {name}?",
                "at_ms": 13500,
                "expected_status": "visible",
                "expected_zone": zones[i],
            }
        )
    if args.camera_motion:
        # Nothing on the desk moves between 10 and 20 seconds; only the camera does. Every
        # expectation below therefore repeats the pre-move region, which is exactly the claim
        # a compensated scene reference has to keep true.
        for i, (name, _, _) in enumerate(specs[1:], start=1):
            cases.append(
                {
                    "question": f"Where is {name}?",
                    "at_ms": 19000,
                    "expected_status": "visible",
                    "expected_zone": zones[i],
                }
            )
        cases.append(
            {
                "question": f"Where is {mover}?",
                "at_ms": 19000,
                "expected_status": "visible",
                "expected_zone": "center",
            }
        )
    manifest = {
        "name": "Composited remote-control clip, bumped camera"
        if args.camera_motion
        else "Composited remote-control clip v1",
        "provenance": "Real public-domain object photographs (Wikimedia Commons) composited with scripted motion, occlusion and noise onto a procedural desk. Not a real recording.",
        "camera": {
            "motion": bool(args.camera_motion),
            "script": "static until 10 s, then a 2 s pan-and-zoom knock that is never undone"
            if args.camera_motion
            else "fixed",
            "objects_move_during_camera_motion": False,
            "labels": "boxes are in the moved frame; expected regions stay those of the "
            "registration frame",
        },
        "clips": [{"path": "composite.mp4", "objects": objects, "cases": cases}],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "labels.json").write_text(json.dumps(labels) + "\n")
    print(
        json.dumps(
            {"video": str(video), "objects": [o["name"] for o in objects], "cases": len(cases)}
        )
    )


if __name__ == "__main__":
    main()
