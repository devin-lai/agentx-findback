import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scripts_helper import load_script

from agentx.vision.video import probe

load = load_script


def object_photo(path: Path, color) -> None:
    image = np.full((300, 140, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (120, 280), color, -1)
    cv2.circle(image, (70, 80), 18, (240, 240, 240), -1)  # a white part that must survive
    cv2.imwrite(str(path), image)


def test_composite_fixture_writes_video_manifest_and_labels(tmp_path, monkeypatch):
    for name, color in (("a", (20, 20, 20)), ("b", (200, 200, 200)), ("c", (30, 120, 200))):
        object_photo(tmp_path / f"{name}.png", color)
    maker = load("make_composite_fixture")
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_composite_fixture",
            "--output",
            str(out),
            "--seconds",
            "2",
            "--fps",
            "4",
            "--object",
            f"Mover=remote={tmp_path / 'a.png'}",
            "--object",
            f"Pale=remote={tmp_path / 'b.png'}",
            "--object",
            f"Blue=cup={tmp_path / 'c.png'}",
        ],
    )
    maker.main()
    info = probe(out / "composite.mp4")
    assert (info.width, info.height, info.duration_ms) == (960, 540, 2000)
    manifest = json.loads((out / "manifest.json").read_text())
    evaluator = load("evaluate")
    dataset = evaluator.Manifest.model_validate(manifest)
    assert [o.name for o in dataset.clips[0].objects] == ["Mover", "Pale", "Blue"]
    assert all(o.box.x2 > o.box.x1 for o in dataset.clips[0].objects)
    assert len(dataset.clips[0].cases) == 10
    labels = json.loads((out / "labels.json").read_text())
    assert len(labels) == 8 and set(labels[0]["boxes"]) == {"Mover", "Pale", "Blue"}
    # The mover's first-frame box sits in the left third; the pale look-alike in the center third.
    first = labels[0]["boxes"]
    assert (first["Mover"][0] + first["Mover"][2]) / 2 < 1 / 3
    assert 1 / 3 < (first["Pale"][0] + first["Pale"][2]) / 2 < 2 / 3
    # Flood-fill cutout keeps the interior white part of the pale object.
    image, mask = maker.cutout(tmp_path / "b.png", 100)
    assert mask.mean() > 0.6 and image.shape[0] == 100


def test_camera_motion_moves_the_camera_and_leaves_the_objects_alone(tmp_path, monkeypatch):
    """The camera-motion fixture was unreachable once: `desk()` crashed on its larger canvas."""
    for name, color in (("a", (20, 20, 20)), ("b", (200, 200, 200)), ("c", (30, 120, 200))):
        object_photo(tmp_path / f"{name}.png", color)
    maker = load("make_composite_fixture")
    out = tmp_path / "bumped"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_composite_fixture",
            "--output",
            str(out),
            "--seconds",
            "13",
            "--fps",
            "2",
            "--camera-motion",
            "--object",
            f"Mover=remote={tmp_path / 'a.png'}",
            "--object",
            f"Pale=remote={tmp_path / 'b.png'}",
            "--object",
            f"Blue=cup={tmp_path / 'c.png'}",
        ],
    )
    maker.main()
    info = probe(out / "composite.mp4")
    assert (info.width, info.height) == (960, 540)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["camera"]["motion"] is True
    assert manifest["camera"]["objects_move_during_camera_motion"] is False
    labels = json.loads((out / "labels.json").read_text())
    before = [row for row in labels if not row["camera_moved"]]
    after = [row for row in labels if row["camera_moved"]]
    assert before and after
    # A resting object's pixels move with the camera; its registered region must not.
    assert before[-1]["boxes"]["Pale"][0] != after[-1]["boxes"]["Pale"][0]
