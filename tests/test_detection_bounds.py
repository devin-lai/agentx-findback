"""A rectangle too thin to register is a missing observation, never a failed run.

`Box` refuses anything narrower than the registration minimum, so every perception adapter has
to decide what an undersized detection means. SAM 2 masks, Cosmos boxes and SAM 2 distractor
proposals already drop them; these check that template matching, RT-DETR tracking and RT-DETR
registration proposals do the same instead of aborting the frame that contains one.
"""

from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import pytest

from agentx.domain.contracts import pixel_box
from agentx.vision.reference import ReferenceDetector
from agentx.vision.rtdetr import RTDetrProposer


def test_pixel_box_normalizes_and_refuses_sub_minimum_rectangles():
    assert pixel_box(100, 50, 300, 200, 400, 200).model_dump() == {
        "x1": 0.25,
        "y1": 0.25,
        "x2": 0.75,
        "y2": 1.0,
    }
    # Clamped to the frame rather than rejected.
    assert pixel_box(-20, -20, 500, 500, 400, 200).model_dump() == {
        "x1": 0.0,
        "y1": 0.0,
        "x2": 1.0,
        "y2": 1.0,
    }
    assert pixel_box(100, 50, 109, 200, 1920, 1080) is None  # under 0.005 of the width
    assert pixel_box(100, 50, 300, 54, 1920, 1080) is None  # under 0.005 of the height
    assert pixel_box(300, 50, 100, 200, 1920, 1080) is None  # inverted


@pytest.fixture
def small_reference(tmp_path) -> tuple[Path, dict, np.ndarray, np.ndarray]:
    """A registered object at the smallest size the catalog accepts on a 4K frame."""
    rng = np.random.default_rng(1)
    width, height = 4096, 2160
    background = rng.integers(0, 60, (height, width, 3), dtype=np.uint8)
    template_w, template_h = 21, 40  # ~0.0051 normalized: just above the registration floor
    patch = rng.integers(0, 255, (template_h, template_w, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "reference.png"), patch)
    descriptor = {
        "id": "object-1",
        "name": "Small tool",
        "label": "custom",
        "registered_at_ms": 0,
        "box": {
            "x1": 0.2,
            "y1": 0.2,
            "x2": 0.2 + template_w / width,
            "y2": 0.2 + template_h / height,
        },
        "reference_path": "reference.png",
    }
    return tmp_path, descriptor, background, patch


def test_reference_detector_reports_a_shrunken_match_as_not_detected(small_reference):
    """The 0.85 search scale of a minimally sized reference falls below the Box minimum."""
    directory, descriptor, background, patch = small_reference
    shrunk = cv2.resize(patch, (round(patch.shape[1] * 0.85), round(patch.shape[0] * 0.85)))
    frame = background.copy()
    frame[500 : 500 + shrunk.shape[0], 900 : 900 + shrunk.shape[1]] = shrunk

    detections = ReferenceDetector([descriptor], directory, 0.82).detect(frame, 1000)

    assert [(d.object_id, d.box, d.reason) for d in detections] == [
        ("object-1", None, "not_detected")
    ]
    assert detections[0].score is not None  # the correlation itself is still recorded


def test_reference_detector_still_locates_a_usable_match(small_reference):
    directory, descriptor, background, patch = small_reference
    frame = background.copy()
    frame[500 : 500 + patch.shape[0], 900 : 900 + patch.shape[1]] = patch

    detection = ReferenceDetector([descriptor], directory, 0.82).detect(frame, 1000)[0]

    assert detection.reason is None and detection.box is not None
    assert detection.box.x1 == pytest.approx(900 / 4096, abs=1e-4)


class StubbedRTDetr:
    """Post-processed RT-DETR output without the weights; geometry handling is what matters."""

    def __init__(self, boxes, labels, scores):
        self.prediction = {"boxes": boxes, "labels": labels, "scores": scores}

    def __call__(self, images, return_tensors=None):
        class Inputs(dict):
            def to(self, *args, **kwargs):
                return self

        return Inputs()

    def post_process_object_detection(self, outputs, target_sizes, threshold):
        import torch

        return [{key: torch.tensor(value) for key, value in self.prediction.items()}]


def test_proposer_drops_one_unusable_detection_instead_of_the_whole_frame():
    # The proposer needs the optional vision extra; a core-only install must skip, not fail.
    torch = pytest.importorskip("torch")

    proposer = RTDetrProposer.__new__(RTDetrProposer)
    proposer.torch = torch
    proposer.device = "cpu"
    proposer.inference_lock = Lock()
    proposer.labels = {0: "cell phone", 1: "cup"}
    proposer.processor = StubbedRTDetr(
        boxes=[[50.0, 50.0, 200.0, 220.0], [400.0, 100.0, 402.0, 160.0]],
        labels=[1, 0],
        scores=[0.93, 0.71],
    )
    proposer.model = lambda **kwargs: None

    proposals = proposer.propose(np.zeros((360, 640, 3), dtype=np.uint8), 0.5)

    # The two-pixel-wide phone is unusable; the cup beside it must still reach the person.
    assert [name for name, _, _ in proposals] == ["cup"]
    assert proposals[0][1].x1 == pytest.approx(50 / 640)
