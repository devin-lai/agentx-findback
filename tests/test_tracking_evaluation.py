import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "evaluate_tracking", Path(__file__).resolve().parents[1] / "scripts/eval/evaluate_tracking.py"
)
tracking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracking)


def test_tracking_score_counts_abstentions_and_wrong_objects():
    rows = [
        {
            "truth_visible": True,
            "predicted_visible": True,
            "iou": 0.8,
            "truth_zone": "left",
            "predicted_zone": "left",
            "reason": None,
        },
        {
            "truth_visible": True,
            "predicted_visible": True,
            "iou": 0,
            "truth_zone": "left",
            "predicted_zone": "left",
            "reason": None,
        },
        {
            "truth_visible": True,
            "predicted_visible": False,
            "iou": 0,
            "truth_zone": "right",
            "predicted_zone": None,
            "reason": "scene_changed",
        },
        {
            "truth_visible": False,
            "predicted_visible": True,
            "iou": 0,
            "truth_zone": None,
            "predicted_zone": "left",
            "reason": None,
        },
    ]
    score = tracking.summarize(rows)
    assert score["recall_at_iou_0_5"] == pytest.approx(1 / 3)
    assert score["precision_at_iou_0_5"] == pytest.approx(1 / 3)
    assert score["false_visible_when_absent"] == 1
    assert score["poor_overlap_when_visible"] == 1
    assert score["abstention_reasons"] == {"scene_changed": 1}
    # Same-region boxes do not count as successful object localization.
    assert score["zone_correct_and_visible"] == 2 and score["correct_at_iou_0_5"] == 1


def test_iou_and_empty_denominators():
    box = {"x1": 0, "y1": 0, "x2": 0.5, "y2": 0.5}
    other = {"x1": 0.25, "y1": 0.25, "x2": 0.75, "y2": 0.75}
    assert tracking.intersection_over_union(box, other) == pytest.approx(1 / 7)
    assert tracking.intersection_over_union(box, None) == 0
    assert tracking.summarize([])["recall_at_iou_0_5"] is None
