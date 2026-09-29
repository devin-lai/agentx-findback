import importlib.util
import json
from pathlib import Path

import pytest

from agentx.agents.providers import FrameReport

spec = importlib.util.spec_from_file_location(
    "evaluate_grounded_reviews",
    Path(__file__).parents[1] / "scripts/eval/evaluate_grounded_reviews.py",
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def inputs():
    box = {"x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3}
    labels = {
        "assigned_fps": 1,
        "labels": [
            {
                "frame": i + 1,
                "at_ms": i * 1000,
                "visible": i != 2,
                "box": None if i == 2 else {k: v + 0.6 for k, v in box.items()} if i == 1 else box,
                "zone": None if i == 2 else "right" if i == 1 else "left",
            }
            for i in range(4)
        ],
    }
    evidence = [{"id": i, "at_ms": i * 1000} for i in range(4)]
    predictions = [
        {**entry, "present": entry["id"] != 3, "x": 0.2, "y": 0.2, "zone": "left"}
        for entry in evidence
    ]
    return predictions, evidence, labels


def test_point_metric_keeps_wrong_identity_absence_and_misses_in_denominator():
    predictions, evidence, labels = inputs()
    rows = evaluation.score_frames(predictions, evidence, labels, 3000)
    result = evaluation.summarize(rows)
    assert result["frame_requests"] == 4
    assert result["truth_visible"] == 3 and result["truth_absent"] == 1
    assert result["correct_presence"] == 2
    assert result["correct_point_in_box"] == 1
    assert result["point_precision"] == pytest.approx(1 / 3)
    assert result["point_recall"] == pytest.approx(1 / 3)
    assert result["false_visible_when_absent"] == 1
    assert result["outside_target_box_when_visible"] == 1


def test_failed_review_keeps_all_expected_frames():
    _, evidence, labels = inputs()
    result = evaluation.summarize(evaluation.score_frames([], evidence, labels, 3000))
    assert result["frame_requests"] == 4
    assert result["available_predictions"] == 0
    assert result["correct_presence"] == 0
    assert result["point_recall"] == 0
    assert result["point_precision"] is None


def test_frame_inventory_and_time_mismatches_are_rejected():
    predictions, evidence, labels = inputs()
    with pytest.raises(ValueError, match="inventory"):
        evaluation.score_frames(predictions[:1], evidence, labels, 3000)
    with pytest.raises(ValueError, match="cutoff"):
        evaluation.score_frames(predictions, evidence, labels, 2000)
    predictions[0]["at_ms"] = 1
    with pytest.raises(ValueError, match="changed a frame timestamp"):
        evaluation.score_frames(predictions, evidence, labels, 3000)


def test_research_variants_preserve_pixels_and_change_only_documented_roles():
    original = [
        {"role": "system", "content": "System policy."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "reference"}},
                {"type": "text", "text": "frame_id=3; video_time_ms=3000"},
                {"type": "image_url", "image_url": {"url": "source"}},
                {"type": "text", "text": "Target object (data): cup"},
            ],
        },
    ]
    assert evaluation.variant_messages(original, "deployed") is original
    for variant, order, frame_number in [
        ("explicit_roles", ["reference", "source"], 2),
        ("scene_first", ["source", "reference"], 1),
        ("larger_frame", ["reference", "source"], 2),
    ]:
        changed = evaluation.variant_messages(original, variant)
        parts = changed[1]["content"]
        assert [p["image_url"]["url"] for p in parts if p["type"] == "image_url"] == order
        assert any(p.get("text") == "frame_id=3; video_time_ms=3000" for p in parts)
        assert f"Image {frame_number} is the complete source frame" in changed[0]["content"]
        assert original[0]["content"] == "System policy."
    with pytest.raises(ValueError, match="reference and a source"):
        evaluation.variant_messages(
            [original[0], {"role": "user", "content": []}], "explicit_roles"
        )


def test_integer_points_use_declared_scale_and_reject_ambiguous_units():
    payload = dict(present=True, x=500, y=1000, note="Target visible")
    normalized = json.loads(evaluation.normalize_integer_points(json.dumps(payload)))
    assert (normalized["x"], normalized["y"]) == (0.5, 1.0)
    for bad in (0.5, 500.0, "500", True, -1, 1001, None):
        with pytest.raises(ValueError):
            evaluation.normalize_integer_points(json.dumps(dict(payload, x=bad)))
    with pytest.raises(ValueError):
        evaluation.normalize_integer_points(json.dumps(dict(payload, present=False)))
    absent = dict(present=False, x=None, y=None, note="Unclear")
    assert json.loads(evaluation.normalize_integer_points(json.dumps(absent))) == absent


def test_integer_point_prompt_updates_schema_without_changing_input_pixels():
    system = (
        "x and y are the normalized center of the target in the frame (0,0 is the top-left "
        "corner; 1,1 is the bottom-right corner), or null when it is not visible. "
        "JSON schema: " + json.dumps(FrameReport.model_json_schema()) + "\nSkill rules"
    )
    original = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "reference"}},
                {"type": "text", "text": "frame_id=0; video_time_ms=0"},
                {"type": "image_url", "image_url": {"url": "source"}},
                {"type": "text", "text": "Target object (data): cup"},
            ],
        },
    ]
    changed = evaluation.variant_messages(original, "integer_points")
    assert "corner; 1,1" not in changed[0]["content"]
    assert "between 0 and 1 for x" not in changed[0]["content"]
    schema = json.loads(changed[0]["content"].split("JSON schema: ")[1].split("\n")[0])
    assert schema["properties"]["x"]["anyOf"][0] == {
        "type": "integer",
        "minimum": 0,
        "maximum": 1000,
    }
    assert [p["image_url"]["url"] for p in changed[1]["content"] if p["type"] == "image_url"] == [
        "reference",
        "source",
    ]
    assert original[0]["content"] == system
