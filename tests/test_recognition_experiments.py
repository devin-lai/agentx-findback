"""Protect paired accuracy experiments from silently incomparable denominators."""

import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def compare(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts" / "eval"))
    return importlib.import_module("compare_tracking_cadence").compare


def paired():
    row = dict(
        at_ms=0,
        truth_visible=True,
        predicted_visible=True,
        iou=0.8,
        truth_zone="left",
        predicted_zone="left",
        reason=None,
    )
    clip = dict(
        sequence="test",
        tracking_fps=5,
        video_sha256="video",
        labels_sha256="labels",
        rows=[row],
        seconds=1,
        inferred_frames=1,
    )
    second = copy.deepcopy(clip)
    second.update(tracking_fps=15, inferred_frames=3)
    return dict(status="complete", evidence_fps=5, clips=[clip, second])


def test_cadence_comparison_keeps_the_evidence_denominator(compare):
    result = compare(paired())
    for variant in result["variants"]:
        assert variant["summary"]["samples"] == 1
        assert variant["summary"]["recall_at_iou_0_5"] == 1
    assert result["variants"][1]["inferred_frames"] == 3


@pytest.mark.parametrize(
    "field,value", [("video_sha256", "other"), ("labels_sha256", "other"), ("sequence", "missing")]
)
def test_cadence_comparison_rejects_changed_inputs(compare, field, value):
    data = paired()
    data["clips"][1][field] = value
    with pytest.raises(ValueError):
        compare(data)


def test_cadence_comparison_rejects_different_sample_times_and_partial_reports(compare):
    data = paired()
    data["clips"][1]["rows"][0]["at_ms"] = 67
    with pytest.raises(ValueError, match="timestamps"):
        compare(data)
    data = paired()
    data["status"] = "running"
    with pytest.raises(ValueError, match="not complete"):
        compare(data)


def test_model_manifest_rejects_unpinned_or_executable_inputs(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/spark"))
    validate = importlib.import_module("download_verified_snapshot").validate_manifest
    manifest = dict(
        repository="facebook/model",
        revision="a" * 40,
        files={"model.safetensors": "b" * 64, "config.json": "c" * 64},
    )
    validate(manifest)
    for field, value in [
        ("revision", "main"),
        ("repository", "../escape"),
        ("files", {"../config.json": "a" * 64}),
        ("files", {"loader.py": "a" * 64}),
    ]:
        bad = copy.deepcopy(manifest)
        bad[field] = value
        with pytest.raises(ValueError):
            validate(bad)
