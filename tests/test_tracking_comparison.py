import copy
import runpy
import sys
from pathlib import Path

import pytest


def test_identical_prediction_gate_catches_changes_that_accuracy_hides(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts" / "eval"
    monkeypatch.setattr(sys, "path", [str(scripts), *sys.path])
    compare = runpy.run_path(str(scripts / "compare_tracking_reports.py"))["compare"]
    box = dict(x1=0.1, y1=0.1, x2=0.5, y2=0.5)
    report = dict(
        status="complete",
        failed_clips=0,
        clips=[
            dict(
                sequence="sample",
                status="complete",
                video_sha256="a",
                labels_sha256="b",
                index_seconds=1.0,
                rows=[
                    dict(
                        at_ms=0,
                        truth_visible=True,
                        truth_box=box,
                        truth_zone="left",
                        predicted_visible=True,
                        predicted_box=box.copy(),
                        predicted_zone="left",
                        iou=1.0,
                        reason=None,
                        detector_score=0.99,
                        identity_score=0.9,
                    )
                ],
            )
        ],
    )
    paired = compare(report, copy.deepcopy(report), require_identical_predictions=True)
    assert paired["identical_prediction_rows"] == 1 and paired["changed_prediction_rows"] == 0
    for field, value in [("predicted_box", {**box, "x1": 0.11}), ("detector_score", 0.98)]:
        candidate = copy.deepcopy(report)
        candidate["clips"][0]["rows"][0][field] = value
        permissive = compare(report, candidate)
        assert permissive["clips"][0]["lost_correct"] == 0
        assert permissive["changed_prediction_rows"] == 1
        with pytest.raises(ValueError, match="1 prediction rows differ"):
            compare(report, candidate, require_identical_predictions=True)
    incomplete = copy.deepcopy(report)
    del incomplete["clips"][0]["rows"][0]["predicted_box"]
    with pytest.raises(ValueError, match="every prediction field"):
        compare(incomplete, incomplete, require_identical_predictions=True)
