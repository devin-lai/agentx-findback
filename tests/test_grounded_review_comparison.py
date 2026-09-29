import copy
import importlib.util
import sys
from pathlib import Path

import pytest

scripts = Path(__file__).parents[1] / "scripts" / "eval"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    "compare_grounded_reviews", scripts / "compare_grounded_reviews.py"
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
sys.path.pop(0)


def report(cutoffs=(1000,)):
    windows = [
        {
            "sequence": "clip",
            "cutoff_ms": cutoff,
            "reference": True,
            "status": "complete",
            "elapsed_seconds": 1,
            "rows": [{"at_ms": cutoff, "truth_visible": True, "truth_box": [0, 0, 1, 1]}],
            "summary": {
                key: int(
                    key
                    not in (
                        "truth_absent",
                        "false_visible_when_absent",
                        "outside_target_box_when_visible",
                    )
                )
                for key in comparison.aggregate.__globals__["COUNTS"]
            },
        }
        for cutoff in cutoffs
    ]
    return {
        "status": "complete",
        "windows": windows,
        "scope": "test",
        "model": "test",
        "protocol_sha256": "test",
        "unique_source_frames": len(windows),
        "protocol": {
            "windows": windows,
            "inputs": {"clip": {"video_sha256": "video", "labels_sha256": "labels"}},
        },
    }


def test_subset_requires_explicit_choice_and_reports_exclusions():
    reports = {"short": report(), "long": report((1000, 2000))}
    with pytest.raises(ValueError, match="subset"):
        comparison.compare_reports(reports)
    result = comparison.compare_reports(reports, allow_subset=True)
    assert result["unique_source_frames"] == 1
    assert result["variants"]["long"]["excluded_windows"] == [["clip", 2000, True]]
    assert result["variants"]["long"]["metrics"]["truth_visible"] == 1


@pytest.mark.parametrize(
    "mutation", ["source", "labels", "time", "truth", "box", "partial", "duplicate"]
)
def test_incompatible_or_incomplete_inputs_are_rejected(mutation):
    left = report()
    right = copy.deepcopy(left)
    if mutation in ("source", "labels"):
        key = "video_sha256" if mutation == "source" else "labels_sha256"
        right["protocol"]["inputs"]["clip"][key] = "changed"
    elif mutation == "partial":
        right["status"] = "running"
    elif mutation == "duplicate":
        right["windows"].append(copy.deepcopy(right["windows"][0]))
    else:
        key, value = {
            "time": ("at_ms", 2000),
            "truth": ("truth_visible", False),
            "box": ("truth_box", None),
        }[mutation]
        right["windows"][0]["rows"][0][key] = value
    with pytest.raises(ValueError):
        comparison.compare_reports({"left": left, "right": right})
