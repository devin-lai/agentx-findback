import importlib.util
from pathlib import Path

import pytest

from agentx.domain.contracts import Box, default_regions, region_for

spec = importlib.util.spec_from_file_location(
    "prepare_tracking_benchmark",
    Path(__file__).parents[1] / "scripts/data/prepare_tracking_benchmark.py",
)
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


def test_tiny_external_box_preserves_geometry_without_relaxing_registration():
    box = preparation.normalized_box([354, 332, 1, 191], 1280, 720)
    assert box.x2 - box.x1 == pytest.approx(1 / 1280)
    assert box.y2 - box.y1 == pytest.approx(191 / 720)
    assert region_for(box, default_regions()) == "left"
    with pytest.raises(ValueError, match="non-empty"):
        Box.model_validate(box.model_dump())


@pytest.mark.parametrize("raw", [[10, 10, 0, 20], [10, 10, -1, 20], [2000, 10, 10, 20]])
def test_invalid_clipped_external_boxes_are_not_fabricated(raw):
    with pytest.raises(ValueError, match="positive clipped area"):
        preparation.normalized_box(raw, 1280, 720)


def test_ordinary_external_box_coordinates_are_unchanged():
    raw = [100, 100, 100, 100]
    box = preparation.normalized_box(raw, 1000, 1000)
    assert box.model_dump() == Box(x1=0.1, y1=0.1, x2=0.2, y2=0.2).model_dump()
