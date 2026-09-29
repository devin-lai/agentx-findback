"""The accelerated geometry must retain the original pixel and overlap contract."""

from itertools import combinations

import numpy as np
import pytest

from agentx.vision.masks import _tensor_geometry, mask_box, mask_geometry


def oracle(masks):
    overlaps = set()
    for i, j in combinations(range(len(masks)), 2):
        union = np.count_nonzero(masks[i] | masks[j])
        if union and np.count_nonzero(masks[i] & masks[j]) / union >= 0.85:
            overlaps.add((i, j))
    return [mask_box(mask) for mask in masks], overlaps


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("shape", [(0, 20, 30), (1, 51, 37), (6, 80, 97), (4, 1080, 1920)])
def test_geometry_matches_pixel_oracle(device, shape):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA parity runs on the Spark node.")
    masks = np.random.default_rng(21).random(shape) > 0.4
    if len(masks) >= 4:
        masks[1] = masks[0]
        masks[2] = False
        masks[3] = False
        masks[3, 0, 0] = masks[3, -1, -1] = True
    tensor = torch.from_numpy(masks).to(device)
    expected = oracle(masks)
    assert _tensor_geometry(tensor) == expected
    assert mask_geometry(tensor) == expected


def test_overlap_boundary_empty_and_subminimum_masks():
    torch = pytest.importorskip("torch")
    masks = np.zeros((5, 1000, 1000), dtype=bool)
    masks[0, :10, :10] = True
    masks[1, :8, :10] = masks[1, 8, :5] = True  # IoU exactly .85
    masks[2] = masks[1]
    masks[2, 8, 4] = False  # IoU .84
    masks[3, 500, 500] = True  # too small for a valid domain box
    boxes, overlaps = _tensor_geometry(torch.from_numpy(masks))
    assert (0, 1) in overlaps and (0, 2) not in overlaps
    assert boxes[3] is None and boxes[4] is None
    assert (boxes, overlaps) == oracle(masks)
