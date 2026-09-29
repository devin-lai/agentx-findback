"""Exact mask geometry with compact device-to-host transfers for CUDA tracking."""

from itertools import combinations

import numpy as np

from agentx.domain.contracts import Box


def mask_box(mask: np.ndarray) -> Box | None:
    """Bound every positive pixel, retaining the domain's minimum box size."""
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    h, w = mask.shape
    return _box((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), h, w)


def _box(bounds, height: int, width: int) -> Box | None:
    try:
        return Box(
            x1=float(bounds[0]) / width,
            y1=float(bounds[1]) / height,
            x2=float(bounds[2]) / width,
            y2=float(bounds[3]) / height,
        )
    except ValueError:
        return None


def mask_geometry(masks) -> tuple[list[Box | None], set[tuple[int, int]]]:
    """Boxes and pairs with mask IoU >= 0.85; input is N x H x W bool.

    CUDA reduces masks on device and transfers only integer bounds and pair
    decisions. CPU inference retains the NumPy reference calculation. Presence
    thresholds and retired companion tracks are applied by the caller.
    """
    if masks.device.type == "cuda":
        return _tensor_geometry(masks)
    arrays = masks.cpu().numpy()
    boxes = [mask_box(mask) for mask in arrays]
    overlaps = set()
    for i, j in combinations(range(len(arrays)), 2):
        union = np.count_nonzero(arrays[i] | arrays[j])
        if union and np.count_nonzero(arrays[i] & arrays[j]) / union >= 0.85:
            overlaps.add((i, j))
    return boxes, overlaps


def _tensor_geometry(masks) -> tuple[list[Box | None], set[tuple[int, int]]]:
    import torch

    count, height, width = masks.shape
    if not count:
        return [], set()
    columns, rows = masks.any(dim=1), masks.any(dim=2)
    x = torch.arange(width, device=masks.device)
    y = torch.arange(height, device=masks.device)
    bounds = torch.stack(
        (
            torch.where(columns, x, width).amin(dim=1),
            torch.where(rows, y, height).amin(dim=1),
            torch.where(columns, x + 1, 0).amax(dim=1),
            torch.where(rows, y + 1, 0).amax(dim=1),
        ),
        dim=1,
    )
    pairs = list(combinations(range(count), 2))
    decisions = []
    if pairs:
        areas = masks.sum(dim=(1, 2), dtype=torch.int64)
        # One pair at a time bounds temporary memory by one source-size mask,
        # rather than materializing N x N x H x W masks. No scalar host reads.
        for i, j in pairs:
            intersection = (masks[i] & masks[j]).sum(dtype=torch.int64)
            union = areas[i] + areas[j] - intersection
            decisions.append((union > 0) & (intersection * 100 >= union * 85))
    packed = (
        torch.cat(
            (bounds.flatten(), torch.stack(decisions).to(torch.int64))
            if decisions
            else (bounds.flatten(),)
        )
        .cpu()
        .tolist()
    )
    boxes = [_box(packed[i * 4 : (i + 1) * 4], height, width) for i in range(count)]
    overlaps = {pair for pair, present in zip(pairs, packed[count * 4 :], strict=True) if present}
    return boxes, overlaps
