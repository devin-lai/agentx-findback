"""Deterministic temporal reduction of per-frame model localizations.

A vision-language model answers one bounded question per frame ("is the target visible, and
where?"). Ordering, last-visible logic and zone naming stay in code, so the model's known
weakness at multi-frame temporal reasoning cannot decide the answer.
"""

from agentx.domain.contracts import Box, region_for, timestamp


def zone_for_point(x: float, y: float, regions: list[dict]) -> str | None:
    half = 0.004
    box = Box(
        x1=min(max(x - half, 0), 1 - 2 * half),
        y1=min(max(y - half, 0), 1 - 2 * half),
        x2=max(min(x + half, 1), 2 * half),
        y2=max(min(y + half, 1), 2 * half),
    )
    return region_for(box, regions)


def camera_at(at_ms: int, poses: list[dict], tolerance_ms: int) -> dict | None:
    """The recorded camera pose nearest a reviewed frame, or None when none is close enough.

    Indexing already estimated the camera for every sampled frame, with the registered objects
    masked out and the full forward history available. A review samples the same recording, so it
    reads those poses instead of estimating its own from eight frames — which is also the only way
    a review and the memory it accompanies can be guaranteed to name the same place.
    """
    if not poses:
        return None
    nearest = min(poses, key=lambda p: abs(p["at_ms"] - at_ms))
    return nearest if abs(nearest["at_ms"] - at_ms) <= tolerance_ms else None


def focus_window(
    x: float, y: float, width: int, height: int, fraction: float
) -> tuple[int, int, int, int]:
    """A square pixel crop centred on a normalized point and clamped inside the frame.

    The side is a fraction of the shorter frame edge, so the same setting means the same
    physical field of view for portrait and landscape recordings.
    """
    side = max(16, round(fraction * min(width, height)))
    side = min(side, width, height)
    left = round(x * width) - side // 2
    top = round(y * height) - side // 2
    left = min(max(left, 0), width - side)
    top = min(max(top, 0), height - side)
    return left, top, left + side, top + side


def focus_point(
    x: float, y: float, window: tuple[int, int, int, int], width: int, height: int
) -> tuple[float, float]:
    """Map a normalized point measured inside a crop back into frame coordinates."""
    left, top, right, bottom = window
    return (
        min(max((left + x * (right - left)) / width, 0.0), 1.0),
        min(max((top + y * (bottom - top)) / height, 0.0), 1.0),
    )


def reduce_frame_reports(target: str, frames: list[dict], regions: list[dict]) -> dict:
    """frames: ordered dicts with id, at_ms, present, x, y, note (model output per frame).

    An entry may also carry `camera` with `scene_x`/`scene_y`: the same point read in
    registration coordinates after the camera moved. The published x and y always stay in the
    frame the model saw, because that is the image the user replays.
    """
    if not frames:
        raise ValueError("No frame reports to reduce.")
    names = {r["id"]: r["name"] for r in regions}
    rows = []
    for f in frames:
        # A region is a place in the registration frame. When the camera has moved, the point
        # has to be read there; when nothing relates the view to it, no region can be named.
        camera = f.get("camera")
        x, y = f.get("x"), f.get("y")
        if camera == "compensated":
            x, y = f.get("scene_x"), f.get("scene_y")
        elif camera == "unavailable":
            x, y = None, None
        zone = None
        if f["present"] and x is not None and y is not None:
            zone = zone_for_point(x, y, regions)
        row = {
            "id": f["id"],
            "at_ms": f["at_ms"],
            "present": bool(f["present"]),
            "x": f.get("x"),
            "y": f.get("y"),
            "zone": zone,
            "zone_name": names.get(zone) if zone else None,
            "camera": camera,
            "note": (f.get("note") or "")[:300],
            "focus": f.get("focus"),
        }
        if "presence_veto" in f:
            row["presence_veto"] = f["presence_veto"]
        rows.append(row)
    present = [r for r in rows if r["present"]]
    first, final = rows[0], rows[-1]
    if not present:
        summary = (
            f"{target} is not visible in any of the {len(rows)} supplied frames between "
            f"{timestamp(first['at_ms'])} and {timestamp(final['at_ms'])}."
        )
        cited = [final["id"]]
    else:
        last = present[-1]
        if last["zone_name"]:
            where = f"in {last['zone_name']}"
        elif last["camera"] == "unavailable":
            # The camera moved and the frame could not be related to the registration view. That
            # is not "somewhere else"; it is a place this recording can no longer name.
            where = "at a position the camera move leaves unnamed"
        else:
            where = "in an unassigned area"
        summary = (
            f"{target} was last visible {where} at {timestamp(last['at_ms'])} (frame {last['id']})."
        )
        later = [r for r in rows if r["id"] > last["id"]]
        if later:
            summary += (
                f" It is not visible in the {len(later)} later frame(s) from "
                f"{timestamp(later[0]['at_ms'])} to {timestamp(later[-1]['at_ms'])}."
            )
        else:
            summary += " It is visible in the final supplied frame."
        path: list[tuple[int, str, int]] = []
        for r in present:
            if r["camera"] == "unavailable":
                continue  # A frame that names no region cannot be a step in a path of regions.
            zone_name = r["zone_name"] or "an unassigned area"
            if not path or path[-1][1] != zone_name:
                path.append((r["at_ms"], zone_name, r["id"]))
        if len(path) > 1:
            summary += (
                " Zones over time: " + " → ".join(f"{timestamp(t)} {z}" for t, z, _ in path) + "."
            )
        unnamed = sum(r["camera"] == "unavailable" for r in present)
        if unnamed:
            summary += (
                f" The camera moved during {unnamed} of the visible frame(s), which could not be "
                "related to the registered view, so those frames name no area."
            )
        cited = [last["id"]] + ([later[0]["id"]] if later else []) + [i for _, _, i in path]
    unique = list(dict.fromkeys(cited))[:8]
    uncertainty = (
        "Each frame was judged independently by the model at one frame per second; moments between "
        "samples and after the cutoff are not observed. Absence in a frame may be occlusion, leaving "
        "the view or a model miss, and never shows a destination."
    )
    return {
        "summary": summary,
        "evidence_frame_ids": unique,
        "uncertainty": uncertainty,
        "frame_reports": rows,
    }
