import pytest
from pydantic import ValidationError

from agentx.domain.contracts import Box, Detection, default_regions, region_for
from agentx.domain.temporal import EventReducer

LEFT = Box(x1=0.1, y1=0.2, x2=0.2, y2=0.3)
RIGHT = Box(x1=0.8, y1=0.2, x2=0.9, y2=0.3)


def test_changes_require_confirmation_and_visibility_is_never_predicted():
    memory = EventReducer(default_regions())
    assert memory.update(0, [Detection("key", LEFT, 0.9)])[0].kind == "appeared"
    assert memory.update(200, [Detection("key", RIGHT, 0.9)]) == []
    change = memory.update(400, [Detection("key", RIGHT, 0.9)])[0]
    assert (change.kind, change.previous_zone, change.zone) == ("moved", "left", "right")
    lost = memory.update(600, [Detection("key", None, None, "not_detected")])[0]
    assert lost.kind == "lost"
    assert lost.zone is None
    assert not memory.objects["key"].visible
    assert memory.update(800, [Detection("key", None, None, "not_detected")]) == []
    assert memory.update(1000, [Detection("key", LEFT, 0.9)])[0].kind == "reappeared"
    with pytest.raises(ValueError, match="strictly increasing"):
        memory.update(999, [])


def test_jitter_does_not_become_movement_and_ambiguity_is_explicit():
    memory = EventReducer(default_regions())
    memory.update(0, [Detection("key", LEFT, 0.9)])
    assert memory.update(200, [Detection("key", RIGHT, 0.9)]) == []
    assert memory.update(400, [Detection("key", LEFT, 0.9)]) == []
    assert memory.update(600, [Detection("key", RIGHT, 0.9)]) == []
    ambiguous = memory.update(800, [Detection("key", None, None, "identity_ambiguous")])[0]
    assert ambiguous.kind == "ambiguous"
    assert not memory.objects["key"].visible


@pytest.mark.parametrize(
    "coordinates",
    [
        (-0.1, 0, 0.5, 0.5),
        (0, 0, 1.1, 0.5),
        (0.5, 0.5, 0.4, 0.6),
        (float("nan"), 0, 0.5, 0.5),
        (0, 0, float("inf"), 0.5),
    ],
)
def test_invalid_boxes_are_rejected(coordinates):
    with pytest.raises(ValidationError):
        Box(**dict(zip(("x1", "y1", "x2", "y2"), coordinates, strict=True)))


def test_region_boundaries_are_unique_and_overlap_is_unknown():
    box = Box(x1=1 / 3 - 0.05, y1=0.2, x2=1 / 3 + 0.05, y2=0.3)
    assert region_for(box, default_regions()) == "center"
    regions = default_regions()
    regions[0]["box"]["x2"] = 0.7
    assert region_for(box, regions) is None


def test_a_compensated_camera_move_keeps_a_region_anchored_to_the_registration_frame():
    """The user drew "left area" on the registration frame; a camera pan must not rename it."""
    import cv2
    import numpy as np

    from agentx.domain.contracts import zone_of
    from agentx.vision.scene import SceneTracker, project_box

    image = np.random.default_rng(12).integers(0, 256, (360, 640, 3), dtype=np.uint8)
    guard = SceneTracker(image, [])
    assert not guard.check(image).changed
    shifted = cv2.warpAffine(image, np.array([[1, 0, 95], [0, 1, 0]], dtype=np.float32), (640, 360))
    state = guard.check(shifted)
    assert state.compensated

    regions = default_regions()
    reducer = EventReducer(regions)
    inverse = np.linalg.inv(state.matrix)
    resting = Box(x1=0.20, y1=0.4, x2=0.30, y2=0.6)
    reducer.update(0, [Detection("remote", resting, 0.9)])

    # The same physical spot after the pan: its pixels moved right, its scene position did not.
    panned = Box(x1=0.20 + 95 / 640, y1=0.4, x2=0.30 + 95 / 640, y2=0.6)
    detection = Detection(
        "remote",
        panned,
        0.9,
        scene_box=project_box(panned, inverse, 640, 360),
        scene_reference="compensated",
    )
    assert zone_of(detection, regions) == "left"
    assert reducer.update(1000, [detection]) == []  # No invented movement event.


def test_an_unnameable_region_is_a_gap_in_knowledge_not_a_movement():
    """A camera the tracker cannot relate to the registration view says nothing about where the
    object is. Treating that silence as a new zone invents a movement for a resting object."""
    from agentx.domain.contracts import zone_of

    regions = default_regions()
    reducer = EventReducer(regions)
    resting = Box(x1=0.4, y1=0.4, x2=0.5, y2=0.6)
    assert reducer.update(0, [Detection("remote", resting, 0.9)])[0].zone == "center"

    blind = Detection("remote", resting, 0.9, scene_reference="unavailable")
    assert zone_of(blind, regions) is None and not blind.region_supported
    for at_ms in (1000, 2000, 3000, 4000):
        assert reducer.update(at_ms, [blind]) == []
    # And nothing is invented on the way back, because the object never left the centre.
    assert reducer.update(5000, [Detection("remote", resting, 0.9)]) == []


def test_a_compensated_position_outside_the_registered_view_is_also_not_a_movement():
    reducer = EventReducer(default_regions())
    resting = Box(x1=0.4, y1=0.4, x2=0.5, y2=0.6)
    reducer.update(0, [Detection("remote", resting, 0.9)])
    # The camera panned so far that the object's registered position is off the original frame.
    outside = Detection("remote", resting, 0.9, scene_box=None, scene_reference="compensated")
    assert not outside.region_supported
    assert reducer.update(1000, [outside]) == []
    assert reducer.update(2000, [outside]) == []


def test_an_object_first_seen_while_the_camera_is_unrelatable_appears_without_a_region():
    reducer = EventReducer(default_regions())
    blind = Detection(
        "remote", Box(x1=0.4, y1=0.4, x2=0.5, y2=0.6), 0.9, scene_reference="unavailable"
    )
    events = reducer.update(0, [blind])
    assert [(e.kind, e.zone) for e in events] == [("appeared", None)]
    assert reducer.update(1000, [blind]) == []


def test_a_reappearance_the_camera_cannot_place_claims_no_area():
    regions = default_regions()
    reducer = EventReducer(regions)
    left = Box(x1=0.05, y1=0.4, x2=0.15, y2=0.6)
    right = Box(x1=0.85, y1=0.4, x2=0.95, y2=0.6)
    reducer.update(0, [Detection("remote", left, 0.9)])
    reducer.update(200, [Detection("remote", None, 0.1, "not_detected")])
    blind = Detection("remote", right, 0.9, scene_reference="unavailable")
    events = reducer.update(400, [blind])
    assert [(e.kind, e.zone, e.previous_zone) for e in events] == [("reappeared", None, "left")]


def test_placing_an_object_first_seen_blind_is_not_a_movement():
    reducer = EventReducer(default_regions())
    left = Box(x1=0.05, y1=0.4, x2=0.15, y2=0.6)
    reducer.update(0, [Detection("remote", left, 0.9, scene_reference="unavailable")])
    assert reducer.update(200, [Detection("remote", left, 0.9)]) == []
    assert reducer.update(400, [Detection("remote", left, 0.9)]) == []
    right = Box(x1=0.85, y1=0.4, x2=0.95, y2=0.6)
    reducer.update(600, [Detection("remote", right, 0.9)])
    moved = reducer.update(800, [Detection("remote", right, 0.9)])
    assert [(e.kind, e.previous_zone, e.zone) for e in moved] == [("moved", "left", "right")]
