import cv2
import numpy as np
import pytest

from agentx.domain.contracts import Box, Detection, default_regions, zone_of
from agentx.vision.scene import SceneTracker, _as_3x3, project_box


def textured(seed: int = 47, size: tuple[int, int] = (480, 640)) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (*size, 3), dtype=np.uint8)


def test_moving_local_texture_cannot_invalidate_the_entire_scene():
    image = np.full((720, 1280, 3), 160, np.uint8)
    patch = np.random.default_rng(31).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    image[100:200, 100:200] = patch
    current = np.full_like(image, 160)
    current[100:200, 240:340] = patch
    guard = SceneTracker(image, [])
    assert len(guard.descriptors) >= 25  # Enough local features to fool a naive rule.
    assert not guard.check(current).changed
    assert not guard.check(image).changed


@pytest.mark.parametrize("transform", ["translation", "rotation", "scale"])
def test_distributed_camera_motion_is_detected_and_recovered(transform):
    image = textured()
    if transform == "translation":
        matrix = np.array([[1, 0, 85], [0, 1, 0]], np.float32)
    else:
        matrix = cv2.getRotationMatrix2D(
            (320, 240), 12 if transform == "rotation" else 0, 1.2 if transform == "scale" else 1.0
        )
    shifted = cv2.warpAffine(image, matrix, (640, 480))
    guard = SceneTracker(image, [])
    state = guard.check(shifted)
    assert state.changed and state.recoverable and state.compensated
    assert not state.blocked
    # The estimate recovers the applied transform rather than only flagging it.
    assert np.allclose(state.matrix, _as_3x3(matrix), atol=1.0)


def test_a_camera_that_returns_to_its_registered_pose_is_stable_again():
    image = textured()
    moved = cv2.warpAffine(image, np.array([[1, 0, 85], [0, 1, 0]], np.float32), (640, 480))
    guard = SceneTracker(image, [])
    assert guard.check(moved).changed
    # Earlier releases latched permanently and discarded the rest of the recording.
    assert not guard.check(image).changed


def test_a_moved_camera_with_no_supported_transform_blocks_the_scene():
    image = textured()
    moved = cv2.warpAffine(image, np.array([[1, 0, 85], [0, 1, 0]], np.float32), (640, 480))
    guard = SceneTracker(image, [])
    assert guard.check(moved).compensated
    unrelated = textured(seed=99)
    state = guard.check(unrelated)
    assert state.blocked and not state.recoverable


def test_an_unmatched_frame_before_any_motion_asserts_nothing():
    guard = SceneTracker(textured(), [])
    state = guard.check(textured(seed=99))
    assert not state.changed and not state.blocked


def test_compensated_positions_are_read_in_registration_coordinates():
    image = textured()
    shift = np.array([[1, 0, 85], [0, 1, 0]], np.float32)
    guard = SceneTracker(image, [])
    state = guard.check(cv2.warpAffine(image, shift, (640, 480)))
    inverse = np.linalg.inv(state.matrix)
    # A desk object that sat on the left of the registered frame stays on the left after the
    # camera pans right, even though its pixels moved to the centre.
    moved_box = Box(x1=0.35, y1=0.4, x2=0.45, y2=0.6)
    scene_box = project_box(moved_box, inverse, 640, 480)
    regions = default_regions()
    assert zone_of(Detection("o", moved_box, 0.9), regions) == "center"
    assert (
        zone_of(
            Detection("o", moved_box, 0.9, scene_box=scene_box, scene_reference="compensated"),
            regions,
        )
        == "left"
    )


def test_a_position_outside_the_registered_view_has_no_region():
    matrix = np.array([[1.0, 0.0, -600.0], [0.0, 1.0, 0.0]])
    assert project_box(Box(x1=0.2, y1=0.4, x2=0.3, y2=0.6), matrix, 640, 480) is None
    assert (
        zone_of(
            Detection("o", Box(x1=0.2, y1=0.4, x2=0.3, y2=0.6), 0.9, scene_reference="compensated"),
            default_regions(),
        )
        is None
    )


def test_almost_collinear_matches_do_not_establish_a_global_transform():
    guard = SceneTracker(np.zeros((480, 640, 3), np.uint8), [])
    points = np.array(
        [[x, x * 0.5 + (i % 2)] for i, x in enumerate(range(30, 600, 10))], np.float32
    )
    assert not guard._distributed(points)


def test_the_run_records_what_the_camera_evidence_showed():
    image = textured()
    guard = SceneTracker(image, [])
    guard.check(image)
    guard.check(cv2.warpAffine(image, np.array([[1, 0, 85], [0, 1, 0]], np.float32), (640, 480)))
    summary = guard.summary()
    assert summary["frames"]["stable"] == 1 and summary["frames"]["compensated"] == 1
    assert summary["peak_motion"] > 0.12


def test_an_unrelatable_view_keeps_the_evidence_and_drops_only_the_region():
    """A camera the tracker cannot relate to the registration frame still produces evidence."""
    box = Box(x1=0.05, y1=0.4, x2=0.15, y2=0.6)
    detection = Detection("remote", box, 0.91, scene_reference="unavailable")
    assert zone_of(detection, default_regions()) is None
    # The box itself is untouched: it is measured in the frame the user replays.
    assert detection.box == box and detection.reason is None


def test_a_moving_foreground_over_a_static_camera_asserts_no_motion():
    """The measured failure of a richer model: on a static overhead clip the operator's hands
    supply most of the matched features, and a homography happily fits *them*. The deployed
    partial affine reports no camera assertion instead of inventing one."""
    table = textured(size=(480, 720))
    hands = np.random.default_rng(5).integers(0, 256, (260, 380, 3), dtype=np.uint8)
    first, second = table.copy(), table.copy()
    first[110:370, 150:530] = hands
    second[110:370, 260:640] = hands  # only the foreground moved
    guard = SceneTracker(first, [])
    state = guard.check(second)
    assert not state.changed and not state.blocked
    assert guard.summary()["peak_motion"] <= SceneTracker.policy["motion_threshold"]


def test_a_folded_or_collapsed_fit_is_not_a_camera():
    from agentx.vision.scene import _plausible

    assert _plausible(np.eye(3), 640, 480)
    mirrored = np.array([[-1.0, 0, 640], [0, 1.0, 0], [0, 0, 1.0]])
    assert not _plausible(mirrored, 640, 480)  # A camera cannot mirror its own frame.
    collapsed = np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 1.0]])
    assert not _plausible(collapsed, 640, 480)


def test_a_stored_transform_is_resolution_independent():
    """A client maps the drawn regions into a later frame without knowing the source size."""
    from agentx.vision.scene import normalized_transform

    # A pan of a quarter of the frame width, in pixels, on a 1280x720 source.
    pixels = np.array([[1.0, 0.0, 320.0], [0.0, 1.0, -72.0], [0.0, 0.0, 1.0]])
    matrix = np.asarray(normalized_transform(pixels, 1280, 720))
    assert matrix[0][2] == pytest.approx(0.25)
    assert matrix[1][2] == pytest.approx(-0.1)
    # The same physical pan on a different resolution yields the same normalized transform.
    other = np.asarray(
        normalized_transform(
            np.array([[1.0, 0.0, 160.0], [0.0, 1.0, -36.0], [0.0, 0.0, 1.0]]), 640, 360
        )
    )
    assert np.allclose(matrix, other)


def test_a_region_follows_the_desk_through_the_knock():
    from agentx.domain.contracts import Region
    from agentx.vision.scene import normalized_transform

    image = textured(size=(360, 640))
    shift = np.array([[1, 0, 150], [0, 1, 0]], np.float32)
    guard = SceneTracker(image, [])
    state = guard.check(cv2.warpAffine(image, shift, (640, 360)))
    matrix = np.asarray(normalized_transform(state.matrix, 640, 360))
    centre = Region.model_validate(default_regions()[1]).box
    left = (matrix @ [centre.x1, 0.5, 1.0])[0]
    # The centre region's left edge was at one third; the camera panned it a quarter right.
    assert left == pytest.approx(1 / 3 + 150 / 640, abs=0.01)
