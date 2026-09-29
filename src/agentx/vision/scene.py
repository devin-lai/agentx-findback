"""Camera-pose tracking for a registered scene.

The registration frame defines the coordinate system a user's regions are drawn in. When the
camera is nudged, panned or zoomed, image coordinates stop meaning what they meant at
registration. Earlier releases only *detected* that situation and then abstained permanently,
which discarded every later observation in a clip whose camera moved once.

This module keeps the detection and adds recovery: when a large camera change is supported by a
distributed, low-residual transform, the transform itself is the missing information. Perception
continues, and the observation's position is additionally expressed in registration coordinates
so zones keep their original meaning. Abstention is reserved for the case the product genuinely
cannot answer: the camera is known to have moved and no supported transform relates the current
view to the registered scene.

The model is a partial affine: rotation, uniform scale and translation. A richer model was
measured and rejected. A homography does describe parallax, and it recovered two benchmark clips
whose camera genuinely translates through a cluttered room. On two clips filmed from a *static*
overhead camera it also fitted the operator's moving hands and asserted a camera motion of 0.30
and 0.53 that never happened, because the hands supply most of the matched features there. Extra
degrees of freedom bought coverage on real camera motion by inventing it elsewhere, so this
module keeps the model that cannot express what it cannot support. The measured camera
recovery result is summarized in `docs/benchmarks/INDEX.md`.

None of this is SLAM. A supported fit is not a metric pose and not proof of a rigid planar scene.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from agentx.domain.contracts import Box, pixel_box

MINIMUM_MATCHES = 25
MINIMUM_INLIER_FRACTION = 0.7
MINIMUM_HULL_FRACTION = 0.05
MINIMUM_AXIS_SPAN = 0.2
MOTION_THRESHOLD = 0.12
MAXIMUM_RESIDUAL = 0.02
MAXIMUM_RELAY_DEPTH = 64
# A fit that maps the frame to this far outside its own area is not something a camera did.
PLAUSIBLE_AREA_RANGE = (0.25, 4.0)
# The ratio test keeps only confidently unique matches. A second, looser pass rescues
# texture-poor frames where the strict test leaves too few correspondences to fit anything.
STRICT_RATIO = 0.7
LOOSE_RATIO = 0.82


def _identity() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def _as_3x3(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape == (3, 3):
        return matrix.astype(np.float64)
    return np.vstack([matrix, [0.0, 0.0, 1.0]]).astype(np.float64)


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map Nx2 pixel points through a 3x3 transform."""
    mapped = cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float32).reshape(-1, 1, 2), matrix.astype(np.float64)
    )
    return mapped.reshape(-1, 2)


def _plausible(matrix: np.ndarray, width: int, height: int) -> bool:
    """Reject a fit that maps the frame to something a camera could not have produced."""
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    mapped = _apply(matrix, corners)
    if not np.all(np.isfinite(mapped)):
        return False
    edges = np.roll(mapped, -1, axis=0) - mapped
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return False  # A camera cannot fold the frame over itself.
    signed = 0.5 * (
        float(np.dot(mapped[:, 0], np.roll(mapped[:, 1], -1)))
        - float(np.dot(np.roll(mapped[:, 0], -1), mapped[:, 1]))
    )
    if signed <= 0:
        return False  # The frame's own winding is positive; a negative one mirrors it.
    ratio = signed / (width * height)
    return PLAUSIBLE_AREA_RANGE[0] <= ratio <= PLAUSIBLE_AREA_RANGE[1]


@dataclass(frozen=True)
class SceneState:
    """What the camera evidence supports for one sampled frame."""

    changed: bool = False
    recoverable: bool = True
    matrix: np.ndarray | None = None
    support: dict = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        """The camera is known to have moved and nothing relates the view to the registration.

        A blocked frame still yields observations: the box is measured in an image the user can
        replay. What it cannot yield is a region name.
        """
        return self.changed and not self.recoverable

    @property
    def compensated(self) -> bool:
        """True when positions need the transform to be read in registration coordinates."""
        return self.changed and self.recoverable and self.matrix is not None


def compose(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Chain two transforms: apply `first`, then `second`."""
    return _as_3x3(second) @ _as_3x3(first)


def normalized_transform(matrix: np.ndarray, width: int, height: int) -> list[list[float]]:
    """Re-express a pixel-space transform in normalized frame coordinates.

    Stored this way, a client can map the regions a user drew into a later frame without knowing
    the source resolution, and the value survives any later re-encoding of that frame.
    """
    scale = np.diag([float(width), float(height), 1.0])
    normalized = np.linalg.inv(scale) @ _as_3x3(matrix) @ scale
    return [[float(v) for v in row] for row in normalized / normalized[2, 2]]


def project_box(box: Box, matrix: np.ndarray, width: int, height: int) -> Box | None:
    """Map a normalized box through a pixel-space camera transform.

    Returns None when the mapped centre leaves the frame, because a region drawn on the
    registration frame says nothing about scene content that was never registered.
    """
    corners = np.array(
        [
            [box.x1 * width, box.y1 * height],
            [box.x2 * width, box.y1 * height],
            [box.x2 * width, box.y2 * height],
            [box.x1 * width, box.y2 * height],
        ],
        dtype=np.float32,
    )
    mapped = _apply(_as_3x3(matrix), corners)
    if not np.all(np.isfinite(mapped)):
        return None
    centre = mapped.mean(axis=0) / [width, height]
    if not (0.0 <= centre[0] < 1.0 and 0.0 <= centre[1] < 1.0):
        return None
    return pixel_box(
        float(mapped[:, 0].min()),
        float(mapped[:, 1].min()),
        float(mapped[:, 0].max()),
        float(mapped[:, 1].max()),
        width,
        height,
    )


class SceneTracker:
    """Estimate the registered scene's pose in each sampled frame.

    `check` never mutates an earlier decision: a camera that returns to its registered pose
    reports a stable scene again, and a camera that keeps moving reports a supported transform
    for as long as the evidence supports one.
    """

    policy = {
        "version": "compensated-affine-v3",
        "minimum_matches": MINIMUM_MATCHES,
        "minimum_inlier_fraction": MINIMUM_INLIER_FRACTION,
        "minimum_hull_fraction": MINIMUM_HULL_FRACTION,
        "minimum_axis_span": MINIMUM_AXIS_SPAN,
        "motion_threshold": MOTION_THRESHOLD,
        "maximum_inlier_residual": MAXIMUM_RESIDUAL,
        "maximum_relay_depth": MAXIMUM_RELAY_DEPTH,
        "plausible_area_range": list(PLAUSIBLE_AREA_RANGE),
        "estimator": (
            "ORB features, ratio-tested matches at 0.70 then 0.82, RANSAC partial affine; a "
            "homography was measured and rejected for fitting a moving foreground"
        ),
        "recovery": (
            "a distributed, low-residual transform re-expresses positions in registration "
            "coordinates instead of abstaining"
        ),
        "relay": (
            "compose through the last supported view when the registration frame is no longer "
            "matchable; bounded depth, never backwards in time"
        ),
        "insufficient_support": "no camera-change assertion; not proof of fixed pose",
        "abstention": "camera known to have moved and no supported transform to the scene",
        "not_slam": "an affine fit is not a metric pose and not proof of a rigid planar scene",
    }

    def __init__(self, frame: np.ndarray, boxes: list[dict]):
        self.height, self.width = frame.shape[:2]
        self.scale = float(min(self.width, self.height))
        self.orb = cv2.ORB.create(nfeatures=1500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        mask = np.full(frame.shape[:2], 255, dtype=np.uint8)
        for b in boxes:
            x1, x2 = int(b["x1"] * self.width), int(b["x2"] * self.width)
            y1, y2 = int(b["y1"] * self.height), int(b["y2"] * self.height)
            mask[max(0, y1 - 15) : y2 + 15, max(0, x1 - 15) : x2 + 15] = 0
        self.points, self.descriptors = self.orb.detectAndCompute(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask
        )
        # The relay view is the most recent frame whose pose was supported, with the transform
        # that maps registration coordinates into it. It starts as the registration frame.
        self.relay_points: Sequence[cv2.KeyPoint] = self.points
        self.relay_descriptors = self.descriptors
        self.relay_matrix = _identity()
        self.relay_depth = 0
        self.moved = False
        self.counts = {"stable": 0, "compensated": 0, "unsupported": 0, "blocked": 0}
        self.peak_motion = 0.0

    # Retained so existing callers can ask "is the registered scene unusable here?".
    @property
    def changed(self) -> bool:
        return self.moved

    def check(self, frame: np.ndarray) -> SceneState:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        points, descriptors = self.orb.detectAndCompute(gray, None)
        if self.descriptors is None or len(self.descriptors) < MINIMUM_MATCHES:
            return self._unsupported("registration frame has too few features")
        if descriptors is None or len(descriptors) < MINIMUM_MATCHES:
            return self._unsupported("frame has too few features")
        direct = self._estimate(self.descriptors, self.points, descriptors, points)
        if direct is not None:
            matrix, support = direct
            support["route"] = "registration"
            support["relay_depth"] = 0
            return self._settle(matrix, support, points, descriptors, 0)
        relayed = self._estimate(self.relay_descriptors, self.relay_points, descriptors, points)
        if relayed is not None and self.relay_depth < MAXIMUM_RELAY_DEPTH:
            step, support = relayed
            composed = compose(self.relay_matrix, step)
            # Each step was checked against the view before it, but the composition is what the
            # product publishes, and a chain of accepted steps can still drift into a shape no
            # camera produced. Check the thing that is actually used.
            if _plausible(composed, self.width, self.height):
                support["route"] = "relay"
                support["relay_depth"] = self.relay_depth + 1
                return self._settle(composed, support, points, descriptors, self.relay_depth + 1)
            return self._unsupported("composed transform drifted beyond a plausible camera")
        return self._unsupported("no supported transform to the registered scene")

    def _settle(
        self,
        matrix: np.ndarray,
        support: dict,
        points: Sequence[cv2.KeyPoint],
        descriptors: np.ndarray,
        depth: int,
    ) -> SceneState:
        # Read the motion off what the transform does to the frame centre and axes, so a
        # homography and an affine are measured the same way.
        half = np.array([self.width / 2, self.height / 2], dtype=np.float32)
        probe = _apply(
            matrix, np.array([half, half + [self.width / 4, 0], half + [0, self.height / 4]])
        )
        translation = float(np.linalg.norm(probe[0] - half) / self.scale)
        axis = probe[1] - probe[0]
        rotation = float(abs(np.arctan2(axis[1], axis[0])))
        scale = float(np.linalg.norm(axis) / (self.width / 4))
        motion = max(translation, rotation, abs(scale - 1))
        support |= {
            "translation": round(translation, 5),
            "rotation": round(rotation, 5),
            "scale": round(scale, 5),
            "motion": round(motion, 5),
        }
        self.peak_motion = max(self.peak_motion, motion)
        # Every accepted estimate already passed the match, inlier, distribution and residual
        # gates, so it is strong enough to carry the scene forward to the next frame.
        self.relay_points, self.relay_descriptors = points, descriptors
        self.relay_matrix, self.relay_depth = matrix, depth
        if motion <= MOTION_THRESHOLD:
            self.moved = False
            self.counts["stable"] += 1
            return SceneState(changed=False, recoverable=True, matrix=matrix, support=support)
        self.moved = True
        self.counts["compensated"] += 1
        return SceneState(changed=True, recoverable=True, matrix=matrix, support=support)

    def _unsupported(self, why: str) -> SceneState:
        support = {"route": "none", "reason": why}
        if self.moved:
            # The camera is known to have left the registered pose and nothing relates the
            # current view to it. This is the one case the product must decline.
            self.counts["blocked"] += 1
            return SceneState(changed=True, recoverable=False, support=support)
        self.counts["unsupported"] += 1
        return SceneState(changed=False, recoverable=True, support=support)

    def _estimate(
        self,
        reference_descriptors: np.ndarray,
        reference_points: Sequence[cv2.KeyPoint],
        descriptors: np.ndarray,
        points: Sequence[cv2.KeyPoint],
    ) -> tuple[np.ndarray, dict] | None:
        if reference_descriptors is None or len(reference_descriptors) < MINIMUM_MATCHES:
            return None
        pairs = self.matcher.knnMatch(reference_descriptors, descriptors, k=2)
        for ratio in (STRICT_RATIO, LOOSE_RATIO):
            good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < ratio * m[1].distance]
            if len(good) < MINIMUM_MATCHES:
                continue
            source = np.asarray([reference_points[m.queryIdx].pt for m in good], dtype=np.float32)
            target = np.asarray([points[m.trainIdx].pt for m in good], dtype=np.float32)
            found = self._partial_affine(source, target)
            if found is not None:
                matrix, support = found
                support["matches"] = len(good)
                support["ratio"] = ratio
                return matrix, support
        return None

    def _accept(
        self,
        matrix: np.ndarray,
        source: np.ndarray,
        target: np.ndarray,
        selected: np.ndarray,
        *,
        model: str,
        minimum_inliers: int,
        minimum_fraction: float,
        maximum_residual: float,
    ) -> tuple[np.ndarray, dict] | None:
        fraction = float(selected.mean())
        if fraction < minimum_fraction or selected.sum() < minimum_inliers:
            return None
        # A locally consistent patch cannot establish a global camera transform. ORB's repeated
        # pyramid features can otherwise provide dozens of inliers inside a tiny region.
        if not all(self._distributed(p[selected]) for p in (source, target)):
            return None
        if not _plausible(matrix, self.width, self.height):
            return None
        projected = _apply(matrix, source[selected])
        residual = float(np.linalg.norm(projected - target[selected], axis=1).mean() / self.scale)
        if not np.isfinite(residual) or residual > maximum_residual:
            return None
        return matrix, {
            "model": model,
            "inliers": int(selected.sum()),
            "inlier_fraction": round(fraction, 4),
            "residual": round(residual, 5),
        }

    def _partial_affine(
        self, source: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, dict] | None:
        matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC)
        if matrix is None or inliers is None:
            return None
        return self._accept(
            _as_3x3(matrix),
            source,
            target,
            inliers.ravel().astype(bool),
            model="partial_affine",
            minimum_inliers=MINIMUM_MATCHES,
            minimum_fraction=MINIMUM_INLIER_FRACTION,
            maximum_residual=MAXIMUM_RESIDUAL,
        )

    def _distributed(self, points: np.ndarray) -> bool:
        span = np.ptp(points, axis=0) / [self.width, self.height]
        hull_fraction = cv2.contourArea(cv2.convexHull(points)) / (self.width * self.height)
        return bool(np.all(span >= MINIMUM_AXIS_SPAN) and hull_fraction >= MINIMUM_HULL_FRACTION)

    def summary(self) -> dict:
        """Measured camera evidence for the completed run, for provenance."""
        return {
            "frames": dict(self.counts),
            "peak_motion": round(self.peak_motion, 5),
            "relay_depth": self.relay_depth,
        }


# Backwards-compatible name for the earlier detection-only guard.
SceneGuard = SceneTracker
