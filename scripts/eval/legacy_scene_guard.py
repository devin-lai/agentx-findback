"""The previously released latching scene guard, retained as an evaluation control.

`SceneTracker` replaced this implementation. Keeping the old behaviour verbatim lets a paired
comparison run both arms in one environment, against identical sources, labels, checkpoints and
sample times, instead of comparing a new run against an older report from a different session.

This module is an experiment control. It is not imported by the application.
"""

import cv2
import numpy as np

from agentx.domain.contracts import Detection
from agentx.vision.scene import SceneState


class LegacySceneGuard:
    """Detection only: once a distributed camera transform is confirmed, the scene stays invalid."""

    policy = {
        "version": "distributed-affine-v2",
        "minimum_matches": 25,
        "minimum_inlier_fraction": 0.7,
        "minimum_hull_fraction": 0.05,
        "minimum_axis_span": 0.2,
        "motion_threshold": 0.12,
        "insufficient_support": "no camera-change assertion; not proof of fixed pose",
        "control": "released behaviour before compensated-affine-v3; evaluation control only",
    }

    def __init__(self, frame: np.ndarray, boxes: list[dict]):
        self.height, self.width = frame.shape[:2]
        self.orb = cv2.ORB.create(nfeatures=500)
        mask = np.full(frame.shape[:2], 255, dtype=np.uint8)
        for b in boxes:
            x1, x2 = int(b["x1"] * self.width), int(b["x2"] * self.width)
            y1, y2 = int(b["y1"] * self.height), int(b["y2"] * self.height)
            mask[max(0, y1 - 15) : y2 + 15, max(0, x1 - 15) : x2 + 15] = 0
        self.points, self.descriptors = self.orb.detectAndCompute(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask
        )
        self.changed = False
        self.counts = {"stable": 0, "blocked": 0}

    def check(self, frame: np.ndarray) -> SceneState:
        blocked = self._check(frame)
        self.counts["blocked" if blocked else "stable"] += 1
        return SceneState(changed=blocked, recoverable=not blocked)

    def _check(self, frame: np.ndarray) -> bool:
        if self.changed:
            return True
        if self.descriptors is None or len(self.descriptors) < 25:
            return False
        points, descriptors = self.orb.detectAndCompute(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None
        )
        if descriptors is None or len(descriptors) < 25:
            return False
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(self.descriptors, descriptors, k=2)
        good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.7 * m[1].distance]
        if len(good) < 25:
            return False
        source = np.asarray([self.points[m.queryIdx].pt for m in good], dtype=np.float32)
        target = np.asarray([points[m.trainIdx].pt for m in good], dtype=np.float32)
        matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC)
        if matrix is None or inliers is None or inliers.mean() < 0.7:
            return False
        selected = inliers.ravel().astype(bool)
        if not all(self._distributed(p[selected]) for p in (source, target)):
            return False
        translation = np.linalg.norm(matrix[:, 2]) / min(self.width, self.height)
        rotation = abs(np.arctan2(matrix[1, 0], matrix[0, 0]))
        scale = np.hypot(matrix[0, 0], matrix[1, 0])
        self.changed = bool(translation > 0.12 or rotation > 0.12 or abs(scale - 1) > 0.12)
        return self.changed

    def _distributed(self, points: np.ndarray) -> bool:
        span = np.ptp(points, axis=0) / [self.width, self.height]
        hull_fraction = cv2.contourArea(cv2.convexHull(points)) / (self.width * self.height)
        return bool(np.all(span >= 0.2) and hull_fraction >= 0.05)

    def summary(self) -> dict:
        return {"frames": dict(self.counts), "control": "legacy latching guard"}


def legacy_anchor(detections: list[Detection], scene, shape) -> list[Detection]:
    """The released reaction to a camera change: discard the frame's findings.

    `_anchor` in the application annotates instead. Restoring the old reaction here gives the
    control arm the released *output* rather than a mixture of both designs.

    One difference remains and is deliberate. The released indexer skipped the detector entirely
    on a changed frame; this hook runs after it and throws the result away, so a stateful tracker
    is still stepped. That cannot change what the control arm reports, because the latching guard
    marks every later frame changed too, so every later output is `scene_changed` whatever the
    tracker's internal state became. It does inflate the control arm's indexing time, which is
    one reason no speed claim is made from these runs. The arm reproduces the published
    desk validation clip for clip, which is the check that matters.
    """
    if not scene.changed:
        return detections
    return [Detection(d.object_id, None, None, "scene_changed") for d in detections]
