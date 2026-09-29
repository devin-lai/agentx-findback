"""Transparent classical-CV baseline. It is not a trained object detector."""

from pathlib import Path

import cv2
import numpy as np

from agentx.domain.contracts import Detection, pixel_box
from agentx.vision.protocols import PerceptionError


class ReferenceDetector:
    def __init__(self, objects: list[dict], data_dir: Path, threshold: float) -> None:
        self.threshold = threshold
        self.objects = objects
        self.templates = {}
        # The camera pose for the frame about to be matched, and the templates rewarped for it.
        self.camera: np.ndarray | None = None
        self.warped: dict[str, np.ndarray] = {}
        for obj in objects:
            image = cv2.imread(str(data_dir / obj["reference_path"]))
            if image is None:
                raise PerceptionError(
                    "A registered reference image is missing. Register the object again."
                )
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if float(gray.std()) < 4:
                raise PerceptionError(f"{obj['name']}: select a textured object, not a flat patch.")
            self.templates[obj["id"]] = gray

    @property
    def provenance(self) -> dict:
        return {
            "adapter": "opencv-reference",
            "opencv_version": cv2.__version__,
            "method": "multi-scale normalized template correlation",
            "camera_compensation": (
                "the registration template is rewarped by the estimated camera transform before "
                "matching, so a panned or zoomed view is compared against the appearance that "
                "camera would actually produce"
            ),
            "learned_model": False,
        }

    def use_camera(self, state) -> None:
        """Adopt the camera pose estimated for the next frame.

        A registration template is a picture taken from the registered pose. Once the camera
        moves, that picture is no longer what the object looks like, and correlation collapses
        exactly when the user most needs an answer.
        """
        matrix = state.matrix if state.compensated else None
        if matrix is None:
            self.camera, self.warped = None, {}
            return
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape == (3, 3):
            if abs(matrix[2, 2]) < 1e-9:
                self.camera, self.warped = None, {}
                return
            matrix = matrix / matrix[2, 2]
        # Only the local linear part changes the template's appearance; translation moves where
        # it is found, which the search already handles.
        linear = matrix[:2, :2].astype(np.float32)
        if self.camera is not None and np.allclose(linear, self.camera, atol=1e-3):
            return
        self.camera = linear
        self.warped = {}
        for key, template in self.templates.items():
            h, w = template.shape
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32) @ linear.T
            low, high = corners.min(axis=0), corners.max(axis=0)
            size = (int(round(high[0] - low[0])), int(round(high[1] - low[1])))
            if min(size) < 8 or max(size) > 4 * max(w, h):
                continue  # An implausible warp is not a better template than the original.
            affine = np.hstack([linear, (-low).reshape(2, 1)])
            self.warped[key] = cv2.warpAffine(
                template, affine, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )

    def detect(self, frame: np.ndarray, at_ms: int) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        results = []
        for obj in self.objects:
            if at_ms < obj["registered_at_ms"]:
                continue
            template = self.warped.get(obj["id"], self.templates[obj["id"]])
            candidates = []
            for scale in (0.85, 1.0, 1.15):
                tw, th = round(template.shape[1] * scale), round(template.shape[0] * scale)
                if tw < 8 or th < 8 or tw >= width or th >= height:
                    continue
                resized = cv2.resize(template, (tw, th))
                response = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
                _, score, _, point = cv2.minMaxLoc(response)
                if not np.isfinite(score) or score < self.threshold:
                    continue
                x, y = point
                candidates.append((score, x, y, tw, th))
                response[
                    max(0, y - th // 2) : min(response.shape[0], y + th // 2 + 1),
                    max(0, x - tw // 2) : min(response.shape[1], x + tw // 2 + 1),
                ] = -1
                _, second, _, p2 = cv2.minMaxLoc(response)
                if second >= self.threshold and second >= score - 0.04:
                    candidates.append((second, p2[0], p2[1], tw, th))
            if not candidates:
                results.append(Detection(obj["id"], None, None, "not_detected"))
                continue
            candidates.sort(reverse=True)
            score, x, y, tw, th = candidates[0]
            ambiguous = any(
                abs(cx - x) > tw * 0.7 or abs(cy - y) > th * 0.7
                for s, cx, cy, _, _ in candidates[1:]
                if s >= score - 0.04
            )
            if ambiguous:
                results.append(Detection(obj["id"], None, score, "identity_ambiguous"))
                continue
            # A downscaled template can correlate best while covering too little of the frame
            # to be a usable location. That is a missing observation, not a failed run.
            box = pixel_box(x, y, x + tw, y + th, width, height)
            results.append(Detection(obj["id"], box, score, None if box else "not_detected"))
        return results
