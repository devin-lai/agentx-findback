"""Optional neural perception adapter. Heavy imports stay outside the application core."""

import importlib.metadata
from pathlib import Path
from threading import Lock
from typing import Any

from agentx.domain.contracts import Box, Detection, pixel_box
from agentx.vision.protocols import PerceptionError
from agentx.vision.weights import cached


class RTDetrDetector:
    def __init__(self, objects: list[dict], settings, fps: float):
        try:
            import supervision as sv
            import torch
            from trackers import ByteTrackTracker
            from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        except ImportError as exc:
            raise PerceptionError(
                "RT-DETR requires the vision extra: uv sync --extra vision"
            ) from exc
        self.torch = torch
        self.sv = sv
        self.objects = objects
        if settings.model_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = settings.model_device
        name, revision, device = settings.detector_model, settings.detector_revision, self.device
        self.processor = cached(
            ("rtdetr-processor", name, revision),
            lambda: RTDetrImageProcessor.from_pretrained(name, revision=revision),
        )
        self.model = cached(
            ("rtdetr", name, revision, device),
            lambda: (
                RTDetrForObjectDetection.from_pretrained(name, revision=revision).to(device).eval()
            ),
        )
        self.labels = {int(k): v for k, v in self.model.config.id2label.items()}
        allowed = set(self.labels.values())
        invalid = [o["label"] for o in objects if o["label"] not in allowed]
        if invalid:
            raise PerceptionError(
                f"Unsupported RT-DETR categories: {', '.join(invalid)}. Use a COCO category or reference tracking."
            )
        labels = [o["label"] for o in objects]
        self.identity = None
        self.references: dict[str, Any] = {}
        if settings.identity_enabled:
            import cv2

            from agentx.vision.identity import AppearanceMatcher

            self.identity = AppearanceMatcher(settings, self.device)
            for obj in objects:
                image = cv2.imread(str(Path(settings.data_dir) / obj["reference_path"]))
                if image is None:
                    raise PerceptionError(
                        f"{obj['name']}: the registration crop is missing. Register the object again."
                    )
                self.references[obj["id"]] = self.identity.embed([image])[0]
        elif len(labels) != len(set(labels)):
            raise PerceptionError(
                "This adapter supports one registered object per category unless identity matching is enabled."
            )
        self.trackers = {label: ByteTrackTracker(frame_rate=max(1, round(fps))) for label in labels}
        self._provenance = {
            "adapter": "rtdetr-bytetrack" + ("-dinov2" if self.identity else ""),
            "model": settings.detector_model,
            "revision": settings.detector_revision,
            "device": self.device,
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "trackers": importlib.metadata.version("trackers"),
            "learned_model": True,
            "identity": self.identity.provenance if self.identity else None,
        }

    @property
    def provenance(self) -> dict:
        return self._provenance

    def detect(self, frame, at_ms: int) -> list[Detection]:
        import numpy as np
        from PIL import Image

        h, w = frame.shape[:2]
        inputs = self.processor(images=Image.fromarray(frame[:, :, ::-1]), return_tensors="pt").to(
            self.device
        )
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        prediction = self.processor.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=0.1
        )[0]
        labels = prediction["labels"].tolist()
        boxes_all = prediction["boxes"].detach().cpu().numpy()
        scores_all = prediction["scores"].detach().cpu().numpy()
        results: list[Detection] = []
        by_label: dict[str, list[dict]] = {}
        for obj in self.objects:
            if at_ms >= obj["registered_at_ms"]:
                by_label.setdefault(obj["label"], []).append(obj)
        for label, group in by_label.items():
            indices = [i for i, code in enumerate(labels) if self.labels[code] == label]
            boxes = boxes_all[indices].reshape(-1, 4)
            scores = scores_all[indices]
            tracked = self.trackers[label].update(
                self.sv.Detections(
                    xyxy=boxes, confidence=scores, class_id=np.zeros(len(indices), dtype=int)
                )
            )
            strong = np.flatnonzero(scores >= 0.5)
            if self.identity is None:
                obj = group[0]
                if len(strong) == 0:
                    results.append(Detection(obj["id"], None, None, "not_detected"))
                elif len(strong) > 1:
                    results.append(Detection(obj["id"], None, None, "identity_ambiguous"))
                else:
                    i = strong[0]
                    results.append(
                        self._detection(obj, boxes[i], float(scores[i]), tracked, w, h, None)
                    )
                continue
            from agentx.vision.identity import crop

            candidates = (
                self.identity.embed([crop(frame, boxes[i]) for i in strong])
                if len(strong)
                else np.zeros((0, 1), dtype=np.float32)
            )
            assignment = self.identity.assign(
                {o["id"]: self.references[o["id"]] for o in group}, candidates
            )
            for obj in group:
                index, similarity, reason = assignment[obj["id"]]
                if index is None:
                    results.append(
                        Detection(
                            obj["id"],
                            None,
                            float(scores[strong].max()) if len(strong) else None,
                            reason,
                            identity=similarity,
                        )
                    )
                else:
                    i = strong[index]
                    results.append(
                        self._detection(obj, boxes[i], float(scores[i]), tracked, w, h, similarity)
                    )
        return results

    def _detection(self, obj, box, score, tracked, w, h, identity) -> Detection:
        import numpy as np

        x1, y1, x2, y2 = box
        track_id = None
        if tracked.tracker_id is not None and len(tracked.xyxy):
            distances = np.abs(tracked.xyxy - box).sum(axis=1)
            j = int(distances.argmin())
            if distances[j] < 8 and int(tracked.tracker_id[j]) >= 0:
                track_id = int(tracked.tracker_id[j])
        # A detection too thin to register is one missing observation, not a failed run.
        normalized = pixel_box(float(x1), float(y1), float(x2), float(y2), w, h)
        return Detection(
            obj["id"],
            normalized,
            score,
            None if normalized else "not_detected",
            track_id=track_id,
            identity=identity,
        )


class RTDetrProposer:
    """Registration candidates from one frame: COCO categories, no tracking, no identity.

    Discovery only proposes; nothing here writes an observation or touches memory. The model
    is loaded once per process because this runs while a person waits on a button press.
    """

    _cache: "RTDetrProposer | None" = None
    _cache_key: tuple | None = None
    _lock = Lock()

    def __init__(self, settings):
        try:
            import torch
            from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        except ImportError as exc:
            raise PerceptionError(
                "RT-DETR requires the vision extra: uv sync --extra vision"
            ) from exc
        self.torch = torch
        if settings.model_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = settings.model_device
        self.processor = RTDetrImageProcessor.from_pretrained(
            settings.detector_model, revision=settings.detector_revision
        )
        self.model = (
            RTDetrForObjectDetection.from_pretrained(
                settings.detector_model, revision=settings.detector_revision
            )
            .to(self.device)
            .eval()
        )
        self.labels = {int(k): v for k, v in self.model.config.id2label.items()}
        # The cached model is shared by concurrent requests; one forward pass at a time.
        self.inference_lock = Lock()
        self.provenance = {
            "adapter": "rtdetr-proposals",
            "model": settings.detector_model,
            "revision": settings.detector_revision,
            "device": self.device,
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "learned_model": True,
        }

    @classmethod
    def shared(cls, settings) -> "RTDetrProposer":
        """One cached proposer per weight/device configuration, rebuilt when that changes."""
        key = (settings.detector_model, settings.detector_revision, settings.model_device)
        with cls._lock:
            if cls._cache is None or cls._cache_key != key:
                cls._cache, cls._cache_key = cls(settings), key
            return cls._cache

    def propose(self, frame, threshold: float) -> list[tuple[str, Box, float]]:
        from PIL import Image

        h, w = frame.shape[:2]
        with self.inference_lock:
            inputs = self.processor(
                images=Image.fromarray(frame[:, :, ::-1]), return_tensors="pt"
            ).to(self.device)
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
            prediction = self.processor.post_process_object_detection(
                outputs, target_sizes=[(h, w)], threshold=threshold
            )[0]
        found = []
        for code, box, score in zip(
            prediction["labels"].tolist(),
            prediction["boxes"].detach().cpu().numpy().tolist(),
            prediction["scores"].detach().cpu().numpy().tolist(),
            strict=True,
        ):
            # Too thin to register, so it is dropped here exactly as the caller drops
            # undersized proposals; one such detection must not discard the usable ones.
            x1, y1, x2, y2 = box
            normalized = pixel_box(x1, y1, x2, y2, w, h)
            if normalized is not None:
                found.append((self.labels[code], normalized, float(score)))
        return sorted(found, key=lambda entry: entry[2], reverse=True)
