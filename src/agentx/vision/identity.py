"""Optional appearance-embedding identity check (DINOv2) for learned detections.

A detector proposes category boxes; this matcher decides which proposal is the registered
object by comparing crop embeddings with the reference crop captured at registration. It lets
several registered objects share a category and turns look-alike candidates into an explicit
`identity_ambiguous` result instead of a silent identity switch.
"""

import importlib.metadata
from typing import Any

import numpy as np

from agentx.vision.weights import cached


class AppearanceMatcher:
    def __init__(self, settings, device: str, *, offline: bool = False):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        source = settings.identity_model_path or settings.identity_model
        kwargs: dict[str, Any] = (
            {"local_files_only": True}
            if settings.identity_model_path
            else {"revision": settings.identity_revision}
        )
        if offline:
            kwargs["local_files_only"] = True
        self.torch = torch
        self.device = device
        key = (source, tuple(sorted(kwargs.items())))
        self.processor = cached(
            ("dinov2-processor", *key), lambda: AutoImageProcessor.from_pretrained(source, **kwargs)
        )
        self.model = cached(
            ("dinov2", *key, device),
            lambda: AutoModel.from_pretrained(source, **kwargs).to(device).eval(),
        )
        self.threshold = settings.identity_threshold
        self.margin = settings.identity_margin
        self.provenance = {
            "identity_model": settings.identity_model,
            "identity_revision": settings.identity_revision
            if not settings.identity_model_path
            else "local checkpoint",
            "identity_threshold": settings.identity_threshold,
            "identity_margin": settings.identity_margin,
            "identity_transformers": importlib.metadata.version("transformers"),
            "identity_method": "cosine similarity of DINOv2 CLS+mean-patch embeddings against the registration crop",
        }

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """L2-normalized embeddings for BGR crops, one row per crop."""
        from PIL import Image

        if not crops:
            return np.zeros((0, 1), dtype=np.float32)
        images = [Image.fromarray(np.ascontiguousarray(c[:, :, ::-1])) for c in crops]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state.float()
        features = self.torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=-1)
        features = self.torch.nn.functional.normalize(features, dim=-1)
        return features.cpu().numpy()

    def assign(
        self, references: dict[str, np.ndarray], candidates: np.ndarray
    ) -> dict[str, tuple[int | None, float | None, str | None]]:
        """Greedy one-to-one assignment: object id -> (candidate index, similarity, reason).

        reason is None for a confirmed match, 'identity_unconfirmed' when no candidate reaches the
        threshold, and 'identity_ambiguous' when a competing candidate, or a competing registered
        object for the same candidate, is within the margin.
        """
        result: dict[str, tuple[int | None, float | None, str | None]] = {}
        if len(candidates) == 0:
            return {oid: (None, None, "not_detected") for oid in references}
        scores = {oid: candidates @ ref for oid, ref in references.items()}
        pairs = sorted(
            ((float(s[i]), oid, i) for oid, s in scores.items() for i in range(len(candidates))),
            reverse=True,
        )
        taken: set[int] = set()
        for sim, oid, i in pairs:
            if oid in result or i in taken:
                continue
            if sim < self.threshold:
                result[oid] = (None, float(scores[oid].max()), "identity_unconfirmed")
                continue
            alternatives = [
                float(scores[oid][j]) for j in range(len(candidates)) if j != i and j not in taken
            ]
            if alternatives and max(alternatives) >= sim - self.margin:
                result[oid] = (None, sim, "identity_ambiguous")
                continue
            # Another registered object that matches this candidate about as well makes the
            # assignment a coin flip; both objects become ambiguous and the candidate is spent.
            # A coin flip: another registered object matches this candidate about as well and
            # has no other candidate to fall back on. Look-alikes with their own candidates keep
            # the greedy one-to-one order, which the fixture evaluation showed to be reliable.
            rivals = []
            for other in references:
                if other == oid or other in result:
                    continue
                own = float(scores[other][i])
                fallback = [
                    float(scores[other][j])
                    for j in range(len(candidates))
                    if j != i and j not in taken and float(scores[other][j]) >= self.threshold
                ]
                if own >= sim - self.margin and not fallback:
                    rivals.append(other)
            if rivals:
                result[oid] = (None, sim, "identity_ambiguous")
                for other in rivals:
                    result[other] = (None, float(scores[other][i]), "identity_ambiguous")
                taken.add(i)
                continue
            result[oid] = (i, sim, None)
            taken.add(i)
        for oid in references:
            result.setdefault(oid, (None, float(scores[oid].max()), "identity_unconfirmed"))
        return result


def crop(frame: np.ndarray, box, pad: float = 0.08) -> np.ndarray:
    """A slightly padded crop for an absolute-pixel xyxy box."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    xa, ya = max(0, int(x1 - dx)), max(0, int(y1 - dy))
    xb, yb = min(w, int(x2 + dx) + 1), min(h, int(y2 + dy) + 1)
    return frame[ya:yb, xa:xb]
