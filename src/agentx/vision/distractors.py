"""Conservative instance-identity checks for causal SAM 2 tracking."""

from dataclasses import replace
from statistics import median

from agentx.domain.contracts import Box, Detection
from agentx.vision.weights import cached

MAX_MASK_BOX_AREA_JUMP = 4.0
RECENT_ACCEPTED_AREAS = 8
MIN_BASELINE_OBSERVATIONS = 3


def box_area(box: Box) -> float:
    return (box.x2 - box.x1) * (box.y2 - box.y1)


def coverage(a: Box | None, b: Box | None) -> float:
    """Intersection over the smaller area catches a mask swallowing another object."""
    if a is None or b is None:
        return 0.0
    overlap = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(
        0.0, min(a.y2, b.y2) - max(a.y1, b.y1)
    )
    area = min((a.x2 - a.x1) * (a.y2 - a.y1), (b.x2 - b.x1) * (b.y2 - b.y1))
    return overlap / area


def propose_distractors(frame, objects: list[dict], settings, device: str) -> list[dict]:
    """Use only the first registration frame; weights must already be cached."""
    import torch
    from PIL import Image
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

    categories = {o.get("label", "") for o in objects}
    name, revision = settings.detector_model, settings.detector_revision
    processor = cached(
        ("rtdetr-processor", name, revision),
        lambda: RTDetrImageProcessor.from_pretrained(
            name, revision=revision, local_files_only=True
        ),
    )
    model = cached(
        ("rtdetr", name, revision, device),
        lambda: (
            RTDetrForObjectDetection.from_pretrained(name, revision=revision, local_files_only=True)
            .to(device)
            .eval()
        ),
    )
    h, w = frame.shape[:2]
    inputs = processor(images=Image.fromarray(frame[:, :, ::-1]), return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model(**inputs)
    predicted = processor.post_process_object_detection(
        output, target_sizes=[(h, w)], threshold=0.5
    )[0]
    excluded = [Box.model_validate(o["box"]) for o in objects]
    candidates: list[dict] = []
    for raw, label, score in zip(
        predicted["boxes"].tolist(),
        predicted["labels"].tolist(),
        predicted["scores"].tolist(),
        strict=True,
    ):
        category = model.config.id2label[label]
        if category not in categories:
            continue
        try:
            box = Box(
                x1=max(0.0, raw[0] / w),
                y1=max(0.0, raw[1] / h),
                x2=min(1.0, raw[2] / w),
                y2=min(1.0, raw[3] / h),
            )
        except ValueError:
            continue
        if any(coverage(box, other) > 0.2 for other in excluded):
            continue
        if any(coverage(box, Box.model_validate(c["box"])) > 0.5 for c in candidates):
            continue
        candidates.append(dict(label=category, box=box.model_dump(), detector_score=float(score)))
        if len(candidates) == settings.sam2_max_distractors:
            break
    return candidates


class CollisionGuard:
    """Identity is not automatically recovered after overlapping same-category tracks."""

    def __init__(self, public: list[dict]):
        self.labels = {o["id"]: o.get("label", "") for o in public}
        # The user's registration rectangle may cover only part of the object. Build the size
        # baseline from actual accepted masks before judging a later mask as anomalous.
        self.accepted_areas: dict[str, list[float]] = {o["id"]: [] for o in public}
        self.auxiliary: dict[str, dict] = {}
        self.retired: set[str] = set()
        self.quarantined: set[str] = set()

    def register_auxiliary(self, objects: list[dict]) -> None:
        self.auxiliary.update({o["id"]: o for o in objects})

    def retire_duplicates(self, additions: list[dict], results: list[Detection]) -> None:
        """A user's later explicit registration supersedes a matching internal track."""
        for obj in additions:
            if obj["id"] not in self.labels:
                continue
            reference = Box.model_validate(obj["box"])
            for d in results:
                if d.object_id in self.auxiliary and (
                    obj.get("label") == self.auxiliary[d.object_id]["label"]
                    and coverage(reference, d.box) > 0.5
                ):
                    self.retired.add(d.object_id)

    def apply(self, results: list[Detection]) -> list[Detection]:
        active = [d for d in results if d.object_id not in self.retired]
        for target in active:
            if target.object_id not in self.labels:
                continue
            label = self.labels[target.object_id]
            collision = target.reason == "identity_ambiguous" or any(
                other.object_id != target.object_id
                and label
                and label
                == self.labels.get(
                    other.object_id, self.auxiliary.get(other.object_id, {}).get("label")
                )
                and coverage(target.box, other.box) >= 0.6
                for other in active
            )
            # A segmentation mask that suddenly engulfs much more than the accepted object
            # can bridge two look-alikes even when no companion existed at registration.
            # Do not let that mask teach the size baseline or silently resume the identity.
            areas = self.accepted_areas[target.object_id]
            area_jump = bool(
                target.box
                and target.reason is None
                and len(areas) >= MIN_BASELINE_OBSERVATIONS
                and box_area(target.box) > MAX_MASK_BOX_AREA_JUMP * median(areas)
            )
            if collision or area_jump:
                self.quarantined.add(target.object_id)
            elif target.box and target.reason is None and target.object_id not in self.quarantined:
                areas.append(box_area(target.box))
                del areas[:-RECENT_ACCEPTED_AREAS]
        return [
            replace(d, box=None, reason="identity_ambiguous")
            if d.object_id in self.quarantined
            else d
            for d in active
            if d.object_id in self.labels
        ]
