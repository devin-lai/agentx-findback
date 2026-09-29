"""Causal SAM 2.1 video tracking, prompted only by each object's registration box."""

import importlib.metadata
from itertools import groupby
from pathlib import Path

import numpy as np

from agentx.domain.contracts import Detection
from agentx.vision.distractors import MAX_MASK_BOX_AREA_JUMP, CollisionGuard, propose_distractors
from agentx.vision.identity import AppearanceMatcher, crop
from agentx.vision.masks import mask_box as mask_box
from agentx.vision.masks import mask_geometry
from agentx.vision.presence import PresenceGate
from agentx.vision.protocols import PerceptionError
from agentx.vision.video import read_frame
from agentx.vision.weights import cached, file_sha256


def prune_session(session, frame_idx: int, window: int) -> None:
    """Keep conditioning data and enough recent memory for forward-only tracking.

    This adapter never re-propagates backwards. The pinned model uses seven mask
    memories and at most sixteen object pointers; thirty-two frames is conservative.
    Always pass frame_idx explicitly after pruning processed_frames.
    """
    before = frame_idx - window
    conditioning = {
        i for output in session.output_dict_per_obj.values() for i in output["cond_frame_outputs"]
    }
    caches = [session.processed_frames]
    caches += list(session.frames_tracked_per_obj.values())
    caches += [output["non_cond_frame_outputs"] for output in session.output_dict_per_obj.values()]
    for cache in caches:
        for old in list(cache):
            if old < before and old not in conditioning:
                del cache[old]


class Sam2Detector:
    def __init__(self, objects: list[dict], settings, source: Path):
        try:
            import torch
            from transformers import Sam2VideoModel, Sam2VideoProcessor
        except ImportError as exc:
            raise PerceptionError(
                "SAM 2 video tracking requires the vision extra with Transformers >=4.57."
            ) from exc
        if not settings.sam2_available:
            raise PerceptionError(
                "Configure AGENTX_SAM2_MODEL_PATH with an existing SAM 2.1 checkpoint."
            )
        self.torch = torch
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if settings.model_device == "auto"
            else settings.model_device
        )
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        path = settings.sam2_model_path
        self.processor = cached(
            ("sam2-processor", path),
            lambda: Sam2VideoProcessor.from_pretrained(path, local_files_only=True),
        )
        # A checkpoint replaced at the same path must load again, or provenance would record the
        # new file's hash for inference that still ran on the old weights.
        weights = Path(path) / "model.safetensors"
        version = (weights.stat().st_size, weights.stat().st_mtime_ns)
        self.model = cached(
            ("sam2", path, version, self.device, str(self.dtype)),
            lambda: (
                Sam2VideoModel.from_pretrained(path, local_files_only=True)
                .to(self.device, dtype=self.dtype)
                .eval()
            ),
        )
        self.window = settings.sam2_memory_frames
        if self.window < max(
            32, self.model.config.num_maskmem + self.model.config.max_object_pointers_in_encoder
        ):
            raise PerceptionError(
                "The configured SAM2 cache is too short for this checkpoint's memory."
            )
        self.state_device = self.device if settings.sam2_state_device == "auto" else "cpu"
        self.session = self.processor.init_video_session(
            inference_device=self.device,
            inference_state_device=self.state_device,
            video_storage_device=self.state_device,
            dtype=self.dtype,
        )
        self.source = source
        self.settings = settings
        self.collisions = CollisionGuard(objects) if settings.sam2_distractor_guard else None
        self.pending = sorted(objects, key=lambda o: (o["registered_at_ms"], o["id"]))
        self.active: dict[int, dict] = {}
        self.frame_idx = 0
        self.last_sample_ms = -1
        self.last_inferred_ms = -1
        self.last_results: list[Detection] = []
        self.threshold = settings.sam2_presence_threshold
        self.appearance = (
            AppearanceMatcher(settings, self.device, offline=True)
            if settings.identity_enabled
            else None
        )
        self.gate = PresenceGate(
            settings.sam2_presence_threshold,
            settings.sam2_confident_presence,
            settings.sam2_appearance_threshold,
        )
        weights_sha256 = file_sha256(Path(settings.sam2_model_path) / "model.safetensors")
        self._provenance = {
            "adapter": "sam2-causal-streaming",
            "model": settings.sam2_model,
            "revision": settings.sam2_revision,
            "model_path": settings.sam2_model_path,
            "weights_sha256": weights_sha256,
            "device": self.device,
            "dtype": str(self.dtype),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "learned_model": True,
            "memory_frames": self.window,
            "state_device": self.state_device,
            "presence_threshold": self.threshold,
            "presence_gate": {
                "confident_presence": self.gate.confident,
                "appearance_threshold": self.gate.similarity,
                "recent_accepted_views": self.gate.recent,
                "policy": "minimum presence, then confident presence OR accepted-view similarity",
                "appearance": {
                    k: v
                    for k, v in self.appearance.provenance.items()
                    if k not in {"identity_threshold", "identity_margin", "identity_method"}
                }
                if self.appearance
                else None,
                "embedding": "L2-normalized DINOv2 CLS + mean patch; registration and accepted views",
            },
            "score_semantics": "sigmoid object-presence logit; not a calibrated identity probability",
            "prompt": "original-frame registration box; no future frames or labels",
            "identity_method": "causal per-object segmentation memory; overlapping masks are ambiguous",
            "model_downloads_at_runtime": False,
            "mask_geometry": "exact integer device reductions on CUDA; NumPy on CPU",
            "distractor_guard": {
                "enabled": bool(self.collisions),
                "maximum_auxiliary_tracks": settings.sam2_max_distractors,
                "proposal_model": settings.detector_model,
                "proposal_revision": settings.detector_revision,
                "proposal_threshold": 0.5,
                "collision_coverage": 0.6,
                "maximum_mask_box_area_jump": MAX_MASK_BOX_AREA_JUMP,
                "policy": "first registration frame same-category companion tracks; persistent ambiguity after collision or abrupt mask bounding-box expansion; auxiliary tracks never become inventory or evidence",
            },
        }

    @property
    def provenance(self) -> dict:
        return self._provenance

    def detect(self, frame, at_ms: int) -> list[Detection]:
        if at_ms <= self.last_sample_ms:
            raise PerceptionError("SAM2 tracking requires strictly increasing sample times.")
        self.last_sample_ms = at_ms
        ready = [o for o in self.pending if o["registered_at_ms"] <= at_ms]
        self.pending = [o for o in self.pending if o["registered_at_ms"] > at_ms]
        for registered_ms, group in groupby(ready, key=lambda o: o["registered_at_ms"]):
            if registered_ms < self.last_inferred_ms:
                raise PerceptionError(
                    "Cannot introduce a registration behind the tracking watermark."
                )
            registration_frame = (
                frame if registered_ms == at_ms else read_frame(self.source, registered_ms)[1]
            )
            self._advance(registration_frame, registered_ms, list(group))
        if self.active and self.last_inferred_ms < at_ms:
            self._advance(frame, at_ms, [])
        return self.last_results if self.active else []

    def _advance(self, frame, at_ms: int, additions: list[dict]) -> None:
        from PIL import Image

        if self.collisions and self.frame_idx == 0 and additions:
            candidates = propose_distractors(frame, additions, self.settings, self.device)
            companions = [
                {**obj, "id": f"__agentx_companion_{i}", "registered_at_ms": at_ms}
                for i, obj in enumerate(candidates)
            ]
            self.collisions.register_auxiliary(companions)
            self._provenance["distractor_guard"].update(
                initialized_at_ms=at_ms, companions=companions
            )
            additions = [*additions, *companions]
        inputs = self.processor(
            images=Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1])), return_tensors="pt"
        )
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        if additions:
            h, w = frame.shape[:2]
            ids, boxes = [], []
            for obj in additions:
                model_id = len(self.active) + 1
                self.active[model_id] = obj
                ids.append(model_id)
                b = obj["box"]
                boxes.append([b["x1"] * w, b["y1"] * h, b["x2"] * w, b["y2"] * h])
            if self.appearance:
                embeddings = self.appearance.embed([crop(frame, box) for box in boxes])
                for model_id, embedding in zip(ids, embeddings, strict=True):
                    self.gate.seed(model_id, embedding)
            self.processor.add_inputs_to_inference_session(
                self.session,
                frame_idx=self.frame_idx,
                obj_ids=ids,
                input_boxes=[boxes],
                original_size=inputs.original_sizes[0],
            )
        with self.torch.inference_mode():
            output = self.model(
                inference_session=self.session,
                frame_idx=self.frame_idx,
                frame=inputs.pixel_values[0],
            )
            masks = self.processor.post_process_masks(
                [output.pred_masks], original_sizes=inputs.original_sizes, binarize=False
            )[0]
            mask_boxes, overlapping_pairs = mask_geometry(masks[:, 0] > 0)
            logits = []
            for index in range(len(self.session.obj_ids)):
                conditioning = (
                    self.frame_idx in self.session.output_dict_per_obj[index]["cond_frame_outputs"]
                )
                logits.append(
                    self.session.get_output(
                        index,
                        self.frame_idx,
                        "object_score_logits",
                        is_conditioning_frame=conditioning,
                    ).reshape(())
                )
            scores = self.torch.sigmoid(self.torch.stack(logits).float()).cpu().tolist()
            results = []
            for index, model_id in enumerate(self.session.obj_ids):
                score = scores[index]
                box = mask_boxes[index] if score >= self.threshold else None
                results.append(
                    Detection(
                        self.active[model_id]["id"],
                        box,
                        score,
                        None if box else "not_detected",
                        track_id=model_id,
                    )
                )
            ambiguous: set[int] = set()
            if self.collisions:
                self.collisions.retire_duplicates(additions, results)
            eligible = {
                i
                for i, d in enumerate(results)
                if d.box and not (self.collisions and d.object_id in self.collisions.retired)
            }
            for i, j in overlapping_pairs:
                if i in eligible and j in eligible:
                    ambiguous.update((i, j))
            for i in ambiguous:
                d = results[i]
                results[i] = Detection(
                    d.object_id, None, d.score, "identity_ambiguous", track_id=d.track_id
                )
        prune_session(self.session, self.frame_idx, self.window)
        previous_banks = (
            {key: list(bank) for key, bank in self.gate.banks.items()} if self.collisions else {}
        )
        self.last_results = self._check_presence(frame, results)
        if self.collisions:
            self.last_results = self.collisions.apply(self.last_results)
            for detection in self.last_results:
                if (
                    detection.reason == "identity_ambiguous"
                    and detection.track_id in previous_banks
                ):
                    self.gate.banks[detection.track_id] = previous_banks[detection.track_id]
        self.last_inferred_ms = at_ms
        self.frame_idx += 1

    def _check_presence(self, frame, results: list[Detection]) -> list[Detection]:
        candidates = [d for d in results if d.box and not d.reason]
        if not candidates:
            return results
        embeddings = {}
        if self.appearance:
            h, w = frame.shape[:2]
            crops = [
                crop(frame, [d.box.x1 * w, d.box.y1 * h, d.box.x2 * w, d.box.y2 * h])
                for d in candidates
                if d.box
            ]
            embeddings = dict(
                zip((d.track_id for d in candidates), self.appearance.embed(crops), strict=True)
            )
        checked = {}
        for d in candidates:
            assert d.track_id is not None and d.score is not None
            accepted, similarity = self.gate.admit(d.track_id, d.score, embeddings.get(d.track_id))
            checked[d.object_id] = Detection(
                d.object_id,
                d.box if accepted else None,
                d.score,
                None if accepted else "identity_unconfirmed",
                d.track_id,
                similarity,
            )
        return [checked.get(d.object_id, d) for d in results]
