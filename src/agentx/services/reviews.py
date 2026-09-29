"""Persist successful interpretations separately from authoritative object observations."""

import hashlib
import json
import tempfile
from pathlib import Path

import cv2
from sqlalchemy import select

from agentx.domain.contracts import Question
from agentx.storage.models import IndexRun, ReviewRecord, Video, uid
from agentx.vision.video import bounded_jpeg, jpeg, read_frame


class ReviewScopeError(ValueError):
    """The requested review would cross the object's registration boundary."""


class Reviews:
    def __init__(self, db, catalog, memory, providers, skills):
        self.db, self.catalog, self.memory = db, catalog, memory
        self.providers, self.skills = providers, skills

    def create(self, run_id: str, question: Question) -> dict:
        run, video, objects, cutoff = self.memory.context(run_id, question.at_ms)
        bodies, traces = [], []
        if question.use_skills:
            body, trace = self.skills.load("review-visual-evidence")
            bodies.append(body)
            traces.append(trace.model_dump())
        target = reference = None
        camera_poses: list[dict] = []
        start_ms = 0
        if question.object_id:
            obj = next((o for o in objects if o.id == question.object_id), None)
            if obj is None:
                raise LookupError("Selected object does not belong to this run.")
            if cutoff < obj.registered_at_ms:
                raise ReviewScopeError("The selected object was not registered at this cutoff.")
            start_ms = obj.registered_at_ms
            target = obj.name if obj.label == "custom" else f"{obj.name} ({obj.label})"
            crop = cv2.imread(str(self.catalog.path(obj.reference_path)))
            if crop is not None:
                reference = jpeg(crop)
            # Indexing already estimated the camera for every sampled frame of this run, with the
            # registered objects masked out. Reading those back is both cheaper than re-estimating
            # from eight frames and the only way the review and the memory can be guaranteed to
            # name the same place.
            camera_poses = self.memory.camera_poses(run_id, max(0, cutoff - 7000), cutoff)
        result = self.providers.review(
            self.catalog.require_source(video),
            cutoff,
            question.text,
            bodies,
            target=target,
            regions=run.regions,
            reference=reference,
            start_ms=start_ms,
            camera_poses=camera_poses,
            camera_tolerance_ms=round(500 / run.config["sample_fps"]),
        )
        review_id = uid()
        for frame in result["frames"]:
            frame["frame_url"] = f"/api/v1/reviews/{review_id}/frames/{frame['id']}"
        response = {
            **result,
            "id": review_id,
            "run_id": run_id,
            "video_id": video.id,
            "source_sha256": video.sha256,
            "question": question.text,
            "as_of_ms": cutoff,
            "skills": traces,
        }
        # Preserve exactly what the model saw. Re-encoding after a runtime migration can
        # change JPEG bytes even when the source video is unchanged.
        for entry in response["frames"]:
            self._frame_bytes(video, response, entry)
        with self.db.session() as session:
            # Deleting the recording while inference runs must not resurrect its review.
            if session.get(IndexRun, run_id) is None:
                raise LookupError("The recording was removed while the review was running.")
            session.add(ReviewRecord(id=review_id, run_id=run_id, response=response))
            session.commit()
        return response

    def list(self, run_id: str) -> list[dict]:
        with self.db.session() as session:
            if session.get(IndexRun, run_id) is None:
                raise LookupError("Run not found.")
            return [
                r.response
                for r in session.scalars(
                    select(ReviewRecord)
                    .where(ReviewRecord.run_id == run_id)
                    .order_by(ReviewRecord.created_at)
                ).all()
            ]

    def frame(self, review_id: str, frame_id: int) -> tuple[int, bytes]:
        with self.db.session() as session:
            record = session.get(ReviewRecord, review_id)
            if record is None:
                raise LookupError("Review not found.")
            response = record.response
            entry = next((f for f in response["frames"] if f["id"] == frame_id), None)
            if entry is None:
                raise LookupError("Frame was not supplied to this review.")
            video = session.get(Video, response["video_id"])
            if video is None:
                raise LookupError("Recording not found.")
        return self._frame_bytes(video, response, entry)

    def _frame_bytes(self, video, response: dict, entry: dict) -> tuple[int, bytes]:
        path = self.catalog.require_source(video)
        # Bind the cache to the source, timestamp, geometry and reviewed bytes. A changed
        # record must not reuse an older frame merely because its image hash was retained.
        scope = [video.sha256, entry["at_ms"], response["provenance"]["max_edge"], entry["sha256"]]
        key = hashlib.sha256(json.dumps(scope).encode()).hexdigest()
        cached = path.parent / "review_frames" / f"{key}.jpg"
        if cached.is_file():
            encoded = cached.read_bytes()
            if hashlib.sha256(encoded).hexdigest() != entry["sha256"]:
                raise ValueError(
                    "Cached evidence no longer matches the frame supplied to the model."
                )
            return entry["at_ms"], encoded
        at_ms, image = read_frame(path, entry["at_ms"])
        encoded = bounded_jpeg(image, response["provenance"]["max_edge"])
        if at_ms != entry["at_ms"] or hashlib.sha256(encoded).hexdigest() != entry["sha256"]:
            raise ValueError("Evidence no longer matches the frame supplied to the model.")
        # Cache within the video's directory so deletion removes it too. Never recreate
        # a removed video directory, and never expose a partially written frame.
        cached.parent.mkdir(exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=cached.parent, prefix=".frame-", delete=False
            ) as f:
                temporary = Path(f.name)
                f.write(encoded)
            temporary.replace(cached)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return at_ms, encoded
