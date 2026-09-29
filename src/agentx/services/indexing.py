import logging
import threading
import time
from dataclasses import asdict

import av
from sqlalchemy import case, select, update

from agentx.domain.contracts import Detection, IndexRequest, zone_of
from agentx.domain.temporal import EventReducer
from agentx.services.source_integrity import SourceIntegrityError
from agentx.storage.models import Event, IndexRun, Observation, RegisteredObject, Video, uid
from agentx.vision.protocols import CameraAware, Detector, PerceptionError
from agentx.vision.reference import ReferenceDetector
from agentx.vision.scene import SceneTracker, normalized_transform, project_box
from agentx.vision.video import frames


def _anchor(detections: list[Detection], scene, shape) -> list[Detection]:
    """Express each finding in registration coordinates when the camera has left its pose.

    The evidence box stays in the current frame, because that is the image a user replays. The
    added `scene_box` is what the region names were drawn against, so "left area" keeps meaning
    the left of the registered desk rather than the left of a moved camera. When no transform
    relates the view to the registration frame, the finding keeps its box and loses its region.
    """
    if not scene.changed:
        return detections
    inverse = None
    if scene.compensated:
        import numpy as np

        try:
            inverse = np.linalg.inv(scene.matrix)
        except np.linalg.LinAlgError:
            inverse = None
    height, width = shape[:2]
    reference = "compensated" if inverse is not None else "unavailable"
    transform = normalized_transform(scene.matrix, width, height) if inverse is not None else None
    return [
        Detection(
            d.object_id,
            d.box,
            d.score,
            d.reason,
            d.track_id,
            d.identity,
            project_box(d.box, inverse, width, height)
            if d.box is not None and inverse is not None
            else None,
            reference,
            transform,
        )
        for d in detections
    ]


logger = logging.getLogger(__name__)

ACTIVE_RUN_STATUSES = ("queued", "running", "cancelling")


class IndexCancelled(Exception):
    """A user-requested stop, acknowledged between bounded inference calls."""


def capture_slack(sample_fps: float) -> float:
    """A capture is stamped on arrival, so its frames jitter around the capture rate. Accept a
    frame up to a quarter interval early, the same tolerance the capture itself enforces."""
    return 250 / sample_fps


def _follow(source, sample_fps: float, start_ms: int, end_ms: int | None = None):
    """Frames of a capture, tolerating a final fragment that is cut off.

    While recording, `end_ms` is the newest frame the writer has declared complete: bytes past it
    may be a fragment that is still being written, which a decoder would conceal rather than
    reject. A sealed capture is read to its end, which after a crash may stop mid-fragment.
    """
    try:
        yield from frames(source, sample_fps, start_ms, end_ms, slack_ms=capture_slack(sample_fps))
    except (av.error.InvalidDataError, av.error.EOFError, av.error.ValueError) as exc:
        logger.info("Capture %s ends mid-fragment after %s ms: %s", source, start_ms, exc)


def object_dict(obj) -> dict:
    return {
        k: getattr(obj, k)
        for k in ("id", "name", "label", "registered_at_ms", "box", "reference_path")
    }


class Indexer:
    def __init__(self, database, settings, catalog, skills, providers=None):
        self.db, self.settings, self.catalog, self.skills = database, settings, catalog, skills
        self.providers = providers
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.thread: threading.Thread | None = None
        self.enqueue_lock = threading.Lock()
        self.cancel_requests: dict[str, threading.Event] = {}

    def start(self):
        with self.db.session.begin() as session:
            session.execute(
                update(IndexRun).where(IndexRun.status == "cancelling").values(status="cancelled")
            )
            session.execute(
                update(IndexRun)
                .where(IndexRun.status == "running")
                .values(
                    status="failed",
                    error="Indexing was interrupted. Start a new run to retry; partial observations remain scoped to this run.",
                )
            )
        self.thread = threading.Thread(target=self._loop, name="agentx-indexer", daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=10)

    def enqueue(self, video_id: str, request: IndexRequest) -> IndexRun:
        with self.enqueue_lock, self.db.session.begin() as session:
            video = session.get(Video, video_id)
            if video is None:
                raise LookupError("Video not found.")
            self.catalog.require_source(video)
            if session.scalar(
                select(IndexRun.id).where(
                    IndexRun.video_id == video_id, IndexRun.status.in_(ACTIVE_RUN_STATUSES)
                )
            ):
                raise ValueError("An index run is already queued or running for this video.")
            objects = session.scalars(
                select(RegisteredObject).where(RegisteredObject.video_id == video_id)
            ).all()
            if not objects:
                raise ValueError("Register at least one object before building memory.")
            if request.backend == "sam2" and not self.settings.sam2_available:
                raise ValueError("Configure an existing SAM 2.1 checkpoint before SAM2 indexing.")
            if request.backend == "cosmos":
                if self.providers is None or not self.settings.cosmos_available:
                    raise ValueError(
                        "Configure a Cosmos service or checkpoint before Cosmos indexing."
                    )
                # Freshness in memory reads uses the stored rate, so clamp it before persisting.
                request = request.model_copy(
                    update={"sample_fps": min(request.sample_fps, self.settings.cosmos_index_fps)}
                )
            run = IndexRun(
                id=uid(),
                video_id=video_id,
                backend=request.backend,
                config=request.model_dump(),
                object_ids=[o.id for o in objects],
                regions=video.regions,
                status="queued",
                processed_ms=0,
                observation_count=0,
                elapsed_seconds=0,
                provenance={"pipeline_version": "0.1.0"},
            )
            session.add(run)
        self.wake.set()
        return run

    def cancel(self, run_id: str) -> IndexRun:
        """Persist intent before notifying the single in-process worker."""
        with self.enqueue_lock, self.db.session.begin() as session:
            session.execute(
                update(IndexRun)
                .where(IndexRun.id == run_id, IndexRun.status.in_(("queued", "running")))
                .values(status=case((IndexRun.status == "queued", "cancelled"), else_="cancelling"))
            )
            run = session.get(IndexRun, run_id)
            if run is None:
                raise LookupError("Run not found.")
            if run.status not in {"cancelling", "cancelled"}:
                raise ValueError("Only queued or running memory versions can be stopped.")
        with self.enqueue_lock:
            if signal := self.cancel_requests.get(run_id):
                signal.set()
        self.wake.set()
        return run

    def _loop(self):
        while not self.stop.is_set():
            with self.db.session() as session:
                run_id = session.scalar(
                    select(IndexRun.id)
                    .where(IndexRun.status == "queued")
                    .order_by(IndexRun.created_at)
                    .limit(1)
                )
            if run_id:
                self.process(run_id)
            else:
                self.wake.wait(0.5)
                self.wake.clear()

    def process(self, run_id: str):
        started = time.monotonic()
        cancelled = threading.Event()
        try:
            with self.enqueue_lock, self.db.session.begin() as session:
                claimed = session.execute(
                    update(IndexRun)
                    .where(IndexRun.id == run_id, IndexRun.status == "queued")
                    .values(status="running")
                    .returning(IndexRun.id)
                ).scalar_one_or_none()
                if claimed is None:
                    return
                self.cancel_requests[run_id] = cancelled
                run = session.get(IndexRun, run_id)
                video = session.get(Video, run.video_id)
                objects = session.scalars(
                    select(RegisteredObject).where(RegisteredObject.id.in_(run.object_ids))
                ).all()
                descriptors = [object_dict(o) for o in objects]
            source = self.catalog.require_source(video)
            detector: Detector
            if run.backend == "reference":
                detector = ReferenceDetector(
                    descriptors, self.settings.data_dir, run.config["match_threshold"]
                )
            elif run.backend == "sam2":
                from agentx.vision.sam2 import Sam2Detector

                detector = Sam2Detector(descriptors, self.settings, source)
            elif run.backend == "cosmos":
                from agentx.vision.cosmos_detector import CosmosDetector

                detector = CosmosDetector(
                    descriptors, self.settings, self.providers, self.settings.data_dir
                )
            else:
                from agentx.vision.rtdetr import RTDetrDetector

                detector = RTDetrDetector(descriptors, self.settings, run.config["sample_fps"])
            _, skill = self.skills.load("observe-object-events")
            provenance = {
                "pipeline_version": "0.1.0",
                **detector.provenance,
                "skill": skill.model_dump(),
                "input_sha256": video.sha256,
                "regions": run.regions,
                "identity_scope": "registered fixed inventory; no long-term re-identification guarantee",
                "scene_guard": SceneTracker.policy,
            }
            live = video.live_status == "recording"
            # Camera captures (growing or sealed) share one sampling policy, so a rebuild of a
            # sealed capture reads exactly the frames that following it live did.
            capture = video.live_status is not None
            if live:
                # A growing capture has no final hash; it is recorded once the capture is sealed.
                provenance["input_sha256"] = None
                provenance["live"] = {"followed_while_recording": True}
            with self.db.session.begin() as session:
                if cancelled.is_set():
                    raise IndexCancelled()
                session.get(IndexRun, run_id).provenance = provenance
            reducer = EventReducer(run.regions)
            guard = None
            total = 0
            batch = []
            sample_fps = run.config["sample_fps"]
            # A live capture is followed from its earliest registration: nothing before it can be
            # remembered, and there is no finished file to read from the start in one pass.
            cursor = min(o["registered_at_ms"] for o in descriptors) if capture else 0
            while True:
                # Read the state before the pass, so the pass after sealing reaches the file's end.
                status, readable_end = self._capture_state(video.id)
                sealed = not live or status != "recording"
                source_frames = (
                    _follow(source, sample_fps, cursor, None if sealed else readable_end - 1)
                    if capture
                    else frames(source, sample_fps)
                )
                for at_ms, frame in source_frames:
                    if cancelled.is_set():
                        raise IndexCancelled()
                    if self.stop.is_set():
                        raise PerceptionError(
                            "Indexing interrupted during application shutdown. Start a new run to retry."
                        )
                    if guard is None:
                        guard = SceneTracker(
                            frame,
                            [
                                o["box"]
                                for o in descriptors
                                if (
                                    o["registered_at_ms"] <= at_ms
                                    if capture
                                    else o["registered_at_ms"] == 0
                                )
                            ],
                        )
                    scene = guard.check(frame)
                    if isinstance(detector, CameraAware):
                        detector.use_camera(scene)
                    detections = _anchor(detector.detect(frame, at_ms), scene, frame.shape)
                    if cancelled.is_set():
                        raise IndexCancelled()
                    events = reducer.update(at_ms, detections)
                    batch.append((at_ms, detections, events))
                    cursor = round(at_ms + 1000 / sample_fps)
                    if len(batch) >= 10:
                        total += self._write_batch(run_id, run.regions, batch, started)
                        batch = []
                if batch and live:
                    # Memory a live question can read must not wait for ten more frames.
                    total += self._write_batch(run_id, run.regions, batch, started)
                    batch = []
                if sealed:
                    break
                if cancelled.wait(0.25):
                    raise IndexCancelled()
                if self.stop.is_set():
                    raise PerceptionError(
                        "Indexing interrupted during application shutdown. Start a new run to retry."
                    )
            if batch:
                total += self._write_batch(run_id, run.regions, batch, started)
            if live:
                with self.db.session() as session:
                    video = session.get(Video, video.id)
                provenance["input_sha256"] = video.sha256
            self.catalog.require_source(video)
            with self.enqueue_lock, self.db.session.begin() as session:
                current = session.get(IndexRun, run_id)
                if current.status == "cancelling" or cancelled.is_set():
                    raise IndexCancelled()
                current.status = "complete"
                # Include causal initialization diagnostics discovered while indexing.
                current.provenance = {
                    **provenance,
                    **detector.provenance,
                    "camera": guard.summary() if guard else None,
                }
                current.observation_count = total
                current.elapsed_seconds = round(time.monotonic() - started, 3)
        except IndexCancelled:
            with self.enqueue_lock, self.db.session.begin() as session:
                session.execute(
                    update(IndexRun)
                    .where(IndexRun.id == run_id, IndexRun.status == "cancelling")
                    .values(
                        status="cancelled", elapsed_seconds=round(time.monotonic() - started, 3)
                    )
                )
        except Exception as exc:
            with self.enqueue_lock, self.db.session.begin() as session:
                run = session.get(IndexRun, run_id)
                if run:
                    run.status = "cancelled" if run.status == "cancelling" else "failed"
                    if run.status == "failed":
                        logger.exception("Indexing failed for run %s", run_id)
                        run.error = (
                            str(exc)[:500]
                            if isinstance(exc, (PerceptionError, SourceIntegrityError))
                            else "The perception worker failed. Check server logs and retry."
                        )
                    else:
                        logger.info("Indexing stopped after cancellation for run %s", run_id)
                    run.elapsed_seconds = round(time.monotonic() - started, 3)
        finally:
            with self.enqueue_lock:
                if self.cancel_requests.get(run_id) is cancelled:
                    self.cancel_requests.pop(run_id, None)

    def _capture_state(self, video_id: str) -> tuple[str | None, int]:
        """A capture's status and its readable end: every frame before it is completely written."""
        with self.db.session() as session:
            row = session.execute(
                select(Video.live_status, Video.duration_ms).where(Video.id == video_id)
            ).one_or_none()
        return (row[0], row[1]) if row else (None, 0)

    def _write_batch(self, run_id, regions, batch, started):
        count = 0
        with self.enqueue_lock, self.db.session.begin() as session:
            current = session.get(IndexRun, run_id)
            if current.status in {"cancelling", "cancelled"}:
                raise IndexCancelled()
            for at_ms, detections, events in batch:
                ids = {}
                for d in detections:
                    observation = Observation(
                        id=uid(),
                        run_id=run_id,
                        object_id=d.object_id,
                        at_ms=at_ms,
                        box=d.box.model_dump() if d.box else None,
                        zone=zone_of(d, regions),
                        score=d.score,
                        reason=d.reason,
                        scene_reference=d.scene_reference,
                        scene_transform=d.scene_transform,
                        track_id=d.track_id,
                        identity_score=d.identity,
                        visible=d.box is not None and not d.reason,
                    )
                    session.add(observation)
                    ids[d.object_id] = observation.id
                    count += 1
                session.flush()
                for event in events:
                    session.add(
                        Event(run_id=run_id, observation_id=ids[event.object_id], **asdict(event))
                    )
            current.processed_ms = batch[-1][0]
            current.observation_count += count
            current.elapsed_seconds = round(time.monotonic() - started, 3)
        return count
