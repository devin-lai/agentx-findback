import hmac
import importlib.util
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Literal

import av
from fastapi import APIRouter, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from agentx.agents.providers import ReviewBusy
from agentx.api.schemas import (
    AnswerResponse,
    EventResponse,
    ObjectResponse,
    ReviewResponse,
    RunResponse,
    SkillResponse,
    SuggestionsResponse,
    VideoResponse,
)
from agentx.api.views import object_view, run_view, video_view
from agentx.domain.contracts import IndexRequest, ObjectMemory, Question, Region, RegisterObject
from agentx.services.demo import SCENES, Scene, create_fixture
from agentx.services.evidence_bundle import BundleBusyError, BundleLimitError
from agentx.services.live import LiveClosed
from agentx.services.reviews import ReviewScopeError
from agentx.storage.models import (
    Event,
    IndexRun,
    Observation,
    QueryRecord,
    RegisteredObject,
    ReviewRecord,
    Video,
)
from agentx.vision.video import jpeg, read_frame

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


def services(request):
    return request.app.state.services


def require_video(s, video_id):
    with s.db.session() as session:
        video = session.get(Video, video_id)
    if video is None:
        raise HTTPException(404, "Video not found.")
    return video


class Login(BaseModel):
    token: str = Field(max_length=512)


@router.get("/health")
def health():
    return {"status": "ok", "product": "AgentX FindBack", "version": "0.1.0"}


@router.get("/auth/status")
def auth_status(request: Request):
    return {"required": bool(services(request).settings.api_token)}


@router.post("/auth/login")
def login(body: Login, request: Request, response: Response):
    token = services(request).settings.api_token
    if token and not hmac.compare_digest(body.token.encode(), token.encode()):
        raise HTTPException(401, "Invalid access token.")
    if token:
        response.set_cookie(
            "agentx_session",
            token,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            max_age=43200,
            path="/",
        )
    return {"authenticated": True}


@router.post("/auth/logout")
def logout(response: Response):
    response.delete_cookie("agentx_session", path="/")
    return {"authenticated": False}


@router.get("/v1/capabilities")
def capabilities(request: Request):
    s = services(request)
    return {
        "version": "0.1.0",
        "planner": s.settings.planner_available,
        "planner_model": s.settings.planner_endpoint[2] or None,
        "cosmos": s.settings.cosmos_available,
        "cosmos_backend": s.settings.cosmos_backend,
        "cosmos_model": s.settings.cosmos_model if s.settings.cosmos_available else None,
        "cosmos_reference_adapter_model": s.settings.cosmos_reference_adapter_model
        if s.settings.cosmos_available and s.settings.cosmos_reference_adapter_model
        else None,
        # A fine-tuned adapter names its base checkpoint; None for a base model or no service.
        "cosmos_base_model": (s.providers.served_model() or {}).get("parent")
        if s.settings.cosmos_available
        else None,
        "cosmos_thinking": s.settings.cosmos_thinking,
        "identity": s.settings.identity_enabled,
        "sam2": s.settings.sam2_available
        and all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers")),
        "sam2_model": s.settings.sam2_model if s.settings.sam2_available else None,
        "rtdetr_installed": all(
            importlib.util.find_spec(name) is not None
            for name in ("torch", "transformers", "trackers")
        ),
        "discovery": s.discovery.available(),
        "max_upload_mb": s.settings.max_upload_mb,
        "max_video_seconds": s.settings.max_video_seconds,
        "reference": True,
        "demo_scenes": [{"id": k, "description": v} for k, v in SCENES.items()],
        "skills": s.skills.NAMES,
        "scope": "Uploaded fixed-camera video, registered objects, timestamped evidence.",
    }


@router.get("/v1/skills", response_model=list[SkillResponse])
def skills(request: Request):
    """The versioned Skills this deployment loads, with their hashes and exact text.

    An answer records the SHA-256 of every Skill in its prompt; this makes that record
    auditable against the instructions the running deployment actually holds.
    """
    return services(request).skills.catalog()


@router.get("/v1/videos", response_model=list[VideoResponse])
def videos(request: Request):
    s = services(request)
    with s.db.session() as session:
        items = session.scalars(select(Video).order_by(Video.created_at.desc())).all()
    return [video_view(v, s.db) for v in items]


@router.post("/v1/videos", status_code=201, response_model=VideoResponse)
async def upload_video(request: Request, file: UploadFile = File()):
    s = services(request)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
        raise HTTPException(415, "Upload an MP4, MOV, MKV, WebM or AVI video.")
    temporary = tempfile.NamedTemporaryFile(dir=s.settings.data_dir, suffix=suffix, delete=False)
    path = Path(temporary.name)
    try:
        total = 0
        with temporary:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > s.settings.max_upload_mb * 1024 * 1024:
                    raise HTTPException(413, "The video exceeds the upload limit.")
                temporary.write(chunk)
        if total == 0:
            raise HTTPException(422, "The uploaded video is empty.")
        video = await run_in_threadpool(s.catalog.import_video, path, file.filename or "video.mp4")
        return video_view(video, s.db)
    except av.FFmpegError:
        raise HTTPException(422, "The upload could not be decoded as a supported video.") from None
    finally:
        path.unlink(missing_ok=True)
        await file.close()


@router.post("/v1/demo", status_code=201, response_model=VideoResponse)
def demo(request: Request, scene: Scene = "fixed"):
    """Create one of the generated controlled samples, already registered."""
    s = services(request)
    names = {
        "fixed": "Controlled desk fixture.mp4",
        "bumped": "Controlled desk fixture, bumped camera.mp4",
    }
    with tempfile.TemporaryDirectory(dir=s.settings.data_dir) as directory:
        path = Path(directory) / "controlled-fixture.mp4"
        objects = create_fixture(path, scene)
        video = s.catalog.import_video(path, names[scene], is_fixture=True)
        for obj in objects:
            s.catalog.register(video.id, obj)
    return video_view(video, s.db)


@router.get("/v1/videos/{video_id}", response_model=VideoResponse)
def get_video(video_id: str, request: Request):
    s = services(request)
    return video_view(require_video(s, video_id), s.db)


@router.delete("/v1/videos/{video_id}", status_code=204)
def delete_video(video_id: str, request: Request):
    s = services(request)
    require_video(s, video_id)
    with s.indexer.enqueue_lock, s.db.session.begin() as session:
        if session.scalar(
            select(IndexRun.id).where(
                IndexRun.video_id == video_id,
                IndexRun.status.in_(["queued", "running", "cancelling"]),
            )
        ):
            raise HTTPException(409, "Wait for indexing to finish before deleting this recording.")
        if s.live.is_sealing(video_id):
            raise HTTPException(409, "This live recording is being sealed. Delete it afterwards.")
        s.live.discard(video_id)
        session.execute(delete(Video).where(Video.id == video_id))
    shutil.rmtree(s.settings.data_dir / "videos" / video_id, ignore_errors=True)


class LiveStart(BaseModel):
    title: str = Field(default="Live camera", max_length=120)
    fps: float = Field(default=5, gt=0, le=30)
    source: Literal["browser", "push"] = "browser"


def require_live(s):
    if not s.settings.live_enabled:
        raise HTTPException(404, "Live capture is disabled on this deployment.")


@router.post("/v1/live", status_code=201, response_model=VideoResponse)
def live_start(body: LiveStart, request: Request):
    """Start a camera recording that memory can follow while frames keep arriving."""
    s = services(request)
    require_live(s)
    video = s.live.create(body.title, min(body.fps, s.settings.live_max_fps), body.source)
    return video_view(video, s.db)


@router.post("/v1/live/{video_id}/frames")
async def live_frame(video_id: str, request: Request):
    """Append one JPEG or PNG frame, timestamped on arrival by the server's clock."""
    s = services(request)
    require_live(s)
    limit = s.settings.live_max_frame_bytes
    if int(request.headers.get("content-length") or 0) > limit:
        raise HTTPException(413, "A live frame exceeds the per-frame size limit.")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise HTTPException(413, "A live frame exceeds the per-frame size limit.")
    try:
        return await run_in_threadpool(s.live.append, video_id, bytes(body))
    except LiveClosed as exc:
        raise HTTPException(409, str(exc)) from None


@router.post("/v1/live/{video_id}/stop", response_model=VideoResponse)
def live_stop(video_id: str, request: Request):
    """Seal the capture: hash it, make its playback proxy, and let memory reach its end."""
    s = services(request)
    require_live(s)
    try:
        video = s.live.seal(video_id)
    except LiveClosed as exc:
        raise HTTPException(409, str(exc)) from None
    return video_view(video, s.db)


@router.get("/v1/videos/{video_id}/media")
def media(video_id: str, request: Request):
    s = services(request)
    video = require_video(s, video_id)
    path = s.catalog.path(video.media_path)
    if not path.is_file():
        raise HTTPException(404, "Playback media is missing.")
    return FileResponse(
        path, media_type="video/mp4", headers={"Cache-Control": "private, max-age=3600"}
    )


@router.get("/v1/videos/{video_id}/frame")
def frame(video_id: str, request: Request, at_ms: int = Query(default=0, ge=0)):
    s = services(request)
    video = require_video(s, video_id)
    if at_ms >= video.duration_ms:
        raise HTTPException(422, "Frame timestamp is outside this video.")
    pts, image = read_frame(s.catalog.require_source(video), at_ms)
    return Response(
        jpeg(image),
        media_type="image/jpeg",
        headers={"X-Frame-Time-Ms": str(pts), "Cache-Control": "no-store"},
    )


@router.post("/v1/videos/{video_id}/objects", status_code=201, response_model=ObjectResponse)
def register(video_id: str, body: RegisterObject, request: Request):
    s = services(request)
    with s.indexer.enqueue_lock:
        return object_view(s.catalog.register(video_id, body))


class SuggestRequest(BaseModel):
    at_ms: int = Field(default=0, ge=0)
    backend: Literal["rtdetr", "cosmos"] = "rtdetr"


@router.post("/v1/videos/{video_id}/suggestions", response_model=SuggestionsResponse)
async def suggest_objects(video_id: str, body: SuggestRequest, request: Request):
    """Advisory registration candidates for one frame. Nothing here enters memory."""
    s = services(request)
    video = require_video(s, video_id)
    if body.at_ms >= video.duration_ms:
        raise HTTPException(422, "Frame timestamp is outside this video.")
    if not s.discovery.available().get(body.backend):
        raise HTTPException(
            503,
            "RT-DETR proposals need the vision extra and its weights."
            if body.backend == "rtdetr"
            else "Cosmos proposals need a configured Cosmos service or local checkpoint.",
        )
    # Resolve user input before mapping perception failures to a sanitized 502.
    s.catalog.require_source(video)
    with s.db.session() as session:
        taken = session.scalars(
            select(RegisteredObject).where(RegisteredObject.video_id == video_id)
        ).all()
    try:
        return await run_in_threadpool(
            s.discovery.suggest, video, body.at_ms, body.backend, list(taken)
        )
    except ReviewBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception:
        # A backend that is unreachable, rejects the request or answers unparseably is not a
        # client error, and its message may name an internal endpoint. Log it, publish neither.
        logger.exception("Object discovery failed for video %s with %s", video_id, body.backend)
        raise HTTPException(
            502, "The perception backend could not propose objects. Nothing was registered."
        ) from None


@router.get("/v1/objects/{object_id}/reference")
def reference(object_id: str, request: Request):
    s = services(request)
    with s.db.session() as session:
        obj = session.get(RegisteredObject, object_id)
    if obj is None:
        raise HTTPException(404, "Object not found.")
    path = s.catalog.path(obj.reference_path)
    if not path.is_file():
        raise HTTPException(404, "Reference image is missing.")
    return FileResponse(path, media_type="image/png")


@router.put("/v1/videos/{video_id}/regions")
def regions(video_id: str, body: list[Region], request: Request):
    s = services(request)
    if not 1 <= len(body) <= 12 or len({r.id for r in body}) != len(body):
        raise HTTPException(422, "Supply 1–12 regions with unique IDs.")
    with s.indexer.enqueue_lock, s.db.session.begin() as session:
        video = session.get(Video, video_id)
        if video is None:
            raise HTTPException(404, "Video not found.")
        if session.scalar(
            select(IndexRun.id).where(
                IndexRun.video_id == video_id,
                IndexRun.status.in_(["queued", "running", "cancelling"]),
            )
        ):
            raise HTTPException(409, "Wait for indexing before changing regions.")
        video.regions = [r.model_dump() for r in body]
    return video.regions


@router.post("/v1/videos/{video_id}/runs", status_code=202, response_model=RunResponse)
def index(video_id: str, body: IndexRequest, request: Request):
    return run_view(services(request).indexer.enqueue(video_id, body))


@router.get("/v1/runs/{run_id}", response_model=RunResponse)
def run(run_id: str, request: Request):
    with services(request).db.session() as session:
        result = session.get(IndexRun, run_id)
    if result is None:
        raise HTTPException(404, "Run not found.")
    return run_view(result)


@router.get("/v1/runs/{run_id}/state", response_model=list[ObjectMemory])
def state(run_id: str, request: Request, at_ms: int | None = Query(default=None, ge=0)):
    return services(request).memory.states(run_id, at_ms)


@router.post("/v1/runs/{run_id}/cancel", response_model=RunResponse)
def cancel_run(run_id: str, request: Request):
    try:
        return run_view(services(request).indexer.cancel(run_id))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None


@router.get("/v1/runs/{run_id}/events", response_model=list[EventResponse])
def events(
    run_id: str,
    request: Request,
    at_ms: int | None = Query(default=None, ge=0),
    object_id: str | None = None,
):
    return services(request).memory.history(run_id, at_ms, object_id)


@router.post("/v1/runs/{run_id}/questions", response_model=AnswerResponse)
def question(run_id: str, body: Question, request: Request):
    return services(request).workflow.answer(run_id, body)


@router.get("/v1/runs/{run_id}/questions", response_model=list[AnswerResponse])
def questions(run_id: str, request: Request):
    s = services(request)
    with s.db.session() as session:
        if session.get(IndexRun, run_id) is None:
            raise HTTPException(404, "Run not found.")
        return [
            q.response
            for q in session.scalars(
                select(QueryRecord)
                .where(QueryRecord.run_id == run_id)
                .order_by(QueryRecord.created_at)
            ).all()
        ]


@router.get("/v1/observations/{observation_id}/frame")
def evidence_frame(observation_id: str, request: Request):
    s = services(request)
    observation, _, video = s.memory.observation(observation_id)
    if not observation.visible:
        raise HTTPException(422, "This observation has no visible object evidence.")
    path = s.catalog.path(video.source_path)
    if not path.is_file():
        raise HTTPException(404, "Original evidence is missing.")
    s.catalog.require_source(video)
    at_ms, image = read_frame(path, observation.at_ms)
    return Response(
        jpeg(image),
        media_type="image/jpeg",
        headers={"X-Frame-Time-Ms": str(at_ms), "Cache-Control": "no-store"},
    )


@router.get("/v1/questions/{question_id}/bundle")
def answer_bundle(question_id: str, request: Request):
    s = services(request)
    directory = Path(tempfile.mkdtemp(prefix="evidence-", dir=s.settings.data_dir))
    destination = directory / "evidence.zip"
    try:
        s.bundles.build(question_id, destination)
    except Exception as exc:
        shutil.rmtree(directory, ignore_errors=True)
        if isinstance(exc, BundleBusyError):
            raise HTTPException(429, str(exc), headers={"Retry-After": "5"}) from None
        if isinstance(exc, BundleLimitError):
            raise HTTPException(413, str(exc)) from None
        raise
    return FileResponse(
        destination,
        media_type="application/zip",
        filename=f"findback-evidence-{question_id}.zip",
        headers={"Cache-Control": "no-store"},
        background=BackgroundTask(shutil.rmtree, directory, ignore_errors=True),
    )


@router.post("/v1/runs/{run_id}/review", response_model=ReviewResponse)
def review(run_id: str, body: Question, request: Request):
    s = services(request)
    if not s.settings.cosmos_available:
        raise HTTPException(
            409, "Configure a Cosmos inference service or a local checkpoint first."
        )
    # Resolve user input before mapping provider failures to a sanitized 502.
    _, video, _, _ = s.memory.context(run_id, body.at_ms)
    if video.live_status == "recording":
        raise HTTPException(
            409, "Visual review cites hashed source frames. Stop the live recording first."
        )
    s.catalog.require_source(video)
    try:
        return s.reviews.create(run_id, body)
    except ReviewBusy as exc:
        raise HTTPException(429, str(exc), headers={"Retry-After": "10"}) from None
    except ReviewScopeError as exc:
        raise HTTPException(422, str(exc)) from None
    except LookupError:
        raise
    except (ValueError, OSError):
        raise HTTPException(
            502, "The visual review could not be validated. No memory was changed."
        ) from None
    except Exception:
        raise HTTPException(502, "Cosmos was unavailable. No memory was changed.") from None


@router.get("/v1/runs/{run_id}/reviews", response_model=list[ReviewResponse])
def reviews(run_id: str, request: Request):
    return services(request).reviews.list(run_id)


@router.get("/v1/reviews/{review_id}/frames/{frame_id}")
def review_frame(review_id: str, frame_id: int, request: Request):
    at_ms, encoded = services(request).reviews.frame(review_id, frame_id)
    return Response(
        encoded,
        media_type="image/jpeg",
        headers={
            "X-Frame-Time-Ms": str(at_ms),
            "Cache-Control": "private, max-age=3600",
        },
    )


@router.get("/v1/runs/{run_id}/export")
def export(run_id: str, request: Request):
    s = services(request)
    with s.db.session() as session:
        run = session.get(IndexRun, run_id)
        if run is None:
            raise HTTPException(404, "Run not found.")
        observations = session.scalars(
            select(Observation).where(Observation.run_id == run_id).order_by(Observation.at_ms)
        ).all()
        events = session.scalars(
            select(Event).where(Event.run_id == run_id).order_by(Event.at_ms)
        ).all()
        queries = session.scalars(
            select(QueryRecord).where(QueryRecord.run_id == run_id).order_by(QueryRecord.created_at)
        ).all()

        reviews = session.scalars(
            select(ReviewRecord)
            .where(ReviewRecord.run_id == run_id)
            .order_by(ReviewRecord.created_at)
        ).all()

        def row(record):
            return {c.name: getattr(record, c.name) for c in record.__table__.columns}

        data = {
            "schema_version": "1.1",
            "run": run_view(run),
            "observations": [row(o) for o in observations],
            "events": [row(e) for e in events],
            "queries": [q.response for q in queries],
            "reviews": [r.response for r in reviews],
        }
    return JSONResponse(
        data, headers={"Content-Disposition": f'attachment; filename="agentx-{run_id}.json"'}
    )
