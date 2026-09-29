from sqlalchemy import select

from agentx.storage.models import IndexRun, RegisteredObject


def video_view(video, db) -> dict:
    with db.session() as session:
        objects = session.scalars(
            select(RegisteredObject).where(RegisteredObject.video_id == video.id)
        ).all()
        runs = session.scalars(
            select(IndexRun)
            .where(IndexRun.video_id == video.id)
            .order_by(IndexRun.created_at.desc())
        ).all()
    return {
        "id": video.id,
        "title": video.title,
        "original_name": video.original_name,
        "sha256": video.sha256,
        "duration_ms": video.duration_ms,
        "width": video.width,
        "height": video.height,
        "fps": video.fps,
        "regions": video.regions,
        "is_fixture": video.is_fixture,
        "live_status": video.live_status,
        "capture": video.capture,
        "created_at": video.created_at.isoformat(),
        "media_url": f"/api/v1/videos/{video.id}/media",
        "poster_url": f"/api/v1/videos/{video.id}/frame?at_ms=0",
        "objects": [object_view(o) for o in objects],
        "runs": [run_view(r) for r in runs],
    }


def object_view(obj):
    return {
        "id": obj.id,
        "name": obj.name,
        "label": obj.label,
        "registered_at_ms": obj.registered_at_ms,
        "box": obj.box,
        "reference_url": f"/api/v1/objects/{obj.id}/reference",
    }


def run_view(run):
    return {
        "id": run.id,
        "video_id": run.video_id,
        "backend": run.backend,
        "status": run.status,
        "processed_ms": run.processed_ms,
        "observation_count": run.observation_count,
        "elapsed_seconds": run.elapsed_seconds,
        "config": run.config,
        "provenance": run.provenance,
        "error": run.error,
        "created_at": run.created_at.isoformat(),
        "object_ids": run.object_ids,
        "regions": run.regions,
    }
