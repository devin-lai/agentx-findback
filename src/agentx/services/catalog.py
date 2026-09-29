import hashlib
import shutil
from pathlib import Path

import cv2
from sqlalchemy import select

from agentx.domain.contracts import RegisterObject, default_regions
from agentx.services.source_integrity import SourceIntegrity, SourceIntegrityError
from agentx.storage.models import IndexRun, RegisteredObject, Video, uid
from agentx.vision.video import create_preview, jpeg, probe, read_frame


class Catalog:
    def __init__(self, database, settings):
        self.db = database
        self.settings = settings
        self.source_integrity = SourceIntegrity()

    def source_problem(self, video) -> str | None:
        if video.live_status == "recording":
            # A growing capture has no final hash yet; it is hashed when the recording is sealed.
            path = self.path(video.source_path)
            return None if path.is_file() else "evidence_missing"
        return self.source_integrity.problem(self.path(video.source_path), video.sha256)

    def require_source(self, video) -> Path:
        if problem := self.source_problem(video):
            messages = {
                "evidence_missing": "Original evidence is missing.",
                "evidence_changed": "Original evidence no longer matches the imported recording.",
                "evidence_unavailable": "Original evidence cannot be read.",
            }
            raise SourceIntegrityError(messages[problem])
        return self.path(video.source_path)

    def path(self, relative: str) -> Path:
        path = (self.settings.data_dir / relative).resolve()
        if not path.is_relative_to(self.settings.data_dir):
            raise ValueError("Invalid stored media path.")
        return path

    def import_video(self, temporary: Path, filename: str, *, is_fixture: bool = False) -> Video:
        info = probe(temporary)
        if info.duration_ms > self.settings.max_video_seconds * 1000:
            raise ValueError(
                f"Video exceeds the {self.settings.max_video_seconds // 60}-minute limit."
            )
        video_id = uid()
        directory = self.settings.data_dir / "videos" / video_id
        directory.mkdir(parents=True)
        try:
            source = directory / "source.mp4"
            shutil.move(str(temporary), source)
            with source.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            preview = directory / "playback.mp4"
            create_preview(source, preview)
            _, first = read_frame(source, 0)
            (directory / "poster.jpg").write_bytes(jpeg(first))
            safe_name = Path(filename.replace("\\", "/")).name[:255] or "video.mp4"
            video = Video(
                id=video_id,
                title=Path(safe_name).stem[:120],
                original_name=safe_name,
                sha256=checksum,
                duration_ms=info.duration_ms,
                width=info.width,
                height=info.height,
                fps=info.fps,
                source_path=str(source.relative_to(self.settings.data_dir)),
                media_path=str(preview.relative_to(self.settings.data_dir)),
                regions=default_regions(),
                is_fixture=is_fixture,
            )
            with self.db.session.begin() as session:
                session.add(video)
            return video
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise

    def register(self, video_id: str, request: RegisterObject) -> RegisteredObject:
        with self.db.session.begin() as session:
            video = session.get(Video, video_id)
            if video is None:
                raise LookupError("Video not found.")
            if request.at_ms >= video.duration_ms:
                raise ValueError("Registration time must be inside the video.")
            if session.scalar(
                select(IndexRun.id).where(
                    IndexRun.video_id == video_id,
                    IndexRun.status.in_(["queued", "running", "cancelling"]),
                )
            ):
                raise ValueError("Wait for the current index run before changing registration.")
            name = request.name.strip()
            if not name:
                raise ValueError("An object name is required.")
            objects = session.scalars(
                select(RegisteredObject).where(RegisteredObject.video_id == video_id)
            ).all()
            if len(objects) >= 10:
                raise ValueError("This MVP supports at most 10 registered objects per video.")
            if any(o.name.casefold() == name.casefold() for o in objects):
                raise ValueError("Object names must be unique within this video.")
            at_ms, frame = read_frame(self.require_source(video), request.at_ms)
            b = request.box
            h, w = frame.shape[:2]
            crop = frame[int(b.y1 * h) : int(b.y2 * h), int(b.x1 * w) : int(b.x2 * w)]
            if min(crop.shape[:2]) < 8:
                raise ValueError("Select an object at least 8 pixels wide and tall.")
            object_id = uid()
            relative = f"videos/{video_id}/reference-{object_id}.png"
            ok = cv2.imwrite(str(self.path(relative)), crop)
            if not ok:
                raise ValueError("Could not store the reference image.")
            obj = RegisteredObject(
                id=object_id,
                video_id=video_id,
                name=name,
                label=request.label.strip().lower(),
                registered_at_ms=at_ms,
                box=b.model_dump(),
                reference_path=relative,
            )
            session.add(obj)
            return obj
