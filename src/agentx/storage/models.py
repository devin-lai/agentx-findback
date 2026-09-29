from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def uid() -> str:
    return uuid4().hex


def now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Video(Base):
    __tablename__ = "videos"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    title: Mapped[str] = mapped_column(String(120))
    original_name: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64))
    duration_ms: Mapped[int] = mapped_column(Integer)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    fps: Mapped[float] = mapped_column(Float)
    source_path: Mapped[str] = mapped_column(String(255))
    media_path: Mapped[str] = mapped_column(String(255))
    regions: Mapped[list] = mapped_column(JSON)
    is_fixture: Mapped[bool] = mapped_column(default=False)
    # None for an imported file. A camera capture is "recording" while frames are still being
    # appended (no final hash yet), then "sealed" once it is hashed and behaves like an upload.
    live_status: Mapped[str | None] = mapped_column(String(20), nullable=True, default=None)
    capture: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class RegisteredObject(Base):
    __tablename__ = "objects"
    __table_args__ = (UniqueConstraint("video_id", "name", name="uq_object_video_name"),)
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(80))
    label: Mapped[str] = mapped_column(String(80))
    registered_at_ms: Mapped[int] = mapped_column(Integer)
    box: Mapped[dict] = mapped_column(JSON)
    reference_path: Mapped[str] = mapped_column(String(255))


class IndexRun(Base):
    __tablename__ = "index_runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    backend: Mapped[str] = mapped_column(String(30))
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    config: Mapped[dict] = mapped_column(JSON)
    object_ids: Mapped[list] = mapped_column(JSON)
    regions: Mapped[list] = mapped_column(JSON)
    processed_ms: Mapped[int] = mapped_column(Integer, default=0)
    observation_count: Mapped[int] = mapped_column(Integer, default=0)
    elapsed_seconds: Mapped[float] = mapped_column(Float, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Observation(Base):
    __tablename__ = "observations"
    __table_args__ = (
        UniqueConstraint("run_id", "object_id", "at_ms", name="uq_observation_time"),
        Index("ix_observation_history", "run_id", "object_id", "at_ms"),
    )
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    run_id: Mapped[str] = mapped_column(ForeignKey("index_runs.id", ondelete="CASCADE"))
    object_id: Mapped[str] = mapped_column(ForeignKey("objects.id", ondelete="CASCADE"))
    at_ms: Mapped[int] = mapped_column(Integer)
    box: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    zone: Mapped[str | None] = mapped_column(String(40), nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str | None] = mapped_column(String(60), nullable=True)
    # registered | compensated | unavailable: how this frame related to the registration pose.
    scene_reference: Mapped[str] = mapped_column(String(20), default="registered")
    # The 3x3 transform from registration pixels into this frame, when one was recovered. It is
    # what lets a reader see where the registered regions ended up after the camera moved.
    scene_transform: Mapped[list | None] = mapped_column(JSON, nullable=True)
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    identity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    visible: Mapped[bool] = mapped_column(default=False)


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (Index("ix_event_history", "run_id", "object_id", "at_ms"),)
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    run_id: Mapped[str] = mapped_column(ForeignKey("index_runs.id", ondelete="CASCADE"))
    object_id: Mapped[str] = mapped_column(ForeignKey("objects.id", ondelete="CASCADE"))
    at_ms: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(30))
    zone: Mapped[str | None] = mapped_column(String(40), nullable=True)
    previous_zone: Mapped[str | None] = mapped_column(String(40), nullable=True)
    reason: Mapped[str | None] = mapped_column(String(60), nullable=True)
    observation_id: Mapped[str] = mapped_column(ForeignKey("observations.id", ondelete="CASCADE"))


class QueryRecord(Base):
    __tablename__ = "queries"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    run_id: Mapped[str] = mapped_column(ForeignKey("index_runs.id", ondelete="CASCADE"), index=True)
    question: Mapped[str] = mapped_column(Text)
    cutoff_ms: Mapped[int] = mapped_column(Integer)
    response: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ReviewRecord(Base):
    __tablename__ = "reviews"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    run_id: Mapped[str] = mapped_column(ForeignKey("index_runs.id", ondelete="CASCADE"), index=True)
    response: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
