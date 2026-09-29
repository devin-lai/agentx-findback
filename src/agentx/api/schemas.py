"""Public response contracts. Storage internals and filesystem paths stay private."""

from typing import Any, Literal

from pydantic import BaseModel

from agentx.domain.contracts import Box, EvidenceRef, ObjectMemory, Region, SkillTrace


class ObjectResponse(BaseModel):
    id: str
    name: str
    label: str
    registered_at_ms: int
    box: Box
    reference_url: str


class RunResponse(BaseModel):
    id: str
    video_id: str
    backend: Literal["reference", "rtdetr", "cosmos", "sam2"]
    status: Literal["queued", "running", "cancelling", "cancelled", "complete", "failed"]
    processed_ms: int
    observation_count: int
    elapsed_seconds: float
    config: dict[str, Any]
    provenance: dict[str, Any]
    error: str | None
    created_at: str
    object_ids: list[str]
    regions: list[Region]


class VideoResponse(BaseModel):
    id: str
    title: str
    original_name: str
    sha256: str
    duration_ms: int
    width: int
    height: int
    fps: float
    regions: list[Region]
    is_fixture: bool
    # "recording" while a camera capture is still growing, "sealed" once it has been hashed.
    live_status: Literal["recording", "sealed"] | None = None
    capture: dict[str, Any] | None = None
    created_at: str
    media_url: str
    poster_url: str
    objects: list[ObjectResponse]
    runs: list[RunResponse]


class EventResponse(BaseModel):
    id: str
    object_id: str
    name: str
    at_ms: int
    kind: str
    zone: str | None
    previous_zone: str | None
    reason: str | None
    observation_id: str


class AnswerResponse(BaseModel):
    id: str
    run_id: str
    question: str
    intent: Literal["location", "last_seen", "history"]
    as_of_ms: int
    answer: str
    states: list[ObjectMemory]
    events: list[EventResponse]
    evidence: list[EvidenceRef]
    skills: list[SkillTrace]
    tools: list[dict[str, Any]]
    # "stepfun" only appears in rows persisted by the previous release.
    planner: Literal["local", "agent", "stepfun"]
    planner_model: str | None = None
    warnings: list[str]
    elapsed_seconds: float


class ReviewFrameResponse(BaseModel):
    id: int
    at_ms: int
    sha256: str
    frame_url: str


class FrameReportResponse(BaseModel):
    id: int
    at_ms: int
    present: bool
    x: float | None
    y: float | None
    zone: str | None
    zone_name: str | None
    note: str
    # How this reviewed frame related to the registration camera pose, when it was checked.
    camera: str | None = None
    # Present only when the focus pass ran: the crop it examined and whether it agreed.
    focus: dict | None = None


class ReviewResponse(BaseModel):
    id: str
    run_id: str
    video_id: str
    source_sha256: str
    question: str
    as_of_ms: int
    mode: Literal["freeform", "grounded"] = "freeform"
    target: str | None = None
    frame_reports: list[FrameReportResponse] = []
    summary: str
    evidence_frame_ids: list[int]
    uncertainty: str
    frames: list[ReviewFrameResponse]
    model: str
    provenance: dict[str, Any]
    skills: list[SkillTrace]
    elapsed_seconds: float
    authoritative: Literal[False]
    notice: str


class ProposalResponse(BaseModel):
    suggested_name: str
    label: str
    box: Box
    score: float | None = None


class SuggestionsResponse(BaseModel):
    """Advisory registration candidates for one frame; never an observation or a memory record."""

    at_ms: int
    backend: Literal["rtdetr", "cosmos"]
    proposals: list[ProposalResponse]
    provenance: dict[str, Any]
    elapsed_seconds: float
    limits: str


class SkillResponse(BaseModel):
    """A versioned instruction file, with the exact text an answer's SHA-256 refers to."""

    name: str
    version: str | None = None
    description: str | None = None
    purpose: str
    sha256: str
    body: str
