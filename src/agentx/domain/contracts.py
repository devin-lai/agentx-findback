"""Public values, independent of HTTP, SQL and inference frameworks."""

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


def timestamp(ms: int) -> str:
    seconds = ms / 1000
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


class Visibility(StrEnum):
    VISIBLE = "visible"
    LAST_SEEN = "last_seen"
    UNKNOWN = "unknown"
    NOT_OBSERVED = "not_observed"


class Box(BaseModel):
    """Normalized coordinates in the source frame."""

    model_config = ConfigDict(frozen=True)
    x1: float = Field(ge=0, le=1, allow_inf_nan=False)
    y1: float = Field(ge=0, le=1, allow_inf_nan=False)
    x2: float = Field(ge=0, le=1, allow_inf_nan=False)
    y2: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def nonempty(self) -> "Box":
        if self.x2 - self.x1 < 0.005 or self.y2 - self.y1 < 0.005:
            raise ValueError("Select a non-empty bounding box.")
        return self

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


class Region(BaseModel):
    id: str = Field(min_length=1, max_length=40)
    name: str = Field(min_length=1, max_length=60)
    box: Box


def default_regions() -> list[dict]:
    return [
        Region(
            id=name.lower(), name=f"{name} area", box=Box(x1=i / 3, y1=0, x2=(i + 1) / 3, y2=1)
        ).model_dump()
        for i, name in enumerate(["Left", "Center", "Right"])
    ]


def pixel_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Box | None:
    """Normalize an absolute-pixel xyxy rectangle, or None when it is not usable evidence.

    A perception backend may return a rectangle thinner than the registration minimum. That is
    a detection this product cannot act on, never a reason to abandon the whole frame, so the
    caller treats None exactly as it treats a missing detection.
    """
    try:
        return Box(
            x1=min(max(x1 / width, 0.0), 1.0),
            y1=min(max(y1 / height, 0.0), 1.0),
            x2=min(max(x2 / width, 0.0), 1.0),
            y2=min(max(y2 / height, 0.0), 1.0),
        )
    except ValueError:
        return None


def region_for(box: Box, regions: list[dict]) -> str | None:
    x, y = box.center
    matches = []
    for raw in regions:
        r = Region.model_validate(raw)
        b = r.box
        if b.x1 <= x < b.x2 and b.y1 <= y < b.y2:
            matches.append(r.id)
    return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True)
class Detection:
    """One backend's finding for one registered object in one sampled frame.

    `box` is always in the current frame, because that is where the evidence image is. When the
    camera has moved, `scene_box` carries the same finding in registration coordinates, which is
    the space the user's regions were drawn in. `scene_box` is None on a compensated detection
    whose position left the registered view; a region cannot name scene content that was never
    registered.
    """

    object_id: str
    box: Box | None
    score: float | None
    reason: str | None = None
    track_id: int | None = None
    identity: float | None = None
    scene_box: Box | None = None
    scene_reference: str = "registered"
    scene_transform: list[list[float]] | None = None

    @property
    def region_supported(self) -> bool:
        """Whether this frame can name a region at all.

        False means the camera evidence cannot relate the frame to the registration view, which
        is a gap in knowledge, not a report that the object is somewhere else.
        """
        if self.scene_reference == "registered":
            return True
        return self.scene_reference == "compensated" and self.scene_box is not None


def zone_of(detection: Detection, regions: list[dict]) -> str | None:
    """Name the region a detection is in, read in registration coordinates.

    A camera that has left its registered pose does not make the detection wrong: the box is
    measured in the frame a user can replay. It makes the *region name* unsupported, because
    "left area" was drawn on the registration frame. So an unrelatable view yields no region
    rather than no observation.
    """
    if detection.box is None or detection.reason:
        return None
    if detection.scene_reference == "unavailable":
        return None
    if detection.scene_reference == "compensated":
        return region_for(detection.scene_box, regions) if detection.scene_box else None
    return region_for(detection.box, regions)


@dataclass(frozen=True)
class MemoryEvent:
    object_id: str
    at_ms: int
    kind: str
    zone: str | None
    previous_zone: str | None = None
    reason: str | None = None


class RegisterObject(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    label: str = Field(default="custom", min_length=1, max_length=80)
    at_ms: int = Field(default=0, ge=0)
    box: Box


class IndexRequest(BaseModel):
    backend: str = Field(default="reference", pattern="^(reference|rtdetr|cosmos|sam2)$")
    sample_fps: float = Field(default=5, ge=1, le=15, allow_inf_nan=False)
    match_threshold: float = Field(default=0.82, ge=0.5, le=0.99, allow_inf_nan=False)


class Question(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    at_ms: int | None = Field(default=None, ge=0)
    object_id: str | None = None
    intent: str | None = Field(default=None, pattern="^(location|last_seen|history)$")
    use_provider: bool = True
    use_skills: bool = True


class SkillTrace(BaseModel):
    name: str
    sha256: str
    purpose: str


class EvidenceRef(BaseModel):
    observation_id: str
    video_id: str
    run_id: str
    at_ms: int
    box: Box
    zone: str | None
    # How this frame related to the registration camera pose: registered, compensated or
    # unavailable. The box is always drawn on the frame it was measured in.
    scene_reference: str = "registered"
    # Registration pixels mapped into this frame, so a reader can see where the regions went.
    scene_transform: list[list[float]] | None = None
    frame_url: str
    media_url: str


class ObjectMemory(BaseModel):
    object_id: str
    name: str
    status: Visibility
    as_of_ms: int
    last_observed_ms: int | None = None
    zone: str | None = None
    current_zone: str | None = None
    reason: str | None = None
    evidence: EvidenceRef | None = None
