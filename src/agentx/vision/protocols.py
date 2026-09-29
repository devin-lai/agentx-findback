"""Perception boundary: new model adapters emit the same observation candidates."""

from typing import Any, Protocol, runtime_checkable

from agentx.domain.contracts import Detection


class PerceptionError(ValueError):
    """An actionable error whose message is safe to show without host paths or secrets."""


class Detector(Protocol):
    @property
    def provenance(self) -> dict: ...

    def detect(self, frame: Any, at_ms: int) -> list[Detection]: ...


@runtime_checkable
class CameraAware(Protocol):
    """An adapter that can use the estimated camera pose for the frame it is about to see.

    Optional. A tracker that carries its own memory through motion does not need it; a matcher
    comparing against a registration-frame template does, because the camera changed the
    appearance it is matching.
    """

    def use_camera(self, state: Any) -> None: ...
