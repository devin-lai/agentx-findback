"""Causal state transitions. Observations never get rewritten using later frames."""

from dataclasses import dataclass

from agentx.domain.contracts import Detection, MemoryEvent, zone_of


@dataclass
class TrackMemory:
    visible: bool = False
    seen: bool = False
    zone: str | None = None
    pending_zone: str | None = None
    pending_count: int = 0
    reason: str | None = None


class EventReducer:
    def __init__(self, regions: list[dict]) -> None:
        self.regions = regions
        self.objects: dict[str, TrackMemory] = {}
        self.last_at_ms = -1

    def update(self, at_ms: int, detections: list[Detection]) -> list[MemoryEvent]:
        if at_ms <= self.last_at_ms:
            raise ValueError("Observation times must be strictly increasing within a run.")
        self.last_at_ms = at_ms
        events = []
        for d in detections:
            state = self.objects.setdefault(d.object_id, TrackMemory())
            if d.box is None or d.reason:
                if state.visible or (d.reason and d.reason != state.reason):
                    events.append(
                        MemoryEvent(
                            d.object_id,
                            at_ms,
                            "ambiguous"
                            if d.reason
                            in {"identity_ambiguous", "identity_unconfirmed", "scene_changed"}
                            else "lost",
                            None,
                            state.zone,
                            d.reason or "not_detected",
                        )
                    )
                state.visible = False
                state.pending_count = 0
                state.reason = d.reason
                continue
            zone = zone_of(d, self.regions)
            if not d.region_supported:
                # The camera cannot be related to the registration view, so this frame says
                # nothing about which registered area the object is in. Reporting that as a
                # change would invent a movement out of a gap in knowledge.
                if not state.visible:
                    # Seen again, but where cannot be said: no area, not the one it was lost from.
                    events.append(
                        MemoryEvent(
                            d.object_id,
                            at_ms,
                            "reappeared" if state.seen else "appeared",
                            None,
                            state.zone,
                        )
                    )
                state.visible = True
                state.seen = True
                state.reason = None
                state.pending_count = 0
                continue
            if not state.visible:
                events.append(
                    MemoryEvent(
                        d.object_id,
                        at_ms,
                        "reappeared" if state.seen else "appeared",
                        zone,
                        state.zone,
                    )
                )
                state.zone = zone
            elif state.zone is None:
                # Visible all along but never placed (first seen while the camera could not be
                # related). Learning the area now is not evidence that the object moved.
                state.zone = zone
                state.pending_count = 0
            elif zone != state.zone:
                state.pending_count = state.pending_count + 1 if zone == state.pending_zone else 1
                state.pending_zone = zone
                if state.pending_count >= 2:
                    events.append(MemoryEvent(d.object_id, at_ms, "moved", zone, state.zone))
                    state.zone = zone
                    state.pending_count = 0
            else:
                state.pending_count = 0
            state.visible = True
            state.seen = True
            state.reason = None
        return events
