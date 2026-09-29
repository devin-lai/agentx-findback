"""Render location claims from memory records, never from planner prose."""

from agentx.domain.contracts import Visibility, timestamp

SUPPORTED_EVENTS = {"appeared", "moved", "reappeared"}
NO_MATCH = (
    "I could not match that request to a registered object in this recording. "
    "Select an object or use its exact registered name."
)
AMBIGUOUS_MATCH = (
    "Several registered objects match. Select one to avoid confusing their identities."
)


def grounded_answer(memory, run, video, obj, cutoff: int, intent: str) -> dict:
    """One selected object's state and replayable claims at a frozen cutoff.

    A planner may choose the object, intent, and tools. It cannot override the
    facts rendered here, even when its proposed citations are syntactically valid.
    Every historical location in the text has a corresponding original frame.
    """
    state = memory.object_state(run, video, obj, cutoff)
    events = memory.history(run.id, cutoff, obj.id) if intent == "history" else []
    regions = {r["id"]: r["name"] for r in run.regions}
    zone = regions.get(state.zone, "an unassigned area")
    evidence = [state.evidence.model_dump(mode="json")] if state.evidence else []
    camera = ""
    if state.evidence is not None and state.evidence.scene_reference == "compensated":
        camera = (
            " The camera had moved by then; this region was recovered from the registration "
            "view, so it names the same place you drew, not the same pixels."
        )
    elif state.evidence is not None and state.evidence.scene_reference == "unavailable":
        camera = (
            " The camera had moved and could not be related to the registration view, so the "
            "frame is evidence of the object but not of which registered area it is in."
        )
    if state.status == Visibility.VISIBLE and intent != "last_seen":
        message = (
            f"At video {timestamp(state.last_observed_ms)}, {obj.name} is visible in {zone}. "
            "This describes the recording, not its real-world current location."
        ) + camera
    elif state.evidence:
        message = (
            f"The last supported observation of {obj.name} at or before {timestamp(cutoff)} "
            f"is at {timestamp(state.last_observed_ms)}, in {zone}."
        ) + camera
        if state.status != Visibility.VISIBLE:
            message += " Its position at the requested time is not confirmed."
            reasons = {
                "scene_changed": "The camera view changed; register a new scene before relying on locations.",
                "identity_ambiguous": "Multiple candidates make the identity uncertain.",
                "identity_unconfirmed": "A candidate was detected but did not match the registered appearance.",
            }
            if state.reason in reasons:
                message += " " + reasons[state.reason]
    else:
        message = (
            f"There is no usable visual evidence for {obj.name} at or before {timestamp(cutoff)}."
        )
    supported = []
    if state.evidence:
        for event in events:
            if event["kind"] not in SUPPORTED_EVENTS:
                continue
            ref = memory.evidence_for(event["observation_id"], cutoff).model_dump(mode="json")
            supported.append(event)
            if ref["observation_id"] not in {e["observation_id"] for e in evidence}:
                evidence.append(ref)
    if supported:
        message += (
            " Recorded changes: "
            + "; ".join(
                f"{timestamp(e['at_ms'])} — {regions.get(e['zone'], 'unassigned area')}"
                for e in supported
            )
            + "."
        )
    for item in evidence:
        observation, _, _ = memory.observation(item["observation_id"])
        if (
            item["at_ms"] > cutoff
            or item["video_id"] != video.id
            or item["run_id"] != run.id
            or observation.object_id != obj.id
        ):
            raise RuntimeError("Evidence failed the object, temporal or source boundary.")
    return {
        "intent": intent,
        "answer": message,
        "states": [state.model_dump(mode="json")],
        "events": events,
        "evidence": evidence,
    }
