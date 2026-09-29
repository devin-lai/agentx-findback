"""Bounded, causal object selection over authoritative observations and events.

The planner translates a description into typed predicates. Code evaluates every
eligible object and decides zero/one/many matches; the planner cannot choose an
arbitrary member of an ambiguous result. Translation itself remains fallible.
"""

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select

from agentx.domain.contracts import Visibility
from agentx.domain.text import find_literal_mention, toolbox_name_aliases
from agentx.services.selection_history import validate_history_quantifiers
from agentx.services.selection_time import validate_time_filters
from agentx.storage.models import Observation


class Predicate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class NamePredicate(Predicate):
    kind: Literal["name"]
    text: str = Field(min_length=1, max_length=120)


class StateFields(Predicate):
    at_ms: int | None = Field(default=None, ge=0)
    status: Literal["visible", "last_seen", "unknown", "not_observed", "not_visible"] | None = None
    zone: str | None = None
    last_observed_zone: str | None = None

    @model_validator(mode="after")
    def meaningful(self):
        if self.status is None and self.zone is None and self.last_observed_zone is None:
            raise ValueError("State selection requires status or zone.")
        return self


class StatePredicate(StateFields):
    kind: Literal["state"]


class OtherStatePredicate(StateFields):
    """At least one different registered object matches the state at that time."""

    kind: Literal["other_state"]


class WindowPredicate(Predicate):
    start_ms: int = Field(default=0, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    min_count: int = Field(default=1, ge=0)
    max_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_range(self):
        if self.end_ms is not None and self.end_ms < self.start_ms:
            raise ValueError("The time window is reversed.")
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("The count range is reversed.")
        if self.min_count == 0 and self.max_count is None:
            raise ValueError("An unconstrained count cannot select objects.")
        return self


class SeenPredicate(WindowPredicate):
    kind: Literal["seen"]
    zone: str
    outside_zone: bool = False


class RegionHistoryPredicate(Predicate):
    """Named quantifiers avoid composing a negation with an outside-zone flag."""

    kind: Literal["never_seen_in", "only_seen_in"]
    zone: str
    start_ms: int = Field(default=0, ge=0)
    end_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_range(self):
        if self.end_ms is not None and self.end_ms < self.start_ms:
            raise ValueError("The time window is reversed.")
        return self


class EventPredicate(WindowPredicate):
    kind: Literal["event"]
    event: Literal["appeared", "moved", "lost", "reappeared", "ambiguous"]
    zone: str | None = None
    previous_zone: str | None = None


class FirstSeenPredicate(Predicate):
    kind: Literal["first_seen"]
    zone: str | None = None
    start_ms: int = Field(default=0, ge=0)
    end_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_range(self):
        if self.end_ms is not None and self.end_ms < self.start_ms:
            raise ValueError("The time window is reversed.")
        return self


class RegisteredPredicate(Predicate):
    kind: Literal["registered"]
    start_ms: int = Field(default=0, ge=0)
    end_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_range(self):
        if self.end_ms is not None and self.end_ms < self.start_ms:
            raise ValueError("The time window is reversed.")
        return self


SelectionPredicate = Annotated[
    NamePredicate
    | StatePredicate
    | OtherStatePredicate
    | SeenPredicate
    | RegionHistoryPredicate
    | EventPredicate
    | FirstSeenPredicate
    | RegisteredPredicate,
    Field(discriminator="kind"),
]


class SelectionRequest(Predicate):
    filters: list[SelectionPredicate] = Field(min_length=1, max_length=8)


class SelectionPlan(Predicate):
    interpretation: str = Field(default="", max_length=600)
    intent: Literal["location", "last_seen", "history"]
    filters: list[SelectionPredicate] = Field(max_length=8)


def describe_selection(
    filters: list[dict], regions: list[dict], cutoff: int, scope_names=()
) -> str:
    """Describe the executed predicates, not the model's potentially inconsistent prose."""
    names = {r["id"]: r["name"] for r in regions}

    def seconds(value):
        return f"{value / 1000:.3f} s"

    def zone(value):
        return names.get(value, value)

    clauses = ["Named candidates: " + ", ".join(sorted(scope_names))] if scope_names else []
    for item in filters:
        kind = item["kind"]
        if kind == "name":
            clauses.append(f"name/category contains {item['text']!r}")
            continue
        at = seconds(item.get("at_ms") if item.get("at_ms") is not None else cutoff)
        start = seconds(item.get("start_ms", 0))
        end = seconds(item.get("end_ms") if item.get("end_ms") is not None else cutoff)
        if kind in {"state", "other_state"}:
            parts = []
            if item.get("status"):
                parts.append(item["status"].replace("_", " "))
            if item.get("zone"):
                parts.append("in " + zone(item["zone"]))
            if item.get("last_observed_zone"):
                parts.append("last observed in " + zone(item["last_observed_zone"]))
            prefix = "a different registered object " if kind == "other_state" else ""
            clauses.append(prefix + ", ".join(parts) + " at " + at)
        elif kind in {"first_seen", "registered"}:
            label = "first sighting" if kind == "first_seen" else "registration"
            place = " in " + zone(item["zone"]) if item.get("zone") else ""
            clauses.append(f"{label}{place} between {start} and {end}")
        elif kind in {"only_seen_in", "never_seen_in"}:
            label = (
                "all recorded sightings in"
                if kind == "only_seen_in"
                else "no recorded sightings in"
            )
            clauses.append(f"{label} {zone(item['zone'])} between {start} and {end}")
        else:
            label = f"{item['event']} events" if kind == "event" else "sightings"
            if item.get("previous_zone"):
                label += " from " + zone(item["previous_zone"])
            if item.get("zone"):
                label += (" outside " if item.get("outside_zone") else " in ") + zone(item["zone"])
            count = (
                f"at least {item.get('min_count', 1)}"
                if item.get("max_count") is None
                else f"{item.get('min_count', 1)}–{item['max_count']}"
            )
            clauses.append(f"{count} {label} between {start} and {end}")
    return "; ".join(clauses)


def validate_selection_terms(
    plan: SelectionPlan,
    text: str,
    regions: list[dict],
    named_objects=(),
    *,
    cutoff: int | None = None,
) -> None:
    """Reject invented literal names/regions; this is not full semantic verification."""
    question = text.casefold()
    # These claims are outside this memory schema, not aliases for a region move.
    # Keep this narrow: it is a capability boundary, not a full semantic verifier.
    claim_text = question
    for obj in named_objects:
        claim_text = claim_text.replace(obj.name.casefold(), "registered object")
    validate_history_quantifiers(plan.filters, claim_text, regions)
    explicit_zero = bool(
        re.search(
            r"\b(no|not|never|without|zero|none|unchanged|stationary|immobile|stayed|remained)\b"
            r"|\b\w+n['’]t\b|\b0\s+(times|events|occurrences)\b",
            claim_text,
        )
    )
    chinese_no_movement = bool(
        re.search(
            r"(?:从来没有|从未|没有|未曾|不曾|一直没有).{0,12}"
            r"(?:换过区域|发生区域变更|移动|变更位置|换过位置)"
            r"|(?:一直|始终)(?:待在|停在|位于).{0,12}(?:区域|位置)",
            claim_text,
        )
    )
    for predicate in plan.filters:
        if (
            isinstance(predicate, NamePredicate)
            and predicate.text.strip().casefold() == "registered"
            and re.search(r"\bregistered\s+(?:items?|objects?|things?|targets?)\b", question)
            and not any(find_literal_mention(obj.name, "registered") for obj in named_objects)
        ):
            raise ValueError(
                "Here 'registered' describes the inventory, not an object name. All candidate "
                "objects are already registered. Remove this name filter and preserve the "
                "requested state/history conditions."
            )
        if isinstance(predicate, EventPredicate) and predicate.max_count:
            count_requested = re.search(
                r"\b(?:once|twice|thrice|times|occurrences|exactly|at\s+(?:most|least)|"
                r"more\s+than|fewer\s+than|less\s+than)\b"
                r"|\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|single)\s+"
                r"(?:recorded\s+)?(?:moves?|disappearances?|appearances?|returns?|events?)\b"
                r"|次|回",
                claim_text,
            )
            if not count_requested:
                raise ValueError(
                    "A positive event max_count adds an unrequested upper bound. "
                    "A singular item does not mean an event happened exactly once. "
                    "For an event that happened, keep min_count=1 and omit max_count "
                    "unless the question explicitly requests an event count."
                )
        if not (
            isinstance(predicate, WindowPredicate)
            and predicate.min_count == 0
            and predicate.max_count == 0
        ):
            continue
        allowed_zero = explicit_zero or (
            isinstance(predicate, EventPredicate)
            and predicate.event == "moved"
            and chinese_no_movement
        )
        if not allowed_zero:
            raise ValueError(
                "A zero count asserts the event or sighting did NOT happen. The question has no "
                "explicit zero/negative condition. Preserve a positive occurrence with min_count "
                "at least 1, or return no filters if its meaning cannot be represented."
            )
    if plan.filters and re.search(
        r"\b(?:teleport\w*|stolen|stole|steal\w*|theft|intention\w*|motive\w*)\b"
        r"|\b(?:carried|carrying|hid|hidden)\b"
        r"|\b(?:inside|into|through)\s+(?:a|the)\s+(?:locked|closed|opaque)\b",
        claim_text,
    ):
        raise ValueError(
            "This asks for a physical cause, intent or hidden-interior fact that memory does not "
            "record. Return no filters; an appeared/moved/lost event cannot establish that claim."
        )

    def contains(term):
        return bool(find_literal_mention(question, term))

    if plan.filters and all(isinstance(p, NamePredicate) for p in plan.filters):
        # A literal name cannot satisfy a following recorded-condition clause.
        # This is a narrow omission check, not a general natural-language proof.
        for obj in named_objects:
            for term in (obj.name, obj.label if obj.label != "custom" else ""):
                if not term:
                    continue
                mention = find_literal_mention(question, term)
                if mention and re.search(
                    r"\b(that|which|whose|with|without|never|only|currently|first|before|after)\b"
                    r"|从来没有|没有发生|一直待在|始终待在|只在|从未|第一次",
                    question[mention.end() :],
                ):
                    raise ValueError(
                        "A name-only filter drops the object's additional description. "
                        "Represent all recorded conditions or return no filters."
                    )

    for predicate in plan.filters:
        if isinstance(predicate, NamePredicate) and not contains(predicate.text):
            raise ValueError(
                "A name filter must copy words from the question; do not invent a target."
            )
        for key in ("zone", "previous_zone", "last_observed_zone"):
            value = getattr(predicate, key, None)
            if value is None:
                continue
            region = next((r for r in regions if r["id"] == value), None)
            if region is None:
                raise ValueError("Use a region ID from the supplied inventory.")
            aliases = [value, region["name"]]
            if value == "center":
                aliases += ["middle", "centre"]
            if not any(contains(alias) for alias in aliases):
                if re.search(
                    r"\b(?:stayed|remained|never changed|no recorded move|zero moves|"
                    r"never crossed|never switched|no zone-to-zone transition)\b"
                    r"|一直待在|始终待在|从来没有换过区域|没有发生区域变更",
                    question,
                ):
                    raise ValueError(
                        "That region is not named. For an object with no recorded region "
                        "transition, use event moved with min_count=0,max_count=0 instead "
                        "of inventing a zone."
                    )
                raise ValueError(
                    "The question does not mention that region. Remove invented qualifiers."
                )
    if cutoff is not None:
        validate_time_filters(plan.filters, claim_text, cutoff)


def state_matches(memory, run, video, obj, cutoff: int, predicate: StateFields) -> bool:
    at = cutoff if predicate.at_ms is None else predicate.at_ms
    if obj.registered_at_ms > at:
        return False
    state = memory.object_state(run, video, obj, at)
    status_matches = (
        state.status != Visibility.VISIBLE
        if predicate.status == "not_visible"
        else predicate.status is None or state.status.value == predicate.status
    )
    # Region alone means currently visible there, never a stale last-seen region.
    zone_matches = predicate.zone is None or (
        (state.zone if predicate.status == "last_seen" else state.current_zone) == predicate.zone
    )
    return (
        status_matches
        and zone_matches
        and (predicate.last_observed_zone is None or state.zone == predicate.last_observed_zone)
    )


def find_objects(
    memory, run, video, objects, cutoff: int, arguments: dict, *, scope_ids: set[str] | None = None
) -> dict:
    request = SelectionRequest.model_validate(arguments)
    zones = {r["id"] for r in run.regions}
    for predicate in request.filters:
        for key in ("at_ms", "start_ms", "end_ms"):
            value = getattr(predicate, key, None)
            if value is not None and value > cutoff:
                raise ValueError("Selection cannot inspect times after the question cutoff.")
        for key in ("zone", "previous_zone", "last_observed_zone"):
            value = getattr(predicate, key, None)
            if value is not None and value not in zones:
                raise ValueError(f"Unknown region {value!r}; use a region ID from this run.")
    memory.catalog.require_source(video)
    registered = [o for o in objects if o.registered_at_ms <= cutoff]
    eligible = [o for o in registered if scope_ids is None or o.id in scope_ids]
    name_ids = {}
    for predicate in request.filters:
        if not isinstance(predicate, NamePredicate):
            continue
        needle = predicate.text.strip().casefold()
        literal = {
            obj.id
            for obj in registered
            if needle
            and (
                find_literal_mention(obj.name, needle)
                or (obj.label != "custom" and needle == obj.label.casefold())
            )
        }
        if not literal:
            for alias in toolbox_name_aliases(needle):
                literal = {obj.id for obj in registered if find_literal_mention(obj.name, alias)}
                if literal:
                    break
        name_ids[predicate.text] = literal
    with memory.db.session() as session:
        observations = session.scalars(
            select(Observation)
            .where(
                Observation.run_id == run.id,
                Observation.at_ms <= cutoff,
                Observation.visible.is_(True),
            )
            .order_by(Observation.at_ms, Observation.id)
        ).all()
    events = memory.history(run.id, cutoff)
    matches = []
    for obj in eligible:
        positives = [o for o in observations if o.object_id == obj.id]
        history = [e for e in events if e["object_id"] == obj.id]
        satisfied = True
        for predicate in request.filters:
            if isinstance(predicate, NamePredicate):
                matched = obj.id in name_ids[predicate.text]
            elif isinstance(predicate, StatePredicate):
                matched = state_matches(memory, run, video, obj, cutoff, predicate)
            elif isinstance(predicate, OtherStatePredicate):
                matched = any(
                    other.id != obj.id
                    and state_matches(memory, run, video, other, cutoff, predicate)
                    for other in registered
                )
            elif isinstance(predicate, FirstSeenPredicate):
                end = cutoff if predicate.end_ms is None else predicate.end_ms
                matched = bool(
                    positives
                    and predicate.start_ms <= positives[0].at_ms <= end
                    and (predicate.zone is None or positives[0].zone == predicate.zone)
                )
            elif isinstance(predicate, RegionHistoryPredicate):
                end = cutoff if predicate.end_ms is None else predicate.end_ms
                samples = [o for o in positives if predicate.start_ms <= o.at_ms <= end]
                matched = (
                    not any(o.zone == predicate.zone for o in samples)
                    if predicate.kind == "never_seen_in"
                    else bool(samples) and all(o.zone == predicate.zone for o in samples)
                )
            elif isinstance(predicate, RegisteredPredicate):
                end = cutoff if predicate.end_ms is None else predicate.end_ms
                matched = predicate.start_ms <= obj.registered_at_ms <= end
            else:
                end = cutoff if predicate.end_ms is None else predicate.end_ms
                if isinstance(predicate, SeenPredicate):
                    count = sum(
                        predicate.start_ms <= o.at_ms <= end
                        and (
                            (o.zone is not None and o.zone != predicate.zone)
                            if predicate.outside_zone
                            else o.zone == predicate.zone
                        )
                        for o in positives
                    )
                else:
                    count = sum(
                        predicate.start_ms <= e["at_ms"] <= end
                        and e["kind"] == predicate.event
                        and (predicate.zone is None or e["zone"] == predicate.zone)
                        and (
                            predicate.previous_zone is None
                            or e["previous_zone"] == predicate.previous_zone
                        )
                        for e in history
                    )
                matched = count >= predicate.min_count and (
                    predicate.max_count is None or count <= predicate.max_count
                )
            if not matched:
                satisfied = False
                break
        if satisfied:
            matches.append({"object_id": obj.id, "name": obj.name})
    return {
        "filters": request.model_dump()["filters"],
        "cutoff_ms": cutoff,
        "eligible_objects": len(eligible),
        "count": len(matches),
        "resolution": "unique" if len(matches) == 1 else "ambiguous" if matches else "none",
        "matches": matches,
        "scope": "Matches recorded observations and events only. Missing detections are not proof of physical absence.",
    }
