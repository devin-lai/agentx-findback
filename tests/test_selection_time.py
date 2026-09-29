import pytest

from agentx.services.selection import SelectionPlan
from agentx.services.selection_time import TimeConstraint, explicit_times, validate_time_filters


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Visible at 4.4 seconds", [TimeConstraint("at", 4400, 4400)]),
        ("Visible at 00:04.4", [TimeConstraint("at", 4400, 4400)]),
        ("Returned before the seventh second", [TimeConstraint("before", end=6999)]),
        ("Visible after the 15-second mark", [TimeConstraint("after", start=15001)]),
        ("Moved in the first 2.5 seconds", [TimeConstraint("through", end=2500)]),
        ("No region changes by 11.4 seconds", [TimeConstraint("through", end=11400)]),
        ("Only seen in Center area by the 11-second mark", [TimeConstraint("through", end=11000)]),
        ("Registered between two and four seconds", [TimeConstraint("range", 2000, 4000)]),
        ("Registered from 2s to 4s", [TimeConstraint("range", 2000, 4000)]),
        ("First observed before the one-second mark", [TimeConstraint("before", end=999)]),
        ("Moved after 1500ms", [TimeConstraint("after", start=1501)]),
        ("Visible at the 3rd second", [TimeConstraint("at", 3000, 3000)]),
        ("Visible at 1.5 minutes", [TimeConstraint("at", 90000, 90000)]),
        ("Find Book 15", []),
        ("Visible at the beginning of the recording", [TimeConstraint("at", 0, 0)]),
        ("In the opening frame", [TimeConstraint("at", 0, 0)]),
        ("At the very start", [TimeConstraint("at", 0, 0)]),
        ("When the clip started", [TimeConstraint("at", 0, 0)]),
        ("First observed before one second", [TimeConstraint("before", end=999)]),
        ("At the start of an occlusion", []),
        ("In the first frame after reappearance", []),
        ("Which ones were in the left bin?", []),
        ("The item that moved seconds after appearing", []),
        ("Tens of objects on a 2 m desk", []),
        ("Which object moved in the first minute?", [TimeConstraint("through", end=60000)]),
        ("Visible at 5s", [TimeConstraint("at", 5000, 5000)]),
    ],
)
def test_explicit_video_time_grammar(text, expected):
    assert explicit_times(text) == expected


def test_dropped_bounds_wrong_conversions_and_future_repair_cannot_select_an_object():
    def check(text, filters, cutoff=9000):
        validate_time_filters(
            SelectionPlan(intent="location", filters=filters).filters, text, cutoff
        )

    with pytest.raises(ValueError, match="dropped or changed"):
        check("Which item moved in the first 2.5 seconds?", [dict(kind="event", event="moved")])
    check(
        "Which item moved in the first 2.5 seconds?",
        [dict(kind="event", event="moved", end_ms=2500)],
    )
    with pytest.raises(ValueError, match="dropped or changed"):
        check(
            "Find the item visible at 4.4 seconds",
            [dict(kind="state", status="visible", at_ms=4000)],
        )
    with pytest.raises(ValueError, match="after the cutoff"):
        check(
            "Find the item visible after the 15-second mark", [dict(kind="state", status="visible")]
        )
    check("Find the item visible after the 15-second mark", [])
    check(
        "Find what returned before the seventh second",
        [dict(kind="event", event="reappeared", end_ms=6600)],
        6600,
    )
    with pytest.raises(ValueError, match="dropped or changed"):
        check("Registered between two and four seconds", [dict(kind="registered", end_ms=4000)])
    check(
        "Registered between two and four seconds",
        [dict(kind="registered", start_ms=2000, end_ms=4000)],
    )


@pytest.mark.parametrize(
    "text",
    [
        "In the right area at the beginning of the recording",
        "In the center area in the opening frame",
        "In the left area when the video began",
    ],
)
def test_recording_start_anchor_cannot_be_dropped_or_shifted(text):
    for fields in ({}, {"at_ms": 5000}, {"at_ms": 1}):
        filters = SelectionPlan(
            intent="location", filters=[dict(kind="state", zone="right", **fields)]
        ).filters
        with pytest.raises(ValueError, match="dropped or changed"):
            validate_time_filters(filters, text, 5000)
    correct = SelectionPlan(
        intent="location", filters=[dict(kind="state", zone="right", at_ms=0)]
    ).filters
    validate_time_filters(correct, text, 5000)


def test_effective_defaults_are_checked_without_weakening_earlier_bounds():
    state = SelectionPlan(intent="location", filters=[dict(kind="state", status="visible")])
    validate_time_filters(state.filters, "Visible in the first frame", 0)
    validate_time_filters(state.filters, "Visible at five seconds", 5000)
    with pytest.raises(ValueError, match="dropped or changed"):
        validate_time_filters(state.filters, "Visible in the first frame", 5000)
    events = SelectionPlan(
        intent="location", filters=[dict(kind="event", event="lost", min_count=0, max_count=0)]
    )
    validate_time_filters(events.filters, "No disappearances through five seconds", 5000)
    with pytest.raises(ValueError, match="dropped or changed"):
        validate_time_filters(events.filters, "No disappearances before five seconds", 5000)
    with pytest.raises(ValueError, match="dropped or changed"):
        validate_time_filters(events.filters, "No disappearances through two seconds", 5000)


def test_recording_start_does_not_replace_a_late_objects_first_sighting():
    first = SelectionPlan(intent="location", filters=[dict(kind="first_seen", zone="left")])
    with pytest.raises(ValueError, match="dropped or changed"):
        validate_time_filters(first.filters, "Visible in the left area at the start", 5000)
    validate_time_filters(first.filters, "First observed in the left area", 5000)


def test_literal_object_name_does_not_create_a_time_anchor():
    from types import SimpleNamespace

    from agentx.services.selection import validate_selection_terms

    plan = SelectionPlan(intent="location", filters=[dict(kind="event", event="lost")])
    validate_selection_terms(
        plan,
        "Find At the beginning which disappeared.",
        [],
        [SimpleNamespace(name="At the beginning", label="custom")],
        cutoff=5000,
    )
