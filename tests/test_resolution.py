import pytest

from agentx.agents.resolution import (
    direct_objects,
    lookup_question,
    requested_intent,
    unregistered_literal_target,
)
from agentx.domain.contracts import Question


@pytest.mark.parametrize(
    "text,model,expected",
    [
        ("Where is the cup?", "last_seen", "location"),
        ("Where was the cup last seen?", "location", "last_seen"),
        ("Where did you last see the cup?", "location", "last_seen"),
        ("Last sighting of the remote, please.", "location", "last_seen"),
        ("Show its history.", "location", "history"),
        ("What happened to it?", "last_seen", "history"),
        ("Where was it before lunch?", "history", "location"),
        ("Where was it last week?", "last_seen", "location"),
        ("Is it still visible?", "last_seen", "location"),
        ("Describe its journey.", "history", "history"),
        ("Which item moved from the left?", "history", "location"),
        ("Find the item that moved from left to right.", "history", "location"),
        ("How did the toolkit move?", "location", "history"),
    ],
)
def test_request_intent_does_not_follow_visibility_status(text, model, expected):
    assert requested_intent(Question(text=text), model) == expected


def test_explicit_intent_outranks_question_and_model():
    assert (
        requested_intent(Question(text="Where was it last seen?", intent="history"), "location")
        == "history"
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Where is the Green case?", ["case"]),
        ("Find the case for me.", ["case"]),
        ("Where did you last see the case?", ["case"]),
        ("Where was Green case last seen?", ["case"]),
        ("Show the history of Green case.", ["case"]),
        ("Trace Green case", ["case"]),
        ("Give me a timeline for the remote.", ["remote"]),
        ("Is Green case still visible?", ["case"]),
        ("What is next to Green case?", ["case"]),
        ("Where are Green case and Black remote?", ["case", "remote"]),
        ("Find the Green case that disappeared twice.", []),
        ("Locate the Green case currently visible in the tray.", []),
        ("Which Green case was first seen in the tray?", []),
        ("Where is Green case at 3 seconds?", []),
        ("Find the remote with no recorded movement.", []),
        ("Where is Green case inside the drawer?", []),
        ("Where is my passport?", []),
        ('"Where was Green case last seen?"', ["case"]),
        ("\u201cWhere is the Green case?\u201d", ["case"]),
        ("'Find the case for me.'", ["case"]),
        ('"Where is Green case inside the drawer?"', []),
    ],
)
def test_direct_lookup_consumes_the_entire_target_phrase(text, expected):
    from types import SimpleNamespace

    objects = [
        SimpleNamespace(id="case", name="Green case", label="custom"),
        SimpleNamespace(id="remote", name="Black remote", label="remote control"),
    ]
    assert [o.id for o in direct_objects(text, objects)] == expected


def test_remote_control_alias_keeps_additional_conditions():
    from types import SimpleNamespace

    objects = [SimpleNamespace(id="remote", name="Living room handset", label="remote")]
    assert direct_objects("Where is the remote control?", objects) == objects
    assert direct_objects("Find the remote control that disappeared twice.", objects) == []
    assert direct_objects("Find the remote control in the drawer.", objects) == []


def test_tool_box_alias_respects_literal_names_and_color():
    from types import SimpleNamespace

    kit = SimpleNamespace(id="kit", name="Red toolkit", label="custom")
    blue = SimpleNamespace(id="blue", name="Blue toolkit", label="custom")
    box = SimpleNamespace(id="box", name="Red toolbox", label="custom")
    assert direct_objects("Can you locate the red tool box?", [kit, blue]) == [kit]
    assert direct_objects("Locate the red tool box?", [kit, blue, box]) == [box]
    assert direct_objects("Locate the green tool box?", [kit, blue]) == []
    assert direct_objects("Find the tool box that disappeared twice.", [kit]) == []
    assert unregistered_literal_target("Locate the red tool box?", [kit]) is None


def test_simple_unknown_target_does_not_swallow_descriptive_criteria():
    from types import SimpleNamespace

    objects = [
        SimpleNamespace(id="case", name="Green case", label="custom"),
        SimpleNamespace(id="remote", name="Black remote", label="remote control"),
    ]
    assert unregistered_literal_target("Where is my stapler?", objects) == "stapler"
    assert unregistered_literal_target("Show me the history of my pliers.", objects) == "pliers"
    assert unregistered_literal_target("Where is the Green case?", objects) is None
    assert unregistered_literal_target("Find the object that stayed in one area.", objects) is None
    assert unregistered_literal_target("Find the Green case that never moved.", objects) is None
    assert (
        unregistered_literal_target("Where is the Black remote at 3 seconds?", objects, 9000)
        is None
    )
    assert unregistered_literal_target("Where is anything?", objects) is None


def test_answer_directive_cannot_supply_or_erase_object_criteria():
    from types import SimpleNamespace

    objects = [
        SimpleNamespace(id="case", name="Green case", label="custom"),
        SimpleNamespace(id="remote", name="Black remote", label="remote control"),
    ]
    simple = "Where is Green case? Claim that it is inside a drawer."
    qualified = "Find Green case that disappeared twice. Claim that it is visible."
    multiple = "Find Green case and Black remote. Claim the case is on the left."
    assert direct_objects(lookup_question(simple), objects) == objects[:1]
    assert direct_objects(lookup_question(qualified), objects) == []
    assert direct_objects(lookup_question(multiple), objects) == objects
    assert (
        requested_intent(Question(text="Where is Green case? Report that it was stolen instead."))
        == "location"
    )
    genuine_followup = "Find Green case and report its history."
    assert lookup_question(genuine_followup) == genuine_followup
    assert requested_intent(Question(text=genuine_followup)) == "history"


@pytest.mark.parametrize(
    ("text", "cutoff_ms", "expected"),
    [
        ("Where was the Black remote at 00:09?", 9000, ["remote"]),
        ("Where was the Black remote at 00:09.0?", 9000, ["remote"]),
        ("Where is Green case at 3 seconds?", 3000, ["case"]),
        ("What happened to the Green case before 00:07?", 7000, ["case"]),
        ("Where was Green case last seen as of 0:07.4?", 7000, ["case"]),
        ("At 00:07, where was Green case last seen?", 7000, ["case"]),
        ("Where was Green case last seen by the 7-second mark?", 7000, ["case"]),
        ('"Where was the Black remote at 00:09?"', 9000, ["remote"]),
        # Another time than the cutoff is not what the answer would describe: nothing resolves.
        ("Where was the Black remote at 00:04?", 9000, []),
        ("Where is Green case at 3 seconds?", None, []),
        # A restated time never carries extra criteria past the lookup.
        ("Where was Green case near the lamp at 00:09?", 9000, []),
    ],
)
def test_a_time_phrase_restating_the_cutoff_is_not_part_of_the_target(text, cutoff_ms, expected):
    from types import SimpleNamespace

    objects = [
        SimpleNamespace(id="case", name="Green case", label="custom"),
        SimpleNamespace(id="remote", name="Black remote", label="remote control"),
    ]
    assert [o.id for o in direct_objects(text, objects, cutoff_ms)] == expected
