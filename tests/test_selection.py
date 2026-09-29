import pytest
from test_agent import configure, planner

from agentx.domain.contracts import Question
from agentx.services.selection import find_objects


@pytest.mark.parametrize(
    "text,kind",
    [
        ("Where is the object that never left the center area?", "only_seen_in"),
        ("Find what was never observed outside the middle.", "only_seen_in"),
        ("Find what was only ever seen in the centre.", "only_seen_in"),
        ("Find what was always in the Center area.", "only_seen_in"),
        ("Find what was never seen inside the center.", "never_seen_in"),
        ("Find what has never been in the center.", "never_seen_in"),
    ],
)
def test_region_history_quantifier_cannot_be_dropped_or_inverted(text, kind):
    from agentx.services.selection import SelectionPlan, validate_selection_terms

    regions = [dict(id="center", name="Center area")]
    for wrong in ("seen", "state", "never_seen_in" if kind == "only_seen_in" else "only_seen_in"):
        plan = SelectionPlan(intent="location", filters=[dict(kind=wrong, zone="center")])
        with pytest.raises(ValueError, match="whole-history quantifier"):
            validate_selection_terms(plan, text, regions)
    validate_selection_terms(
        SelectionPlan(intent="location", filters=[dict(kind=kind, zone="center")]), text, regions
    )
    validate_selection_terms(SelectionPlan(intent="location", filters=[]), text, regions)


@pytest.mark.parametrize(
    "text",
    [
        "Find the item in the center.",
        "Find the only item in the center.",
        "Find the item that was not always in the center.",
        "Find the item that wasn't always in the center.",
        "Find the item that never left any area.",
    ],
)
def test_region_history_guard_does_not_invent_a_quantifier(text):
    from agentx.services.selection import SelectionPlan
    from agentx.services.selection_history import validate_history_quantifiers

    filters = SelectionPlan(intent="location", filters=[dict(kind="seen", zone="center")]).filters
    validate_history_quantifiers(filters, text.casefold(), [dict(id="center", name="Center area")])


def test_region_history_guard_preserves_custom_names_and_windows():
    from agentx.services.selection import SelectionPlan, validate_selection_terms

    regions = [dict(id="desk", name="Workbench"), dict(id="middle", name="中央区域")]
    for text, zone, start in [
        ("Find what never left the Workbench after 9 seconds.", "desk", 9001),
        ("找出从未离开中央区域的物品。", "middle", 0),
    ]:
        plan = SelectionPlan(
            intent="location", filters=[dict(kind="only_seen_in", zone=zone, start_ms=start)]
        )
        validate_selection_terms(plan, text, regions, cutoff=11000)
        plan.filters = SelectionPlan(
            intent="location", filters=[dict(kind="seen", zone=zone)]
        ).filters
        with pytest.raises(ValueError, match="whole-history quantifier"):
            validate_selection_terms(plan, text, regions, cutoff=11000)


def test_never_left_is_repaired_before_memory_selects_an_object(app, indexed):
    _, run = indexed
    s = configure(app)
    s.providers._chat = planner(
        [
            dict(intent="location", filters=[dict(kind="seen", zone="center")]),
            dict(intent="location", filters=[dict(kind="only_seen_in", zone="center")]),
        ]
    )
    answer = s.workflow.answer(
        run["id"],
        Question(text="Where is the object that never left the center area?", at_ms=11000),
    )
    assert answer["planner"] == "agent"
    assert [o["name"] for o in answer["states"]] == ["Blue remote"]
    assert any("whole-history quantifier" in t.get("result", "") for t in answer["tools"])
    assert all(e["at_ms"] <= 11000 for e in answer["evidence"])


def test_opening_frame_first_sighting_error_gets_specific_repair(app, indexed):
    _, run = indexed
    s = configure(app)
    s.providers._chat = planner(
        [
            dict(intent="location", filters=[dict(kind="first_seen", zone="center", start_ms=0)]),
            dict(intent="location", filters=[dict(kind="state", zone="center", at_ms=0)]),
        ]
    )
    answer = s.workflow.answer(
        run["id"],
        Question(
            text="Locate whichever object occupied the center area in the opening frame.",
            at_ms=9000,
        ),
    )
    assert answer["planner"] == "agent"
    assert [o["name"] for o in answer["states"]] == ["Blue remote"]
    assert any("first_seen with start_ms=0 alone" in t.get("result", "") for t in answer["tools"])
    executed = [t for t in answer["tools"] if isinstance(t.get("arguments"), list)]
    assert len(executed) == 1 and executed[0]["arguments"][0]["at_ms"] == 0


@pytest.mark.parametrize(
    "text",
    [
        "Find the only item in the center.",
        "Find the only registered object currently inside the middle area.",
        "Locate the only target within the centre region!",
    ],
)
def test_only_item_does_not_mean_only_ever_seen_in_region(text):
    from agentx.services.selection import SelectionPlan, validate_selection_terms

    regions = [dict(id="center", name="Center area")]
    plan = SelectionPlan(intent="location", filters=[dict(kind="only_seen_in", zone="center")])
    with pytest.raises(ValueError, match="modifies the number"):
        validate_selection_terms(plan, text, regions)
    plan = SelectionPlan(intent="location", filters=[dict(kind="state", zone="center")])
    validate_selection_terms(plan, text, regions)


def test_only_item_and_after_window_preserve_ambiguity(app, indexed):
    _, run = indexed
    s = configure(app)
    for text, wrong, repaired, feedback in [
        (
            "Find the only item in the center.",
            dict(kind="only_seen_in", zone="center"),
            dict(kind="state", zone="center"),
            "modifies the number",
        ),
        (
            "Find the item that never left the center after 9 seconds.",
            dict(kind="only_seen_in", zone="center", end_ms=9000),
            dict(kind="only_seen_in", zone="center", start_ms=9001),
            "LOWER bound",
        ),
    ]:
        s.providers._chat = planner(
            [
                dict(intent="location", filters=[wrong]),
                dict(intent="location", filters=[repaired]),
            ]
        )
        answer = s.workflow.answer(run["id"], Question(text=text, at_ms=11000))
        assert answer["planner"] == "agent"
        assert not answer["states"] and not answer["evidence"]
        assert "Several registered objects match" in answer["answer"]
        assert any(feedback in t.get("result", "") for t in answer["tools"])


@pytest.mark.parametrize(
    "text,allowed",
    [
        ("Show the history of the item that disappeared.", False),
        ("Find the one item that disappeared before 7 seconds.", False),
        ("Find the item that disappeared exactly once.", True),
        ("Find the item with one recorded disappearance.", True),
        ("Find the item that disappeared at most once.", True),
        ("Find the item that disappeared twice.", True),
        ("找出消失过一次的物品。", True),
    ],
)
def test_positive_event_count_requires_a_count_request(text, allowed):
    from agentx.services.selection import SelectionPlan, validate_selection_terms

    plan = SelectionPlan.model_validate(
        dict(intent="history", filters=[dict(kind="event", event="lost", max_count=1)])
    )
    if allowed:
        validate_selection_terms(plan, text, [])
    else:
        with pytest.raises(ValueError, match="unrequested upper bound"):
            validate_selection_terms(plan, text, [])
    # The general occurrence condition keeps every matching event count eligible.
    plan.filters[0].max_count = None
    validate_selection_terms(plan, text, [])


def test_agent_repairs_invented_exact_count_before_memory_selection(app, indexed):
    _, run = indexed
    s = configure(app)
    s.providers._chat = planner(
        [
            dict(intent="history", filters=[dict(kind="event", event="lost", max_count=1)]),
            dict(intent="history", filters=[dict(kind="event", event="lost")]),
        ]
    )
    answer = s.workflow.answer(
        run["id"], Question(text="Show the history of the item that disappeared.", at_ms=7000)
    )
    assert answer["planner"] == "agent" and answer["states"][0]["name"] == "Red toolkit"
    assert any("unrequested upper bound" in t.get("result", "") for t in answer["tools"])


def test_registered_inventory_modifier_is_not_a_name_filter(app, indexed):
    _, run = indexed
    s = configure(app)
    events = [dict(kind="event", event="lost"), dict(kind="event", event="reappeared")]
    s.providers._chat = planner(
        [
            dict(intent="location", filters=[dict(kind="name", text="registered"), *events]),
            dict(intent="location", filters=events),
        ]
    )
    answer = s.workflow.answer(
        run["id"],
        Question(
            text="Locate the registered thing which vanished and subsequently came back.",
            at_ms=9000,
        ),
    )
    assert answer["planner"] == "agent"
    assert [o["name"] for o in answer["states"]] == ["Red toolkit"]
    assert any("describes the inventory" in t.get("result", "") for t in answer["tools"])


def test_registered_guard_preserves_an_explicit_registered_name():
    from types import SimpleNamespace

    from agentx.services.selection import SelectionPlan, validate_selection_terms

    plan = SelectionPlan(
        intent="location",
        filters=[dict(kind="name", text="registered"), dict(kind="event", event="lost")],
    )
    validate_selection_terms(
        plan,
        "Find the Registered thing that disappeared.",
        [],
        [SimpleNamespace(name="Registered thing", label="custom")],
    )


def test_time_repair_preserves_registration_and_exclusive_history_bound(app, indexed):
    _, run = indexed
    s = configure(app)
    for text, wrong, repaired, expected_names, feedback in [
        (
            "Locate the item that was registered at the very beginning.",
            dict(kind="registered", start_ms=0),
            dict(kind="registered", start_ms=0, end_ms=0),
            [],
            "start_ms=0 AND end_ms=0",
        ),
        (
            "Find the item that never left the center before 8 seconds.",
            dict(kind="only_seen_in", zone="center", end_ms=8000),
            dict(kind="only_seen_in", zone="center", end_ms=7999),
            ["Blue remote"],
            "BEFORE excludes",
        ),
    ]:
        s.providers._chat = planner(
            [
                dict(intent="location", filters=[wrong]),
                dict(intent="location", filters=[repaired]),
            ]
        )
        answer = s.workflow.answer(run["id"], Question(text=text, at_ms=11000))
        assert answer["planner"] == "agent"
        assert [o["name"] for o in answer["states"]] == expected_names
        assert any(feedback in t.get("result", "") for t in answer["tools"])
        if not expected_names:
            assert "Several registered objects match" in answer["answer"]
            assert not answer["evidence"]


def test_selection_checks_all_objects_current_visibility_and_historical_time(app, indexed):
    _, run = indexed
    memory = app.state.services.memory
    context = memory.context(run["id"], 11000)
    cases = [
        ([dict(kind="state", last_observed_zone="center")], {"Blue remote", "Red toolkit"}),
        ([dict(kind="name", text="toolkit")], {"Red toolkit"}),
        ([dict(kind="name", text="red tool box")], {"Red toolkit"}),
        ([dict(kind="name", text="blue tool box")], set()),
        ([dict(kind="name", text="tool")], set()),
        ([dict(kind="name", text=" ")], set()),
        ([dict(kind="state", status="visible", zone="center")], {"Blue remote", "Red toolkit"}),
        ([dict(kind="state", zone="right", at_ms=7000)], set()),
        ([dict(kind="state", status="last_seen", zone="right", at_ms=7000)], {"Red toolkit"}),
        ([dict(kind="state", status="visible", zone="left", at_ms=5000)], set()),
        ([dict(kind="state", status="visible", zone="left", at_ms=2000)], {"Red toolkit"}),
        ([dict(kind="first_seen", zone="left")], {"Red toolkit"}),
        ([dict(kind="first_seen", zone="right")], set()),
        ([dict(kind="first_seen", end_ms=999)], {"Blue remote", "Red toolkit"}),
        ([dict(kind="first_seen", start_ms=1000)], set()),
        ([dict(kind="only_seen_in", zone="center")], {"Blue remote"}),
        ([dict(kind="never_seen_in", zone="right")], {"Blue remote"}),
        ([dict(kind="only_seen_in", zone="center", start_ms=9000)], {"Blue remote", "Red toolkit"}),
        ([dict(kind="event", event="lost", min_count=2)], set()),
        (
            [
                dict(kind="event", event="reappeared", zone="center"),
                dict(kind="event", event="lost"),
            ],
            {"Red toolkit"},
        ),
        ([dict(kind="event", event="moved", zone="right", end_ms=1999)], set()),
        ([dict(kind="seen", zone="right", min_count=0, max_count=0)], {"Blue remote"}),
        (
            [dict(kind="seen", zone="center", outside_zone=True, min_count=0, max_count=0)],
            {"Blue remote"},
        ),
        ([dict(kind="registered", start_ms=5001)], set()),
    ]
    for filters, expected in cases:
        result = find_objects(memory, *context, {"filters": filters})
        assert {o["name"] for o in result["matches"]} == expected, filters
        assert result["count"] == len(expected)
        assert result["resolution"] == (
            "unique" if len(expected) == 1 else "ambiguous" if expected else "none"
        )
    early = memory.context(run["id"], 2000)
    assert (
        find_objects(
            memory, *early, {"filters": [dict(kind="seen", zone="right", min_count=0, max_count=0)]}
        )["count"]
        == 2
    )
    assert (
        find_objects(memory, *early, {"filters": [dict(kind="event", event="lost")]})["count"] == 0
    )


def test_selection_rejects_future_unknown_unbounded_or_untyped_predicates(app, indexed):
    _, run = indexed
    memory = app.state.services.memory
    context = memory.context(run["id"], 7000)
    bad_filters = [
        [],
        [dict(kind="state")],
        [dict(kind="state", status="visible", at_ms=8000)],
        [dict(kind="state", zone="invented")],
        [dict(kind="event", event="lost", start_ms=8000)],
        [dict(kind="event", event="lost", end_ms=8000)],
        [dict(kind="first_seen", end_ms=8000)],
        [dict(kind="first_seen", start_ms=6000, end_ms=5000)],
        [dict(kind="only_seen_in", zone="center", end_ms=8000)],
        [dict(kind="never_seen_in", zone="center", start_ms=6000, end_ms=5000)],
        [dict(kind="registered", start_ms=6000, end_ms=5000)],
        [dict(kind="event", event="lost", min_count=0)],
        [dict(kind="event", event="lost", min_count=2, max_count=1)],
        [dict(kind="state", status="visible", at_ms=True)],
        [dict(kind="state", zone="left", object_id="forced")],
        [dict(kind="state", status="visible")] * 9,
    ]
    for filters in bad_filters:
        with pytest.raises(ValueError):
            find_objects(memory, *context, {"filters": filters})


@pytest.mark.parametrize(
    "filters,expected",
    [
        ([dict(kind="state", status="visible", zone="center")], None),
        ([dict(kind="state", status="visible", zone="right")], None),
        ([dict(kind="first_seen", zone="left")], "Red toolkit"),
    ],
)
def test_agent_selection_derives_zero_many_or_unique_matches(app, indexed, filters, expected):
    _, run = indexed
    s = configure(app)
    s.providers._chat = planner([dict(intent="location", filters=filters)])
    text = (
        "Find the item that started in the left area."
        if expected
        else f"Find the item currently visible in the {filters[0]['zone']} area."
    )
    answer = s.workflow.answer(run["id"], Question(text=text, at_ms=11000))
    assert answer["planner"] == "agent"
    assert [o["name"] for o in answer["states"]] == ([expected] if expected else [])
    if expected is None:
        assert not answer["evidence"]
    if filters[0].get("zone") == "center":
        assert "Several registered objects match" in answer["answer"]


def test_agent_requires_verified_indirect_selection_even_without_skills(app, indexed):
    _, run = indexed
    s = configure(app)
    s.providers._chat = planner([dict(intent="location", filters=[])])
    answer = s.workflow.answer(
        run["id"], Question(text="Find the item that disappeared.", at_ms=7000, use_skills=False)
    )
    assert answer["planner"] == "agent"
    assert not answer["states"] and not answer["evidence"]
    assert answer["tools"][-1]["result"] == "No supported selection predicate; no object claim"


def test_future_registration_cannot_satisfy_absence_or_registration_selection(app, indexed):
    from agentx.storage.models import RegisteredObject

    _, run = indexed
    s = app.state.services
    with s.db.session.begin() as session:
        context = s.memory.context(run["id"], 7000)
        for obj in context[2]:
            if obj.name == "Blue remote":
                session.get(RegisteredObject, obj.id).registered_at_ms = 8000
    context = s.memory.context(run["id"], 7000)
    for filters in [[dict(kind="registered")], [dict(kind="state", status="not_visible")]]:
        result = find_objects(s.memory, *context, {"filters": filters})
        assert all(o["name"] != "Blue remote" for o in result["matches"])
    assert (
        find_objects(
            s.memory, *context, {"filters": [dict(kind="state", status="visible", at_ms=2000)]}
        )["count"]
        == 1
    )


def test_selection_cannot_invent_a_region_or_object_name(app, indexed):
    from agentx.services.selection import SelectionPlan, validate_selection_terms

    _, run = indexed
    s = configure(app)
    regions = s.memory.context(run["id"], 11000)[0].regions
    for predicate in [dict(kind="name", text="Blue remote"), dict(kind="first_seen", zone="left")]:
        plan = SelectionPlan(intent="location", filters=[predicate])
        with pytest.raises(ValueError):
            validate_selection_terms(plan, "Find the item that was visible at the start.", regions)
    s.providers._chat = planner(
        [
            dict(intent="location", filters=[dict(kind="first_seen", zone="left")]),
            dict(intent="location", filters=[dict(kind="state", status="visible", at_ms=0)]),
        ]
    )
    answer = s.workflow.answer(
        run["id"], Question(text="Find the item that was visible at the start.", at_ms=11000)
    )
    assert answer["planner"] == "agent" and not answer["states"]
    assert "Several registered objects match" in answer["answer"]
    assert any("does not mention" in tool["result"] for tool in answer["tools"])


def test_selection_rejects_unrequested_time_windows_but_accepts_explicit_times():
    from agentx.services.selection import SelectionPlan, validate_selection_terms

    filters = [dict(kind="event", event="moved", min_count=0, max_count=0, end_ms=1500)]
    plan = SelectionPlan(intent="history", filters=filters)
    with pytest.raises(ValueError, match="no numeric time"):
        validate_selection_terms(plan, "Show the item with no recorded movement.", [], cutoff=11000)
    for text in ["Show the item with no recorded movement through 1.5 seconds."]:
        validate_selection_terms(plan, text, [], cutoff=11000)
    plan.filters[0].end_ms = None
    validate_selection_terms(plan, "Show the item with no recorded movement.", [], cutoff=11000)
    plan.filters[0].end_ms = 11000
    validate_selection_terms(plan, "Show the item with no recorded movement.", [], cutoff=11000)
    first = SelectionPlan(
        intent="location", filters=[dict(kind="state", status="visible", at_ms=0)]
    )
    validate_selection_terms(first, "Find what was visible at the start.", [], cutoff=11000)
    with pytest.raises(ValueError, match="no numeric time"):
        validate_selection_terms(first, "Find what is visible.", [], cutoff=11000)


@pytest.mark.parametrize(
    "text",
    [
        "Find the object that teleported through a locked drawer.",
        "Find the object that was stolen.",
        "Find the object inside the closed cupboard.",
        "Find the object a person carried into a locked drawer.",
        "Locate the item someone hid under an opaque cover.",
    ],
)
def test_unrecorded_physical_claim_cannot_be_replaced_with_a_move(text):
    from agentx.services.selection import SelectionPlan, validate_selection_terms

    plan = SelectionPlan(intent="location", filters=[dict(kind="event", event="moved")])
    with pytest.raises(ValueError, match="does not record"):
        validate_selection_terms(plan, text, [], cutoff=11000)
    validate_selection_terms(SelectionPlan(intent="location", filters=[]), text, [], cutoff=11000)


def test_trace_describes_executed_bounds_instead_of_model_claims(app, indexed):
    _, run = indexed
    s = configure(app)
    s.providers._chat = planner(
        [
            dict(
                interpretation="This text incorrectly claims the event happened before one second.",
                intent="location",
                filters=[dict(kind="event", event="moved", zone="right")],
            )
        ]
    )
    answer = s.workflow.answer(
        run["id"], Question(text="Find the item that moved to the right.", at_ms=11000)
    )
    tool = next(t for t in answer["tools"] if t["name"] == "find_objects")
    assert "between 0.000 s and 11.000 s" in tool["interpretation"]
    assert "before one second" not in tool["interpretation"]
    assert "incorrectly claims" in tool["model_interpretation"]


def test_stayed_region_requires_positive_observations_and_no_unknown_regions(app, indexed):
    from agentx.storage.models import Observation

    _, run = indexed
    s = app.state.services
    context = s.memory.context(run["id"], 7000)
    # The toolkit is missing during this interval; absence cannot prove it stayed there.
    result = find_objects(
        s.memory,
        *context,
        {"filters": [dict(kind="only_seen_in", zone="right", start_ms=6500)]},
    )
    assert result["count"] == 0

    # A positive observation with no assigned region is not evidence for "only in center".
    with s.db.session.begin() as session:
        from sqlalchemy import select

        remote = next(o for o in context[2] if o.name == "Blue remote")
        observation = session.scalars(
            select(Observation).where(
                Observation.run_id == run["id"],
                Observation.object_id == remote.id,
                Observation.visible.is_(True),
            )
        ).first()
        observation.zone = None
    result = find_objects(
        s.memory, *context, {"filters": [dict(kind="only_seen_in", zone="center")]}
    )
    assert result["count"] == 0


def test_named_ambiguity_offers_authoritative_choices_on_agent_path(app, indexed):
    video, run = indexed
    s = configure(app)
    s.providers._chat = planner(
        [dict(action="final", answer="Choose one", object_id=video["objects"][0]["id"])]
    )
    answer = s.workflow.answer(
        run["id"], Question(text="Where are Red toolkit and Blue remote?", at_ms=7000)
    )
    assert answer["planner"] == "agent" and not answer["evidence"]
    matches = next(t["matching_objects"] for t in answer["tools"] if t["name"] == "resolve_object")
    assert {o["object_id"] for o in matches} == {o["id"] for o in video["objects"]}


def test_unregistered_literal_target_cannot_become_any_visible_object(app, indexed):
    _, run = indexed
    s = configure(app)

    def forbidden_model_call(*args, **kwargs):
        raise AssertionError("A missing literal target needs no model request.")

    s.providers._chat = forbidden_model_call
    for text, intent, target in (
        ("Where is my stapler?", "location", "stapler"),
        ("Show me the history of my pliers.", "history", "pliers"),
    ):
        answer = s.workflow.answer(run["id"], Question(text=text, at_ms=4200))
        assert answer["planner"] == "agent" and answer["planner_model"] is None
        assert answer["intent"] == intent and not answer["warnings"]
        assert not answer["states"] and not answer["evidence"]
        assert any(
            tool["name"] == "resolve_object" and tool["requested_target"] == target
            for tool in answer["tools"]
        )


def test_named_qualifier_is_checked_and_cannot_select_a_different_object(app, indexed):
    _, run = indexed
    s = configure(app)
    # Even when the model omits the name, code retains the literal registered scope.
    s.providers._chat = planner(
        [dict(intent="location", filters=[dict(kind="event", event="lost")])]
    )
    question = Question(text="Find the Blue remote that disappeared.", at_ms=11000)
    answer = s.workflow.answer(run["id"], question)
    assert answer["planner"] == "agent"
    assert not answer["states"] and not answer["evidence"]
    selection = next(t for t in answer["tools"] if t["name"] == "find_objects")
    assert len(selection["object_scope"]) == 1
    # A planner outage cannot discard the description and turn it into a name lookup.
    s.providers._chat = planner([])
    fallback = s.workflow.answer(run["id"], question)
    assert fallback["planner"] == "local"
    assert not fallback["states"] and not fallback["evidence"]
    assert fallback["warnings"]


def test_named_qualifier_allows_a_supported_match(app, indexed):
    _, run = indexed
    s = configure(app)
    s.providers._chat = planner(
        [dict(intent="location", filters=[dict(kind="event", event="lost")])]
    )
    answer = s.workflow.answer(
        run["id"], Question(text="Find the Red toolkit that disappeared.", at_ms=11000)
    )
    assert answer["planner"] == "agent"
    assert [o["name"] for o in answer["states"]] == ["Red toolkit"]
    assert answer["evidence"]
    assert [skill["name"] for skill in answer["skills"]] == ["compile-object-selection"]


def test_other_state_excludes_self_and_future_registrations(app, indexed):
    from agentx.storage.models import RegisteredObject

    _, run = indexed
    s = app.state.services
    context = s.memory.context(run["id"], 7000)
    filters = [
        dict(kind="state", status="visible", zone="center"),
        dict(kind="other_state", status="not_visible"),
    ]
    found = find_objects(s.memory, *context, {"filters": filters})
    assert [o["name"] for o in found["matches"]] == ["Blue remote"]
    # At nine seconds both are visible; the other-state condition must still hold.
    assert not find_objects(s.memory, *s.memory.context(run["id"], 9000), {"filters": filters})[
        "matches"
    ]
    assert not find_objects(
        s.memory,
        *context,
        {
            "filters": [
                dict(kind="state", status="not_visible"),
                dict(kind="other_state", status="not_visible"),
            ]
        },
    )["matches"]
    with pytest.raises(ValueError, match="after the question cutoff"):
        find_objects(
            s.memory,
            *context,
            {"filters": [dict(kind="other_state", status="not_visible", at_ms=8000)]},
        )
    with s.db.session.begin() as session:
        kit = next(o for o in context[2] if o.name == "Red toolkit")
        session.get(RegisteredObject, kit.id).registered_at_ms = 8000
    assert not find_objects(s.memory, *s.memory.context(run["id"], 7000), {"filters": filters})[
        "matches"
    ]


def test_model_cannot_drop_a_named_relative_clause_to_a_name_only_filter(app, indexed):
    _, run = indexed
    s = configure(app)
    s.providers._chat = planner(
        [
            dict(intent="location", filters=[dict(kind="name", text="Blue remote")]),
            dict(intent="location", filters=[dict(kind="event", event="lost", min_count=2)]),
        ]
    )
    answer = s.workflow.answer(
        run["id"], Question(text="Find the Blue remote that disappeared twice.", at_ms=11000)
    )
    assert answer["planner"] == "agent" and not answer["states"]
    assert any("name-only filter drops" in tool["result"] for tool in answer["tools"])


def test_shared_category_can_be_narrowed_by_recorded_conditions(app, indexed):
    from agentx.storage.models import RegisteredObject

    _, run = indexed
    s = configure(app)
    with s.db.session.begin() as session:
        for obj in s.memory.context(run["id"], 7000)[2]:
            session.get(RegisteredObject, obj.id).label = "cup"
    s.providers._chat = planner(
        [dict(intent="location", filters=[dict(kind="state", zone="center", status="visible")])]
    )
    answer = s.workflow.answer(
        run["id"], Question(text="Find the cup that is visible in the center.", at_ms=7000)
    )
    assert answer["planner"] == "agent"
    assert [o["name"] for o in answer["states"]] == ["Blue remote"]


@pytest.mark.parametrize(
    "text,event",
    [
        ("Find the object that went out of view.", "lost"),
        ("Locate the item that moved.", "moved"),
        ("Locate the item that reappeared.", "reappeared"),
    ],
)
def test_positive_occurrence_cannot_be_compiled_as_zero(text, event):
    from agentx.services.selection import SelectionPlan, validate_selection_terms

    zero = SelectionPlan(
        intent="location", filters=[dict(kind="event", event=event, min_count=0, max_count=0)]
    )
    with pytest.raises(ValueError, match="zero count"):
        validate_selection_terms(zero, text, [], cutoff=11000)
    positive = SelectionPlan(intent="location", filters=[dict(kind="event", event=event)])
    validate_selection_terms(positive, text, [], cutoff=11000)


@pytest.mark.parametrize(
    "text",
    [
        "Find the item with no movement.",
        "Find the item that didn't move.",
        "Find the item that stayed stationary.",
        "Find the item that moved zero times.",
    ],
)
def test_explicit_negative_count_remains_supported(text):
    from agentx.services.selection import SelectionPlan, validate_selection_terms

    plan = SelectionPlan(
        intent="location", filters=[dict(kind="event", event="moved", min_count=0, max_count=0)]
    )
    validate_selection_terms(plan, text, [], cutoff=11000)


def test_beginning_constraint_is_repaired_before_any_memory_selection(app, indexed):
    from agentx.services.answering import NO_MATCH

    _, run = indexed
    s = configure(app)
    s.providers._chat = planner(
        [
            dict(
                interpretation="In the right area at time zero",
                intent="location",
                filters=[dict(kind="state", zone="right")],
            ),
            dict(intent="location", filters=[dict(kind="state", zone="right", at_ms=0)]),
        ]
    )
    answer = s.workflow.answer(
        run["id"],
        Question(
            text="Locate the item in the right area at the beginning of the recording.", at_ms=5000
        ),
    )
    assert answer["planner"] == "agent" and answer["answer"] == NO_MATCH
    assert not answer["states"] and not answer["evidence"]
    executed = [
        t
        for t in answer["tools"]
        if t["name"] == "find_objects" and isinstance(t.get("arguments"), list)
    ]
    assert len(executed) == 1 and executed[0]["arguments"][0]["at_ms"] == 0
