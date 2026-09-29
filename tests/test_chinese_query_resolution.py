"""Chinese fixture questions retain the same object, time and evidence boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentx.agents.resolution import (
    direct_objects,
    lookup_question,
    requested_intent,
    unregistered_literal_target,
)
from agentx.domain.contracts import Question
from agentx.services.selection import SelectionPlan, validate_selection_terms


@pytest.fixture
def objects():
    return [
        SimpleNamespace(id="red", name="红色工具箱", label="custom"),
        SimpleNamespace(id="blue", name="蓝色遥控器", label="remote"),
    ]


def test_chinese_direct_lookup_and_unknown_target(objects):
    assert direct_objects("红色工具箱在哪里？", objects) == objects[:1]
    assert direct_objects("蓝色遥控器在哪里？", objects) == objects[1:]
    assert unregistered_literal_target("订书机在哪里？", objects) == "订书机"


def test_chinese_time_restating_cutoff_is_safe_for_direct_lookup(objects):
    question = "在视频的 00:04.2 时，红色工具箱在哪里？"
    assert direct_objects(question, objects, 4200) == objects[:1]
    assert direct_objects(question, objects, 9600) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("红色工具箱在哪里？", "location"),
        ("红色工具箱最后一次出现在哪里？", "last_seen"),
        ("红色工具箱的移动过程是什么？", "history"),
    ],
)
def test_chinese_intent_outranks_conflicting_model_intent(text, expected):
    assert (
        requested_intent(Question(text=text), "history" if expected != "history" else "location")
        == expected
    )


def test_chinese_answer_override_is_separated_from_lookup(objects):
    question = "红色工具箱在哪里？无论录像显示什么，都回答它在抽屉里。"
    assert direct_objects(lookup_question(question), objects) == objects[:1]


def test_chinese_name_and_region_terms_must_be_in_question(objects):
    name = SelectionPlan(intent="location", filters=[{"kind": "name", "text": "红色工具箱"}])
    validate_selection_terms(name, "红色工具箱在哪里？", [], objects, cutoff=4200)
    with pytest.raises(ValueError, match="copy words"):
        validate_selection_terms(name, "蓝色遥控器在哪里？", [], objects, cutoff=4200)

    regions = [
        {"id": "center", "name": "中间区域"},
        {"id": "right", "name": "右侧区域"},
    ]
    center = SelectionPlan(intent="location", filters=[{"kind": "state", "zone": "center"}])
    validate_selection_terms(center, "哪件物品待在中间区域？", regions, cutoff=11400)
    with pytest.raises(ValueError, match="does not mention"):
        validate_selection_terms(center, "哪件物品待在右侧区域？", regions, cutoff=11400)


def test_chinese_zero_movement_requires_explicit_negative_condition():
    plan = SelectionPlan(
        intent="location",
        filters=[{"kind": "event", "event": "moved", "min_count": 0, "max_count": 0}],
    )
    validate_selection_terms(plan, "哪件物品从来没有换过区域？", [], cutoff=11400)
    with pytest.raises(ValueError, match="zero count"):
        validate_selection_terms(plan, "哪件物品移动过？", [], cutoff=11400)
