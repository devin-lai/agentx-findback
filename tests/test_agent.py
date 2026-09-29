import json

import pytest
from pydantic import ValidationError

from agentx.agents.providers import extract_json_object, parse_review
from agentx.domain.contracts import Question


def planner(script):
    """Replay scripted planner turns; each entry may be a dict or a callable of the messages."""
    turns = list(script)

    def respond(base, key, model, messages, **kwargs):
        step = turns.pop(0)
        return json.dumps(step(messages) if callable(step) else step)

    return respond


def object_id(messages, name):
    inventory = messages[0]["content"].split("Registered objects: ", 1)[1].split("\n")[0]
    return next(o["id"] for o in json.loads(inventory) if o["name"] == name)


def configure(app):
    s = app.state.services
    s.settings.planner_base_url = "http://planner.invalid/v1"
    s.settings.planner_model = "test-planner"
    return s


def test_planner_thinking_extension_is_opt_in(app, monkeypatch):
    import httpx

    s = configure(app)
    bodies = []

    def post(client, url, **kwargs):
        bodies.append(kwargs["json"])
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": '{"action":"final","answer":"test"}'}}]},
        )

    monkeypatch.setattr(httpx.Client, "post", post)
    s.providers.planner_chat([{"role": "user", "content": "test"}])
    assert "chat_template_kwargs" not in bodies[-1]
    s.settings.planner_thinking = False
    s.providers.planner_chat([{"role": "user", "content": "test"}])
    assert bodies[-1]["chat_template_kwargs"] == {"enable_thinking": False}
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}
    s.providers.planner_chat([{"role": "user", "content": "test"}], response_schema=schema)
    assert bodies[-1]["response_format"] == {"type": "json_object"}
    s.settings.planner_structured_outputs = True
    s.providers.planner_chat([{"role": "user", "content": "test"}], response_schema=schema)
    assert bodies[-1]["response_format"]["json_schema"]["schema"] == schema
    assert bodies[-1]["response_format"]["json_schema"]["strict"] is True
    s.settings.planner_selection_thinking_budget = 512
    s.providers.planner_chat([{"role": "user", "content": "test"}], response_schema=schema)
    assert bodies[-1]["thinking_token_budget"] == 512
    assert bodies[-1]["max_tokens"] == 1536
    assert bodies[-1]["chat_template_kwargs"] == {"enable_thinking": True}
    # Direct tool steps retain their existing low-latency request, even after opt-in.
    s.providers.planner_chat([{"role": "user", "content": "test"}])
    assert "thinking_token_budget" not in bodies[-1]
    assert bodies[-1]["chat_template_kwargs"] == {"enable_thinking": False}


def test_stepfun_effort_is_provider_specific_and_excludes_vllm_fields(app, monkeypatch):
    import httpx

    s = app.state.services
    s.settings.stepfun_base_url = "https://api.stepfun.com/v1"
    s.settings.stepfun_model = "step-3.7-flash"
    s.settings.stepfun_api_key = "test-only"
    s.settings.stepfun_reasoning_effort = "low"
    s.settings.planner_max_tokens = 4096
    s.settings.planner_thinking = False
    s.settings.planner_selection_thinking_budget = 512
    requests = []

    def post(_client, url, **kwargs):
        requests.append((url, kwargs["json"]))
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": '{"action":"final","answer":"ok"}'}}]},
        )

    monkeypatch.setattr(httpx.Client, "post", post)
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}
    s.providers.planner_chat([{"role": "user", "content": "test"}], response_schema=schema)
    url, body = requests[-1]
    assert url == "https://api.stepfun.com/v1/chat/completions"
    assert body["reasoning_effort"] == "low"
    assert body["max_tokens"] == 4096
    assert "chat_template_kwargs" not in body
    assert "thinking_token_budget" not in body

    s.settings.planner_base_url = "http://planner.invalid/v1"
    s.settings.planner_model = "nemotron"
    s.providers.planner_chat([{"role": "user", "content": "test"}], response_schema=schema)
    url, body = requests[-1]
    assert url == "http://planner.invalid/v1/chat/completions"
    assert "reasoning_effort" not in body
    assert body["thinking_token_budget"] == 512
    assert body["chat_template_kwargs"] == {"enable_thinking": True}

    with pytest.raises(ValidationError):
        type(s.settings)(stepfun_reasoning_effort="invalid", _env_file=None)


def test_extract_json_object_handles_reasoning_blocks_and_fences():
    text = '<think>\nlooking at frames\n</think>\n```json\n{"a": 1}\n```'
    assert extract_json_object(text) == {"a": 1}
    with pytest.raises(ValueError):
        extract_json_object('<think> unfinished reasoning {"a": 2}')
    with pytest.raises(ValueError):
        extract_json_object('prefix {"a": [1, 2]} trailing text')
    with pytest.raises(ValueError):
        extract_json_object('{"a": 1} {"a": 2}')
    with pytest.raises(ValueError):
        extract_json_object("no object here")
    with pytest.raises(ValueError):
        extract_json_object("[1, 2]")
    review = parse_review(
        '<think>frame 1 shows it</think>{"summary":"Right area at 5 s.","evidence_frame_ids":[1],"uncertainty":"none"}',
        2,
    )
    assert review.evidence_frame_ids == [1]


def test_agent_step_clips_long_text_and_ignores_unknown_keys():
    from agentx.agents.agent import AgentStep

    step = AgentStep.model_validate(
        {
            "thought": "x" * 5000,
            "action": "final",
            "answer": "y" * 2000,
            "reasoning": "ignored",
            "evidence_observation_ids": list(range(20)),
        }
    )
    assert len(step.thought) == 600 and len(step.answer or "") == 1500
    assert step.evidence_observation_ids == [str(i) for i in range(8)]
    with pytest.raises(ValidationError):
        AgentStep.model_validate({"action": "destroy"})


def test_agent_history_answer_cites_only_retrieved_observations(client, app, indexed):
    _, run = indexed
    s = configure(app)
    seen = {}

    def final(messages):
        history = json.loads(messages[-1]["content"].split(": ", 1)[1])
        ids = [e["observation_id"] for e in history["events"] if e["kind"] in {"appeared", "moved"}]
        seen["ids"] = ids
        return {
            "action": "final",
            "answer": "Red toolkit appeared on the left and moved to the right.",
            "intent": "history",
            "object_id": history["object_id"],
            "evidence_observation_ids": ids[:2],
        }

    s.providers._chat = planner(
        [
            lambda m: {
                "action": "call_tool",
                "tool": "get_state",
                "arguments": {"object_id": object_id(m, "Red toolkit")},
            },
            lambda m: {
                "action": "call_tool",
                "tool": "get_history",
                "arguments": {"object_id": object_id(m, "Red toolkit")},
            },
            final,
        ]
    )
    answer = s.workflow.answer(
        run["id"], Question(text="What happened to Red toolkit?", at_ms=7000)
    )
    assert answer["planner"] == "agent"
    assert [t["name"] for t in answer["tools"] if t["name"] != "load_skill"] == [
        "get_state",
        "get_history",
        "validate_answer",
    ]
    assert set(seen["ids"][:2]) <= {e["observation_id"] for e in answer["evidence"]}
    assert answer["states"][0]["evidence"]["observation_id"] in {
        e["observation_id"] for e in answer["evidence"]
    }
    assert "not confirmed" in answer["answer"]
    assert all(e["at_ms"] <= 7000 for e in answer["evidence"])
    assert answer["events"] and all(e["at_ms"] <= 7000 for e in answer["events"])
    assert not answer["warnings"]
    saved = client.get(f"/api/v1/runs/{run['id']}/questions").json()[-1]
    assert saved["planner"] == "agent" and saved["planner_model"] == "test-planner"


def test_agent_step_budget_and_duplicate_calls_are_bounded(client, app, indexed):
    _, run = indexed
    s = configure(app)
    s.settings.planner_max_steps = 3
    calls = []

    def repeat(messages):
        calls.append(messages[-1]["content"][:160])
        return {
            "action": "call_tool",
            "tool": "get_state",
            "arguments": {"object_id": object_id(messages, "Blue remote")},
        }

    s.providers._chat = planner([repeat, repeat, repeat, repeat])
    answer = s.workflow.answer(run["id"], Question(text="Where is Blue remote?", at_ms=5000))
    assert answer["planner"] == "local"
    assert any("local" in w for w in answer["warnings"])
    assert answer["evidence"]  # the deterministic fallback still answers
    assert len(calls) == 3
    assert '"repeated_call": true' in calls[2]


def test_agent_review_tool_is_single_use_and_non_authoritative(client, app, indexed, monkeypatch):
    _, run = indexed
    s = configure(app)
    s.settings.cosmos_base_url = "http://cosmos.invalid/v1"
    routed = []

    def chat(base, key, model, messages, **kwargs):
        routed.append(model)
        if model == s.settings.cosmos_model:
            return json.dumps(
                {
                    "summary": "The square is on the right.",
                    "evidence_frame_ids": [0],
                    "uncertainty": "n/a",
                }
            )
        last = messages[-1]["content"]
        if "review_frames may be used only once" in last:
            return json.dumps(
                {
                    "action": "final",
                    "answer": "Red toolkit was last seen on the right; Cosmos saw a square there.",
                    "intent": "location",
                    "object_id": object_id(messages, "Red toolkit"),
                    "evidence_observation_ids": [],
                }
            )
        if last.startswith("TOOL_RESULT review_frames"):
            return json.dumps(
                {"action": "call_tool", "tool": "review_frames", "arguments": {"question": "again"}}
            )
        if last.startswith("TOOL_RESULT get_state"):
            return json.dumps(
                {
                    "action": "call_tool",
                    "tool": "review_frames",
                    "arguments": {"question": "What is near it?"},
                }
            )
        return json.dumps(
            {
                "action": "call_tool",
                "tool": "get_state",
                "arguments": {"object_id": object_id(messages, "Red toolkit")},
            }
        )

    monkeypatch.setattr(s.providers, "_chat", chat)
    before = client.get(f"/api/v1/runs/{run['id']}/export").json()
    answer = s.workflow.answer(run["id"], Question(text="What is next to Red toolkit?", at_ms=5000))
    assert answer["planner"] == "agent"
    names = [t["name"] for t in answer["tools"]]
    assert names.count("review_frames") == 2
    assert "error" in answer["tools"][-2]["result"]
    assert routed.count(s.settings.cosmos_model) == 1
    # The state's own evidence is attached when the agent cites nothing.
    assert len(answer["evidence"]) == 1
    after = client.get(f"/api/v1/runs/{run['id']}/export").json()
    assert after["observations"] == before["observations"] and after["events"] == before["events"]
    assert len(after["reviews"]) == 1 and after["reviews"][0]["authoritative"] is False


def test_agent_refuses_unknown_tools_and_future_observations(client, app, indexed):
    _, run = indexed
    s = configure(app)
    future = [
        o
        for o in client.get(f"/api/v1/runs/{run['id']}/export").json()["observations"]
        if o["visible"] and o["at_ms"] > 3000
    ]

    s.providers._chat = planner(
        [
            {"action": "call_tool", "tool": "drop_table", "arguments": {}},
            {
                "action": "call_tool",
                "tool": "get_observation",
                "arguments": {"observation_id": future[0]["id"]},
            },
            lambda m: {
                "action": "final",
                "answer": "It is on the right.",
                "intent": "location",
                "object_id": future[0]["object_id"],
                "evidence_observation_ids": [future[0]["id"]],
            },
        ]
    )
    answer = s.workflow.answer(run["id"], Question(text="Where is Red toolkit?", at_ms=2000))
    assert answer["planner"] == "agent"
    assert "error" in answer["tools"][-2]["result"]
    assert answer["evidence"] and all(e["at_ms"] <= 2000 for e in answer["evidence"])
    assert "Left area" in answer["answer"] and "right" not in answer["answer"]
    assert any("cited evidence" in w for w in answer["warnings"])


def test_grounded_review_orders_frames_in_code(client, app, indexed, monkeypatch):
    from agentx.domain.grounding import reduce_frame_reports

    video, run = indexed
    s = app.state.services
    s.settings.cosmos_base_url = "http://cosmos.invalid/v1"
    toolkit = next(o for o in video["objects"] if o["name"] == "Red toolkit")
    calls = []

    def locate(base, key, model, messages, **kwargs):
        content = messages[-1]["content"]
        texts = [c["text"] for c in content if c["type"] == "text"]
        images = [c for c in content if c["type"] == "image_url"]
        calls.append((texts, len(images)))
        at_ms = int(next(t for t in texts if t.startswith("frame_id=")).split("video_time_ms=")[1])
        if at_ms < 3000:
            return json.dumps({"present": True, "x": 0.17, "y": 0.43, "note": "left"})
        if at_ms < 6000:
            return json.dumps({"present": True, "x": 0.82, "y": 0.43, "note": "right"})
        return json.dumps({"present": False, "x": None, "y": None, "note": "covered"})

    monkeypatch.setattr(s.providers, "_chat", locate)
    response = client.post(
        f"/api/v1/runs/{run['id']}/review",
        json={"text": "Where was it last visible?", "at_ms": 7000, "object_id": toolkit["id"]},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["mode"] == "grounded" and data["target"] == "Red toolkit"
    assert len(calls) == 8 and all(n == 2 for _, n in calls)  # reference crop + frame
    assert all(any("Target object (data): Red toolkit" in t for t in texts) for texts, _ in calls)
    assert "last visible in Right area at 00:05.0" in data["summary"]
    assert "Zones over time: 00:00.0 Left area → 00:03.0 Right area" in data["summary"]
    assert data["evidence_frame_ids"][:2] == [5, 6]
    reports = data["frame_reports"]
    assert [r["zone"] for r in reports] == ["left"] * 3 + ["right"] * 3 + [None, None]
    assert data["authoritative"] is False
    assert data["provenance"]["per_frame_calls"] == 8 and data["provenance"]["reference_crop"]
    saved = client.get(f"/api/v1/runs/{run['id']}/reviews").json()[-1]
    assert saved["frame_reports"] == reports
    assert client.get(saved["frames"][5]["frame_url"]).status_code == 200
    # Pure reduction: never visible.
    never = reduce_frame_reports(
        "Cup", [{"id": i, "at_ms": i * 1000, "present": False} for i in range(3)], run["regions"]
    )
    assert "not visible in any of the 3 supplied frames" in never["summary"]
    assert never["evidence_frame_ids"] == [2]
    with pytest.raises(LookupError):
        s.reviews.create(run["id"], Question(text="x", at_ms=7000, object_id="not-here"))


def test_json_object_with_backticks_inside_a_string_is_kept():
    text = '{"summary": "The label reads ```ABC```", "evidence_frame_ids": [1], "uncertainty": ""}'
    assert extract_json_object(text)["summary"] == "The label reads ```ABC```"
    assert extract_json_object("```json\n" + text + "\n```")["evidence_frame_ids"] == [1]


def test_agent_path_enforces_the_selected_object(client, app, indexed, monkeypatch):
    video, run = indexed
    s = configure(app)
    remote = next(o for o in video["objects"] if o["name"] == "Blue remote")
    toolkit = next(o for o in video["objects"] if o["name"] == "Red toolkit")

    def wanders(base, key, model, messages, **kwargs):
        # The model ignores the selection and answers about the toolkit instead.
        if messages[-1]["content"].startswith("TOOL_RESULT"):
            return json.dumps(
                {
                    "action": "final",
                    "answer": "Red toolkit is on the right.",
                    "intent": "location",
                    "object_id": toolkit["id"],
                    "evidence_observation_ids": [],
                }
            )
        return json.dumps(
            {"action": "call_tool", "tool": "get_state", "arguments": {"object_id": toolkit["id"]}}
        )

    monkeypatch.setattr(s.providers, "_chat", wanders)
    answer = s.workflow.answer(
        run["id"], Question(text="Where is it?", at_ms=5000, object_id=remote["id"])
    )
    assert answer["planner"] == "agent"
    assert {st["object_id"] for st in answer["states"]} == {remote["id"]}
    assert "Blue remote" in answer["answer"] and "Red toolkit" not in answer["answer"]
    assert answer["evidence"] and answer["evidence"][0]["zone"] == "center"
    assert any("different object" in w for w in answer["warnings"])
    # A foreign object id is rejected before any planner call, as on the local path.
    response = client.post(
        f"/api/v1/runs/{run['id']}/questions",
        json={"text": "Where is it?", "at_ms": 5000, "object_id": "not-in-this-run"},
    )
    assert response.status_code == 404


def test_planner_transport_failure_is_a_soft_fallback(client, app, indexed, monkeypatch):
    import httpx

    _, run = indexed
    s = configure(app)

    def down(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(s.providers, "_chat", down)
    answer = s.workflow.answer(run["id"], Question(text="Where is Red toolkit?", at_ms=5000))
    assert answer["planner"] == "local" and "unavailable" in answer["warnings"][0]
    monkeypatch.setattr(s.providers, "_chat", lambda *a, **k: (_ for _ in ()).throw(TypeError("x")))
    crashed = s.workflow.answer(run["id"], Question(text="Where is Red toolkit?", at_ms=5000))
    assert crashed["planner"] == "local" and "internal error" in crashed["warnings"][0]


@pytest.mark.parametrize(
    "at_ms,claim,expected,forbidden",
    [
        (2000, "The toolkit is on the right at 00:11.0.", "Left area", "00:11.0"),
        (7000, "It is inside Alice's drawer.", "not confirmed", "drawer"),
        (11000, "It was stolen and is outside the room.", "Center area", "stolen"),
    ],
)
def test_model_prose_cannot_override_memory(
    client, app, indexed, at_ms, claim, expected, forbidden
):
    _, run = indexed
    s = configure(app)
    # Even a plausible final with no tool calls must not publish an invented fact.
    s.providers._chat = planner(
        [
            lambda m: {
                "action": "final",
                "answer": claim,
                "uncertainty": "Definitely in the drawer.",
                "object_id": object_id(m, "Red toolkit"),
            }
        ]
    )
    answer = s.workflow.answer(run["id"], Question(text="Where is Red toolkit?", at_ms=at_ms))
    assert answer["planner"] == "agent"
    assert expected in answer["answer"] and forbidden not in answer["answer"]
    assert "drawer" not in answer["answer"]
    assert len(answer["states"]) == 1 and answer["evidence"]
    assert answer["tools"][-1]["answer_source"] == "verified_memory_v1"


def test_short_object_references_are_scoped_and_saved_as_real_ids(app, indexed):
    video, run = indexed
    s = configure(app)
    remote = next(o for o in video["objects"] if o["name"] == "Blue remote")

    def retrieve(messages):
        result = json.loads(messages[-1]["content"].split(": ", 1)[1])
        assert "not registered" in result["error"]
        reference = object_id(messages, "Blue remote")
        assert reference.startswith("object_") and len(reference) < 12
        assert remote["id"] not in messages[0]["content"]
        return dict(action="call_tool", tool="get_state", arguments={"object_id": reference})

    def finish(messages):
        result = json.loads(messages[-1]["content"].split(": ", 1)[1])
        assert result["object_id"] == object_id(messages, "Blue remote")
        return dict(
            action="final",
            answer="Supported location",
            object_id=result["object_id"],
            evidence_observation_ids=[result["evidence_observation_id"]],
        )

    s.providers._chat = planner(
        [
            dict(action="call_tool", tool="get_state", arguments={"object_id": "object_999"}),
            retrieve,
            finish,
        ]
    )
    answer = s.workflow.answer(run["id"], Question(text="Where is Blue remote?", at_ms=5000))
    assert answer["planner"] == "agent" and not answer["warnings"]
    assert answer["states"][0]["object_id"] == remote["id"]
    calls = [t for t in answer["tools"] if t["name"] == "get_state"]
    assert calls[1]["arguments"]["object_id"] == remote["id"]


def test_wrong_object_citation_and_exploratory_states_do_not_escape(client, app, indexed):
    video, run = indexed
    s = configure(app)
    toolkit = next(o for o in video["objects"] if o["name"] == "Red toolkit")
    remote = next(o for o in video["objects"] if o["name"] == "Blue remote")

    def final(messages):
        wrong = json.loads(messages[-1]["content"].split(": ", 1)[1])
        return {
            "action": "final",
            "answer": "Red toolkit is on the right.",
            "object_id": toolkit["id"],
            "evidence_observation_ids": [wrong["evidence_observation_id"]],
        }

    s.providers._chat = planner(
        [
            {"action": "call_tool", "tool": "get_state", "arguments": {"object_id": toolkit["id"]}},
            final,
        ]
    )
    answer = s.workflow.answer(
        run["id"], Question(text="Find it", at_ms=5000, object_id=remote["id"])
    )
    assert [s["object_id"] for s in answer["states"]] == [remote["id"]]
    assert "Blue remote" in answer["answer"] and "Center area" in answer["answer"]
    assert all(
        s.memory.observation(e["observation_id"])[0].object_id == remote["id"]
        for e in answer["evidence"]
    )
    assert any("cited evidence" in warning for warning in answer["warnings"])


def test_missing_source_cannot_publish_model_location(client, app, indexed):
    _, run = indexed
    s = configure(app)
    _, video, _, _ = s.memory.context(run["id"], 5000)
    s.catalog.path(video.source_path).unlink()
    s.providers._chat = planner(
        [
            lambda m: {
                "action": "final",
                "answer": "It is right here.",
                "object_id": object_id(m, "Red toolkit"),
            }
        ]
    )
    answer = s.workflow.answer(run["id"], Question(text="Where is Red toolkit?", at_ms=5000))
    assert answer["planner"] == "agent" and answer["evidence"] == []
    assert "no usable visual evidence" in answer["answer"]
    assert answer["states"][0]["status"] == "unknown"


def test_rejected_planner_request_reports_the_endpoint_message(app, monkeypatch):
    """A 400 is a configuration mismatch; the operator needs the server's own reason."""
    import httpx

    from agentx.agents.providers import PlannerUnavailable

    s = configure(app)
    s.settings.planner_selection_thinking_budget = 512

    def post(client, url, **kwargs):
        return httpx.Response(
            400,
            request=httpx.Request("POST", url),
            json={
                "error": {
                    "message": "thinking_token_budget is set but reasoning_config is not "
                    "configured. Please set --reasoning-parser.",
                    "type": "BadRequestError",
                }
            },
        )

    monkeypatch.setattr(httpx.Client, "post", post)
    with pytest.raises(PlannerUnavailable) as failure:
        s.providers.planner_chat(
            [{"role": "user", "content": "test"}], response_schema={"type": "object"}
        )
    assert "HTTP 400" in str(failure.value)
    assert "reasoning-parser" in str(failure.value)
    assert "test-planner" in str(failure.value)


def test_doctor_probe_reports_each_request_shape_separately(app, monkeypatch):
    """The tool loop and typed selection do not send the same fields, so both are probed."""
    import httpx

    from agentx.cli import probe_planner

    s = configure(app)
    s.settings.planner_selection_thinking_budget = 512

    def post(client, url, **kwargs):
        if "thinking_token_budget" in kwargs["json"]:
            return httpx.Response(
                400, request=httpx.Request("POST", url), text="no reasoning parser"
            )
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": '{"ok":true}'}}]},
        )

    monkeypatch.setattr(httpx.Client, "post", post)
    result = probe_planner(s.settings)
    assert result["tool_loop"] == "ok"
    assert result["object_selection"].startswith("failed:")
    assert "no reasoning parser" in result["object_selection"]


def test_doctor_probe_is_silent_without_a_configured_planner(app):
    from agentx.cli import probe_planner

    assert probe_planner(app.state.services.settings) == {"configured": False}


def test_doctor_cosmos_probe_checks_served_names_without_exposing_endpoint(app, monkeypatch):
    import httpx

    from agentx.cli import probe_cosmos

    settings = app.state.services.settings
    settings.cosmos_base_url = "http://cosmos.invalid/v1"
    settings.cosmos_model = "base-cosmos"
    settings.cosmos_reference_adapter_model = "missing-adapter"

    def get(client, url, **kwargs):
        assert url == "http://cosmos.invalid/v1/models"
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={"data": [{"id": "base-cosmos", "parent": None}]},
        )

    monkeypatch.setattr(httpx.Client, "get", get)
    result = probe_cosmos(settings)
    assert result == {
        "configured": True,
        "backend": "http",
        "base_model": "ok",
        "reference_adapter": "unreachable_or_name_mismatch",
    }
    assert "cosmos.invalid" not in str(result)


def test_doctor_cosmos_probe_reports_unconfigured_service(app):
    from agentx.cli import probe_cosmos

    assert probe_cosmos(app.state.services.settings) == {"configured": False}


def test_skills_endpoint_publishes_the_text_behind_each_hash(client):
    """An answer stores a Skill hash; it is only auditable if the text can be read back."""
    import hashlib

    body = client.get("/api/v1/skills")
    assert body.status_code == 200, body.text
    catalog = body.json()
    assert {s["name"] for s in catalog} == set(client.get("/api/v1/capabilities").json()["skills"])
    for entry in catalog:
        assert entry["version"], entry["name"]
        assert entry["description"]
        assert hashlib.sha256(entry["body"].encode()).hexdigest() == entry["sha256"]


def test_a_saved_answer_cites_a_skill_hash_the_endpoint_serves(client, indexed):
    _, run = indexed
    answer = client.post(
        f"/api/v1/runs/{run['id']}/questions", json={"text": "Where is Red toolkit?", "at_ms": 2000}
    )
    assert answer.status_code == 200, answer.text
    published = {s["name"]: s["sha256"] for s in client.get("/api/v1/skills").json()}
    cited = answer.json()["skills"]
    assert cited, "the local workflow still records its Skill provenance"
    for trace in cited:
        assert published[trace["name"]] == trace["sha256"]


def test_several_invented_citations_produce_one_statement_not_a_tally(client, app, indexed):
    """The warning tells a reader the citations were removed, not how many were invented."""
    _, run = indexed
    s = configure(app)
    toolkit = client.get(f"/api/v1/runs/{run['id']}/state?at_ms=5000").json()[0]

    s.providers._chat = planner(
        [
            lambda m: {
                "action": "call_tool",
                "tool": "get_state",
                "arguments": {"object_id": object_id(m, toolkit["name"])},
            },
            lambda m: {
                "action": "final",
                "answer": "It is on the right.",
                "intent": "location",
                "object_id": object_id(m, toolkit["name"]),
                "evidence_observation_ids": ["invented-a", "invented-b", "invented-c"],
            },
        ]
    )
    answer = s.workflow.answer(run["id"], Question(text=f"Where is {toolkit['name']}?", at_ms=5000))

    assert answer["planner"] == "agent"
    assert [w for w in answer["warnings"] if "cited evidence" in w] == [
        "The agent cited evidence it never retrieved or that is unsupported for the "
        "selected object; it was removed."
    ]
    # The rendered evidence still comes from memory, never from the invented identifiers.
    assert answer["evidence"] and all(
        e["observation_id"] not in {"invented-a", "invented-b", "invented-c"}
        for e in answer["evidence"]
    )


def test_a_reviewer_timeout_is_a_failed_tool_call_not_a_lost_answer(
    client, app, indexed, monkeypatch
):
    import httpx

    _, run = indexed
    s = configure(app)
    s.settings.cosmos_base_url = "http://cosmos.invalid/v1"

    def chat(base, key, model, messages, **kwargs):
        if model == s.settings.cosmos_model:
            raise httpx.ReadTimeout("reviewer too slow")
        last = messages[-1]["content"]
        if last.startswith("TOOL_RESULT review_frames"):
            return json.dumps(
                {
                    "action": "final",
                    "answer": "Red toolkit was last seen in the left area.",
                    "intent": "location",
                    "object_id": object_id(messages, "Red toolkit"),
                    "evidence_observation_ids": [],
                }
            )
        return json.dumps(
            {"action": "call_tool", "tool": "review_frames", "arguments": {"question": "Where?"}}
        )

    monkeypatch.setattr(s.providers, "_chat", chat)
    answer = s.workflow.answer(run["id"], Question(text="Where is Red toolkit?", at_ms=5000))
    assert answer["planner"] == "agent"
    review = next(t for t in answer["tools"] if t["name"] == "review_frames")
    assert "did not respond" in json.dumps(review["result"])
