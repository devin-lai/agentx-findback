import json
import time

import pytest
from fastapi.testclient import TestClient

from agentx.api.app import create_app
from agentx.config import Settings
from agentx.domain.contracts import Question
from agentx.storage.models import IndexRun


def test_authentication_protects_api_media_and_docs(tmp_path):
    app = create_app(
        Settings(
            data_dir=tmp_path,
            database_url="",
            api_token="test-secret",
            worker_enabled=False,
            _env_file=None,
        )
    )
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/v1/videos").status_code == 401
        assert client.get("/api/v1/videos/fake/media").status_code == 401
        assert client.get("/docs").status_code == 401
        assert client.post("/api/auth/login", json={"token": "wrong"}).status_code == 401
        login = client.post("/api/auth/login", json={"token": "test-secret"})
        assert login.status_code == 200
        assert "HttpOnly" in login.headers["set-cookie"]
        assert "SameSite=strict" in login.headers["set-cookie"]
        assert client.get("/api/v1/videos").status_code == 200
        assert (
            client.post("/api/v1/demo", headers={"Origin": "https://elsewhere.example"}).status_code
            == 403
        )
        client.post("/api/auth/logout")
        assert client.get("/api/v1/videos").status_code == 401
        assert (
            client.get(
                "/api/v1/videos", headers={"Authorization": "Bearer test-secret"}
            ).status_code
            == 200
        )


def test_agent_token_reads_and_asks_but_cannot_change_memory(tmp_path):
    app = create_app(
        Settings(
            data_dir=tmp_path,
            database_url="",
            api_token="operator-secret",
            agent_token="agent-secret",
            worker_enabled=False,
            _env_file=None,
        )
    )
    operator = {"Authorization": "Bearer operator-secret"}
    agent = {"Authorization": "Bearer agent-secret"}
    with TestClient(app) as client:
        video = client.post("/api/v1/demo", headers=operator).json()
        run = client.post(f"/api/v1/videos/{video['id']}/runs", headers=operator, json={}).json()
        app.state.services.indexer.process(run["id"])
        assert client.get("/api/v1/videos", headers=agent).status_code == 200
        assert client.get(f"/api/v1/runs/{run['id']}", headers=agent).status_code == 200
        answer = client.post(
            f"/api/v1/runs/{run['id']}/questions",
            headers=agent,
            json={"text": "Where was the Red toolkit last seen?", "at_ms": 7000},
        )
        assert answer.status_code == 200 and answer.json()["states"]
        box = {"x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3}
        writes = [
            client.post("/api/v1/demo", headers=agent),
            client.post(f"/api/v1/videos/{video['id']}/runs", headers=agent, json={}),
            client.post(
                f"/api/v1/videos/{video['id']}/objects",
                headers=agent,
                json={"name": "Mug", "label": "custom", "at_ms": 0, "box": box},
            ),
            client.post(
                f"/api/v1/videos/{video['id']}/suggestions", headers=agent, json={"at_ms": 0}
            ),
            client.post(f"/api/v1/runs/{run['id']}/review", headers=agent, json={}),
            client.put(f"/api/v1/videos/{video['id']}/regions", headers=agent, json=[]),
            client.delete(f"/api/v1/videos/{video['id']}", headers=agent),
        ]
        assert [w.status_code for w in writes] == [403] * len(writes)
        assert "operator" in writes[0].json()["detail"]
        assert len(client.get("/api/v1/videos", headers=operator).json()) == 1
        # Not a sign-in: no UI session can carry the agent token.
        assert client.post("/api/auth/login", json={"token": "agent-secret"}).status_code == 401
        client.cookies.set("agentx_session", "agent-secret")
        assert client.get("/api/v1/videos").status_code == 401
        basic = {"Authorization": "Basic YWdlbnQ6c2VjcmV0"}
        assert client.get("/api/v1/videos", headers=basic).status_code == 401
        client.cookies.clear()
        # A non-ASCII token is refused instead of crashing the comparison.
        latin = {"Authorization": b"Bearer ag\xe9"}
        assert client.get("/api/v1/videos", headers=latin).status_code == 401


def test_network_access_without_token_is_rejected(tmp_path):
    app = create_app(
        Settings(data_dir=tmp_path, database_url="", worker_enabled=False, _env_file=None)
    )
    with TestClient(app, client=("192.0.2.10", 50000)) as client:
        assert client.get("/api/v1/videos").status_code == 503


@pytest.mark.parametrize("failure", [RuntimeError, ValueError])
def test_job_failure_is_durable_and_retry_is_separate(client, app, indexed, monkeypatch, failure):
    video, original = indexed

    def fail(*args, **kwargs):
        raise failure("Internal endpoint credential: synthetic-private-canary")

    monkeypatch.setattr("agentx.services.indexing.frames", fail)
    run = client.post(f"/api/v1/videos/{video['id']}/runs", json={}).json()
    app.state.services.indexer.process(run["id"])
    failed = client.get(f"/api/v1/runs/{run['id']}").json()
    assert failed["status"] == "failed"
    assert failed["error"] == "The perception worker failed. Check server logs and retry."
    assert "synthetic-private-canary" not in json.dumps(failed)
    assert client.get(f"/api/v1/runs/{run['id']}/state").status_code == 422
    assert client.get(f"/api/v1/runs/{original['id']}").json()["status"] == "complete"
    assert client.post(f"/api/v1/videos/{video['id']}/runs", json={}).status_code == 202


def test_restart_recovers_abandoned_jobs_without_rewriting_completed_run(client, app, indexed):
    video, original = indexed
    queued = client.post(f"/api/v1/videos/{video['id']}/runs", json={}).json()
    with app.state.services.db.session.begin() as session:
        session.get(IndexRun, queued["id"]).status = "running"
    indexer = app.state.services.indexer
    indexer.start()
    indexer.close()
    assert client.get(f"/api/v1/runs/{queued['id']}").json()["status"] == "failed"
    assert client.get(f"/api/v1/runs/{original['id']}").json()["status"] == "complete"
    # Re-running the schema migration must preserve existing data.
    app.state.services.db.migrate()
    assert len(client.get("/api/v1/videos").json()) == 1


def test_worker_consumes_persisted_queue(client, app):
    video = client.post("/api/v1/demo").json()
    run = client.post(f"/api/v1/videos/{video['id']}/runs", json={}).json()
    app.state.services.indexer.start()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        result = client.get(f"/api/v1/runs/{run['id']}").json()
        if result["status"] in {"complete", "failed"}:
            break
        time.sleep(0.1)
    assert result["status"] == "complete", result


def test_planner_agent_is_bounded_and_failure_falls_back(client, app, indexed, monkeypatch):
    _, run = indexed
    s = app.state.services
    s.settings.planner_base_url = "http://planner.invalid/v1"
    s.settings.planner_model = "test-planner"
    captured = []

    def respond(base, key, model, messages, **kwargs):
        captured.append(messages)
        last = messages[-1]["content"]
        if last.startswith("TOOL_RESULT get_state"):
            state = json.loads(last.split(": ", 1)[1])
            return json.dumps(
                {
                    "thought": "Cite the retrieved sighting.",
                    "action": "final",
                    "answer": "Red toolkit was last supported in the right area.",
                    "intent": "history",
                    "object_id": state["object_id"],
                    "evidence_observation_ids": [state["evidence_observation_id"], "forged-id"],
                    "uncertainty": "Its position at the cutoff is unconfirmed.",
                }
            )
        objects = json.loads(
            messages[0]["content"].split("Registered objects: ", 1)[1].split("\n")[0]
        )
        target = next(o for o in objects if o["name"] == "Red toolkit")
        return json.dumps(
            {
                "thought": "Read memory.",
                "action": "call_tool",
                "tool": "get_state",
                "arguments": {"object_id": target["id"]},
            }
        )

    monkeypatch.setattr(s.providers, "_chat", respond)
    answer = s.workflow.answer(run["id"], Question(text="Trace Red toolkit", at_ms=7000))
    assert answer["planner"] == "agent"
    assert answer["planner_model"] == "test-planner"
    assert answer["intent"] == "history"
    system = captured[0][0]["content"]
    for name in ("retrieve-object-history", "verify-location-answer", "answer-with-memory-tools"):
        assert name in system
    assert all(e["at_ms"] <= 7000 for e in answer["evidence"])
    assert answer["evidence"] and all(
        e["observation_id"] != "forged-id" for e in answer["evidence"]
    )
    assert {e["zone"] for e in answer["evidence"]} == {"left", "right"}
    assert any("cited evidence" in w for w in answer["warnings"])
    assert [t["name"] for t in answer["tools"]][-3:] == [
        "load_skill",
        "get_state",
        "validate_answer",
    ]
    assert "not confirmed" in answer["answer"]
    monkeypatch.setattr(
        s.providers,
        "_chat",
        lambda *args, **kwargs: '{"action":"delete_everything","tool":"get_state"}',
    )
    result = s.workflow.answer(run["id"], Question(text="Where is Red toolkit?", at_ms=4000))
    assert result["planner"] == "local"
    assert result["warnings"]
    assert result["evidence"]
    monkeypatch.setattr(s.providers, "_chat", lambda *args, **kwargs: "not json at all")
    result = s.workflow.answer(run["id"], Question(text="Where is anything?", at_ms=4000))
    assert result["planner"] == "local"
    assert not result["evidence"]


def test_cosmos_receives_only_past_frames_and_cannot_update_memory(
    client, app, indexed, monkeypatch
):
    video, run = indexed
    s = app.state.services
    s.settings.cosmos_base_url = "http://localhost:9999/v1"
    calls = []

    def review(*args, **kwargs):
        calls.append(args)
        return json.dumps(
            {
                "summary": "An object is visible.",
                "evidence_frame_ids": [0],
                "uncertainty": "Cannot establish its present real-world location.",
            }
        )

    monkeypatch.setattr(s.providers, "_chat", review)
    before = client.get(f"/api/v1/runs/{run['id']}/export").json()
    response = client.post(
        f"/api/v1/runs/{run['id']}/review", json={"text": "What changed?", "at_ms": 4000}
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["authoritative"] is False
    assert all(f["at_ms"] <= 4000 for f in data["frames"])
    content = calls[0][-1][1]["content"]
    assert any(item["type"] == "image_url" for item in content)
    after = client.get(f"/api/v1/runs/{run['id']}/export").json()
    for key in ("run", "observations", "events", "queries"):
        assert after[key] == before[key]
    assert len(after["reviews"]) == 1
    monkeypatch.setattr(
        s.providers,
        "_chat",
        lambda *args, **kwargs: json.dumps(
            {"summary": "Bad reference", "evidence_frame_ids": [999], "uncertainty": ""}
        ),
    )
    assert (
        client.post(
            f"/api/v1/runs/{run['id']}/review", json={"text": "What?", "at_ms": 4000}
        ).status_code
        == 502
    )


def test_skill_registry_rejects_arbitrary_file_loading(app):
    with pytest.raises(ValueError):
        app.state.services.skills.load("../../.env")


def test_query_skill_ablation_preserves_evidence_invariants(client, app, indexed):
    _, run = indexed
    s = app.state.services
    with_skills = s.workflow.answer(run["id"], Question(text="Where is Red toolkit?", at_ms=7000))
    without_skills = s.workflow.answer(
        run["id"], Question(text="Where is Red toolkit?", at_ms=7000, use_skills=False)
    )
    assert not without_skills["skills"]
    assert without_skills["answer"] == with_skills["answer"]
    assert without_skills["evidence"] == with_skills["evidence"]
    assert without_skills["tools"][-1]["name"] == "validate_answer"


def test_provider_http_contract_and_invalid_output(monkeypatch):
    import httpx

    from agentx.agents.providers import Providers

    captured = []
    real_client = httpx.Client

    def handler(request):
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning_content": '{"action":"final","answer":"ok"}',
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(
        "agentx.agents.providers.httpx.Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    providers = Providers(
        Settings(
            stepfun_base_url="https://example.invalid/v1",
            stepfun_api_key="test",
            stepfun_model="test",
            _env_file=None,
        )
    )
    payload = providers.planner_chat([{"role": "user", "content": "hi"}])
    assert json.loads(payload)["action"] == "final"
    assert str(captured[0].url) == "https://example.invalid/v1/chat/completions"
    assert captured[0].headers["authorization"] == "Bearer test"
    assert json.loads(captured[0].content)["response_format"] == {"type": "json_object"}
    assert json.loads(captured[0].content)["model"] == "test"


def test_planner_rejects_token_cutoff_before_reusing_reasoning(monkeypatch):
    import httpx

    from agentx.agents.providers import PlannerUnavailable, Providers

    requests = []

    def post(client, url, **kwargs):
        requests.append(kwargs["json"])
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": "",
                            "reasoning_content": '{"filters":[]}',
                        },
                    }
                ],
                "usage": {"completion_tokens": 1800},
            },
        )

    monkeypatch.setattr(httpx.Client, "post", post)
    providers = Providers(
        Settings(
            stepfun_base_url="https://example.invalid/v1",
            stepfun_api_key="test",
            stepfun_model="test",
            _env_file=None,
        )
    )
    usage = []
    providers.usage_sink = lambda model, payload: usage.append(payload)
    with pytest.raises(PlannerUnavailable, match="exhausted max_tokens"):
        providers.planner_chat([{"role": "user", "content": "Return JSON"}])
    assert len(requests) == 1
    assert usage == [{"completion_tokens": 1800}]


@pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": [{"message": None}]}])
def test_unexpected_provider_bodies_are_provider_failures_not_missing_resources(monkeypatch, body):
    import httpx

    from agentx.agents.providers import PlannerUnavailable, Providers

    real_client = httpx.Client
    monkeypatch.setattr(
        "agentx.agents.providers.httpx.Client",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)),
            **kwargs,
        ),
    )
    providers = Providers(
        Settings(
            stepfun_base_url="https://example.invalid/v1",
            stepfun_api_key="test",
            stepfun_model="test",
            cosmos_base_url="https://example.invalid/v1",
            _env_file=None,
        )
    )
    with pytest.raises(PlannerUnavailable):
        providers.planner_chat([{"role": "user", "content": "hi"}])
    with pytest.raises(ValueError, match="unexpected response body"):
        providers._generate([{"role": "user", "content": "hi"}], max_tokens=50, json_mode=True)
