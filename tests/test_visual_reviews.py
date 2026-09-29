import hashlib
import json

import pytest
from sqlalchemy import select

from agentx.agents.providers import parse_review
from agentx.storage.models import RegisteredObject, ReviewRecord

VALID = '{"summary":"An object is visible at 0 ms.","evidence_frame_ids":[0],"uncertainty":"Only sampled frames are known."}'


@pytest.mark.parametrize("coordinate_contract", ["normalized", "integer_1000"])
def test_grounded_review_never_uses_registration_from_the_future(
    client, app, indexed, monkeypatch, coordinate_contract
):
    from agentx.domain.contracts import Question

    video, run = indexed
    s = app.state.services
    target = video["objects"][0]["id"]
    with s.db.session.begin() as session:
        obj = session.get(RegisteredObject, target)
        obj.registered_at_ms = 4000
    s.settings.cosmos_base_url = "http://example.invalid/v1"
    s.settings.cosmos_reference_grounding = coordinate_contract
    s.settings.cosmos_reference_max_edge = 768
    sampled = []

    def locate(base, key, model, messages, **kwargs):
        content = messages[-1]["content"]
        label = next(c["text"] for c in content if c.get("text", "").startswith("frame_id="))
        sampled.append(int(label.split("video_time_ms=")[1]))
        return '{"present":false,"x":null,"y":null,"note":"not visible"}'

    monkeypatch.setattr(s.providers, "_chat", locate)
    with pytest.raises(ValueError, match="not registered"):
        s.reviews.create(run["id"], Question(text="Where?", at_ms=3000, object_id=target))
    rejected = client.post(
        f"/api/v1/runs/{run['id']}/review",
        json={"text": "Where?", "at_ms": 3000, "object_id": target},
    )
    assert rejected.status_code == 422 and "not registered" in rejected.json()["detail"]
    assert sampled == [] and s.reviews.list(run["id"]) == []
    review = s.reviews.create(run["id"], Question(text="Where?", at_ms=6000, object_id=target))
    assert sorted(sampled) == [4000, 5000, 6000]
    assert review["provenance"]["registered_at_ms"] == 4000
    assert all(4000 <= f["at_ms"] <= 6000 for f in review["frames"])


@pytest.mark.parametrize("payload", [VALID, f"```json\n{VALID}\n```", f"```\n{VALID}\n```"])
def test_review_accepts_only_whole_json_or_fence(payload):
    assert parse_review(payload, 2).evidence_frame_ids == [0]


@pytest.mark.parametrize(
    "payload",
    [
        "Here is the answer: " + VALID,
        VALID + "\n" + VALID,
        VALID.replace("[0]", "[true]"),
        VALID.replace("[0]", '["0"]'),
        VALID.replace("[0]", "[-1]"),
        VALID.replace("[0]", "[2]"),
        VALID.replace("[0]", "[0,0]"),
        VALID.replace("[0]", "[]"),
        VALID.replace('"summary"', '"description"'),
        VALID[:-1] + ',"extra":"command"}',
    ],
)
def test_review_rejects_invalid_schema_or_evidence(payload):
    with pytest.raises(ValueError):
        parse_review(payload, 2)


def test_review_retry_persistence_frame_integrity_and_skill_ablation(
    client, app, indexed, monkeypatch
):
    video, run = indexed
    s = app.state.services
    s.settings.cosmos_base_url = "http://example.invalid/v1"
    calls = []

    def generate(*args, **kwargs):
        calls.append(args[-1])
        return '{"unvalidated":"output"}' if len(calls) == 1 else VALID

    monkeypatch.setattr(s.providers, "_chat", generate)
    response = client.post(
        f"/api/v1/runs/{run['id']}/review", json={"text": "What changed?", "at_ms": 3500}
    )
    assert response.status_code == 200, response.text
    review = response.json()
    assert review["provenance"]["attempts"] == 2
    assert review["source_sha256"] == video["sha256"]
    assert review["skills"][0]["name"] == "review-visual-evidence"
    assert "Sampled frames omit intermediate actions" in calls[0][0]["content"]
    assert all(f["at_ms"] <= 3500 for f in review["frames"])
    # The original request, including inline evidence, is unchanged during repair.
    assert calls[1][:2] == calls[0]
    assert client.get(f"/api/v1/runs/{run['id']}/reviews").json() == [review]
    frame = review["frames"][0]
    image = client.get(frame["frame_url"])
    assert image.status_code == 200
    assert hashlib.sha256(image.content).hexdigest() == frame["sha256"]
    assert int(image.headers["X-Frame-Time-Ms"]) == frame["at_ms"]
    assert client.get(f"/api/v1/reviews/{review['id']}/frames/999").status_code == 404
    response = client.post(
        f"/api/v1/runs/{run['id']}/review",
        json={"text": "What changed?", "at_ms": 3500, "use_skills": False},
    )
    assert response.status_code == 200
    assert response.json()["skills"] == []
    assert "Sampled frames omit intermediate actions" not in calls[-1][0]["content"]
    assert "An object disappearing does not prove" in calls[-1][0]["content"]
    assert response.json()["provenance"]["prompt_sha256"] != review["provenance"]["prompt_sha256"]
    with s.db.session() as session:
        record = session.get(ReviewRecord, review["id"])
        broken = json.loads(json.dumps(record.response))
        broken["frames"][0]["sha256"] = "0" * 64
        record.response = broken
        session.commit()
    assert client.get(frame["frame_url"]).status_code == 422
    assert client.delete(f"/api/v1/videos/{video['id']}").status_code == 204
    with s.db.session() as session:
        assert session.scalar(select(ReviewRecord)) is None


def test_review_busy_invalid_output_and_unconfigured_do_not_persist(
    client, app, indexed, monkeypatch
):
    _, run = indexed
    s = app.state.services
    url = f"/api/v1/runs/{run['id']}/review"
    body = {"text": "Where is the object?", "at_ms": 0}
    assert client.post(url, json=body).status_code == 409
    s.settings.cosmos_base_url = "http://example.invalid/v1"
    with s.providers.review_lock:
        busy = client.post(url, json=body)
        assert busy.status_code == 429
        assert busy.headers["Retry-After"] == "10"
    calls = []

    def invalid(*args, **kwargs):
        calls.append(args)
        return "invalid"

    monkeypatch.setattr(s.providers, "_chat", invalid)
    assert client.post(url, json=body).status_code == 502
    assert len(calls) == 2
    assert not s.providers.review_lock.locked()
    assert client.get(f"/api/v1/runs/{run['id']}/reviews").json() == []
    assert client.post(url, json={**body, "at_ms": 999999}).status_code == 422


def test_native_adapter_is_selected_and_reports_actual_provenance(
    client, app, indexed, monkeypatch, tmp_path
):
    _, run = indexed
    (tmp_path / "config.json").write_text("{}")
    s = app.state.services
    s.settings.cosmos_backend = "transformers"
    s.settings.cosmos_model_path = str(tmp_path)
    calls = []

    def generate(messages, **kwargs):
        calls.append(messages)
        s.providers.local_cosmos.provenance = {
            "backend": "transformers",
            "device": "unit-test-stub",
        }
        return VALID

    monkeypatch.setattr(s.providers.local_cosmos, "generate", generate)
    assert client.get("/api/v1/capabilities").json()["cosmos"] is True
    response = client.post(
        f"/api/v1/runs/{run['id']}/review", json={"text": "What is visible?", "at_ms": 0}
    )
    assert response.status_code == 200
    assert response.json()["provenance"]["device"] == "unit-test-stub"
    assert len(calls) == 1
