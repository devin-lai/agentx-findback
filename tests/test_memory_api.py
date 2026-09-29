import pytest
from sqlalchemy import func, select

from agentx.storage.models import Event, IndexRun, Observation, QueryRecord, RegisteredObject, Video


def ask(client, run, time, text="Where is the Red toolkit?", **kwargs):
    result = client.post(
        f"/api/v1/runs/{run['id']}/questions", json={"text": text, "at_ms": time, **kwargs}
    )
    assert result.status_code == 200, result.text
    return result.json()


def test_real_video_to_causal_answers_and_original_evidence(client, indexed):
    video, run = indexed
    assert run["provenance"]["learned_model"] is False
    assert run["provenance"]["input_sha256"] == video["sha256"]
    assert run["observation_count"] == 120
    for cutoff, status, zone in [
        (2000, "visible", "left"),
        (5000, "visible", "right"),
        (7000, "last_seen", "right"),
        (11000, "visible", "center"),
    ]:
        answer = ask(client, run, cutoff)
        state = answer["states"][0]
        assert (state["status"], state["zone"]) == (status, zone)
        assert all(e["at_ms"] <= cutoff for e in answer["evidence"])
        assert all(e["video_id"] == video["id"] for e in answer["evidence"])
        assert {t["name"] for t in answer["tools"]} >= {
            "load_skill",
            "get_state",
            "validate_answer",
        }
        assert all(len(s["sha256"]) == 64 for s in answer["skills"])
        frame = client.get(answer["evidence"][0]["frame_url"])
        assert frame.status_code == 200
        assert frame.headers["content-type"] == "image/jpeg"
        assert int(frame.headers["x-frame-time-ms"]) <= cutoff
        if status == "last_seen":
            assert state["current_zone"] is None
            assert state["last_observed_ms"] == 5800
            assert "not confirmed" in answer["answer"]
            assert "inside" not in answer["answer"]
    answer = ask(client, run, 2000, "Show the history of Red toolkit")
    assert "Right area" not in answer["answer"]
    assert all(e["at_ms"] <= 2000 for e in answer["events"])
    future = client.post(
        f"/api/v1/runs/{run['id']}/questions",
        json={"text": "Where is Red toolkit?", "at_ms": 99999},
    )
    assert future.status_code == 422
    assert client.get(f"/api/v1/runs/{run['id']}/state?at_ms=-1").status_code == 422
    playback = client.get(video["media_url"], headers={"Range": "bytes=0-99"})
    assert playback.status_code == 206
    assert len(playback.content) == 100
    assert playback.headers["content-range"].startswith("bytes 0-99/")
    exported = client.get(f"/api/v1/runs/{run['id']}/export").json()
    assert len(exported["observations"]) == 120
    assert exported["queries"]
    assert len(client.get(f"/api/v1/runs/{run['id']}/questions").json()) == 5


def test_unknown_object_is_not_invented_and_cross_run_object_is_rejected(client, indexed):
    _, run = indexed
    answer = ask(client, run, 5000, "Where is my passport?")
    assert not answer["evidence"]
    assert not answer["states"]
    assert "could not match" in answer["answer"]
    response = client.post(
        f"/api/v1/runs/{run['id']}/questions", json={"text": "Where?", "object_id": "foreign"}
    )
    assert response.status_code == 404
    assert client.get(f"/api/v1/runs/{run['id']}/events?object_id=foreign").status_code == 404


def test_unconfirmed_identity_returns_unknown_with_only_historical_evidence(client, app, indexed):
    _, run = indexed
    s = app.state.services
    _, _, objects, _ = s.memory.context(run["id"], 5000)
    target = next(obj for obj in objects if obj.name == "Red toolkit")
    with s.db.session.begin() as session:
        latest = session.scalar(
            select(Observation)
            .where(
                Observation.run_id == run["id"],
                Observation.object_id == target.id,
                Observation.at_ms <= 5000,
            )
            .order_by(Observation.at_ms.desc())
            .limit(1)
        )
        assert latest is not None and latest.visible
        uncertain_at = latest.at_ms
        latest.visible = False
        latest.box = None
        latest.zone = None
        latest.reason = "identity_unconfirmed"
    answer = ask(client, run, uncertain_at, object_id=target.id, use_provider=False)
    state = answer["states"][0]
    assert state["status"] == "unknown" and state["reason"] == "identity_unconfirmed"
    assert state["current_zone"] is None
    assert state["evidence"] and state["last_observed_ms"] < uncertain_at
    assert "identity" in answer["answer"] or "registered appearance" in answer["answer"]


def test_no_evidence_means_no_supported_location_including_history(client, app, indexed):
    video, run = indexed
    s = app.state.services
    with s.db.session() as session:
        source = s.catalog.path(session.get(Video, video["id"]).source_path)
    source.unlink()
    answer = ask(client, run, 11000, "Show the history of Red toolkit")
    assert not answer["evidence"]
    assert answer["states"][0]["status"] == "unknown"
    assert answer["states"][0]["reason"] == "evidence_missing"
    assert "no usable visual evidence" in answer["answer"]
    assert "Recorded changes" not in answer["answer"]


def test_rebuild_keeps_history_and_region_snapshots(client, indexed):
    video, original = indexed
    renamed = video["regions"]
    renamed[0]["name"] = "New name"
    assert client.put(f"/api/v1/videos/{video['id']}/regions", json=renamed).status_code == 200
    answer = ask(client, original, 2000)
    assert "Left area" in answer["answer"]
    result = client.post(f"/api/v1/videos/{video['id']}/runs", json={})
    assert result.status_code == 202
    assert result.json()["id"] != original["id"]
    assert client.post(f"/api/v1/videos/{video['id']}/runs", json={}).status_code == 422
    assert client.delete(f"/api/v1/videos/{video['id']}").status_code == 409
    assert client.put(f"/api/v1/videos/{video['id']}/regions", json=renamed).status_code == 409
    assert client.get(f"/api/v1/runs/{original['id']}").json()["status"] == "complete"


def test_deletion_cascades_memory_and_media(client, app, indexed):
    video, run = indexed
    ask(client, run, 2000)
    assert client.delete(f"/api/v1/videos/{video['id']}").status_code == 204
    assert client.get(video["media_url"]).status_code == 404
    with app.state.services.db.session() as session:
        for model in [Video, RegisteredObject, IndexRun, Observation, Event, QueryRecord]:
            assert session.scalar(select(func.count()).select_from(model)) == 0
    assert not (app.state.services.settings.data_dir / "videos" / video["id"]).exists()


def test_invalid_uploads_and_missing_records(client, app):
    assert client.post("/api/v1/videos", files={"file": ("readme.txt", b"test")}).status_code == 415
    assert client.post("/api/v1/videos", files={"file": ("empty.mp4", b"")}).status_code == 422
    assert (
        client.post("/api/v1/videos", files={"file": ("broken.mp4", b"not video")}).status_code
        == 422
    )
    app.state.services.settings.max_upload_mb = 1
    assert (
        client.post(
            "/api/v1/videos", files={"file": ("large.mp4", b"x" * (1024 * 1024 + 1))}
        ).status_code
        == 413
    )
    assert client.get("/api/v1/videos/missing").status_code == 404
    assert client.get("/api/v1/observations/missing/frame").status_code == 404
    with pytest.raises(ValueError, match="Invalid stored media path"):
        app.state.services.catalog.path("../../escape")


def test_actual_upload_and_late_registration_do_not_retroactively_create_memory(
    client, app, indexed
):
    video, _ = indexed
    s = app.state.services
    with s.db.session() as session:
        source = s.catalog.path(session.get(Video, video["id"]).source_path)
    with source.open("rb") as stream:
        uploaded = client.post(
            "/api/v1/videos", files={"file": ("../../desk.mp4", stream, "video/mp4")}
        )
    assert uploaded.status_code == 201, uploaded.text
    new_video = uploaded.json()
    assert new_video["title"] == "desk"
    assert new_video["sha256"] == video["sha256"]
    assert client.post(f"/api/v1/videos/{new_video['id']}/runs", json={}).status_code == 422
    object_body = {
        "name": "Late toolkit",
        "at_ms": 9000,
        "box": {"x1": 280 / 640, "y1": 125 / 360, "x2": 370 / 640, "y2": 189 / 360},
    }
    assert (
        client.post(f"/api/v1/videos/{new_video['id']}/objects", json=object_body).status_code
        == 201
    )
    assert (
        client.post(f"/api/v1/videos/{new_video['id']}/objects", json=object_body).status_code
        == 422
    )
    run = client.post(f"/api/v1/videos/{new_video['id']}/runs", json={}).json()
    s.indexer.process(run["id"])
    result = ask(client, run, 2000, "Where is Late toolkit?")
    assert result["states"][0]["status"] == "not_observed"
    assert not result["evidence"]
    assert ask(client, run, 11000, "Where is Late toolkit?")["states"][0]["status"] == "visible"
