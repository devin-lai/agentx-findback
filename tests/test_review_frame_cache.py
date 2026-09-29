import hashlib

import pytest

from agentx.storage.models import ReviewRecord


def saved_review(client, app, indexed, monkeypatch):
    video, run = indexed
    services = app.state.services
    services.settings.cosmos_base_url = "http://example.invalid/v1"
    monkeypatch.setattr(
        services.providers,
        "_chat",
        lambda *a, **kw: (
            '{"summary":"Observed.","evidence_frame_ids":[0],"uncertainty":"Sampled frames only."}'
        ),
    )
    response = client.post(
        f"/api/v1/runs/{run['id']}/review", json={"text": "What is visible?", "at_ms": 7000}
    )
    assert response.status_code == 200, response.text
    review = response.json()
    with services.db.session() as session:
        from agentx.storage.models import Video

        source = services.catalog.path(session.get(Video, video["id"]).source_path)
    return review, source


def test_saved_review_replays_exact_bytes_without_reencoding(client, app, indexed, monkeypatch):
    review, source = saved_review(client, app, indexed, monkeypatch)
    assert len(list((source.parent / "review_frames").glob("*.jpg"))) == len(review["frames"])

    def different_runtime(*args, **kwargs):
        raise AssertionError("A preserved review frame should not need the video decoder.")

    monkeypatch.setattr("agentx.services.reviews.read_frame", different_runtime)
    for frame in review["frames"]:
        response = client.get(frame["frame_url"])
        assert response.status_code == 200
        assert hashlib.sha256(response.content).hexdigest() == frame["sha256"]
        assert response.headers["x-frame-time-ms"] == str(frame["at_ms"])


@pytest.mark.parametrize("change", ["missing", "changed"])
def test_cache_does_not_bypass_original_source_integrity(client, app, indexed, monkeypatch, change):
    review, source = saved_review(client, app, indexed, monkeypatch)
    if change == "missing":
        source.unlink()
    else:
        source.write_bytes(b"replaced original")
    assert client.get(review["frames"][0]["frame_url"]).status_code in (404, 422)


def test_modified_cache_is_rejected(client, app, indexed, monkeypatch):
    review, source = saved_review(client, app, indexed, monkeypatch)
    for path in (source.parent / "review_frames").glob("*.jpg"):
        path.write_bytes(b"changed cached bytes")
    response = client.get(review["frames"][0]["frame_url"])
    assert response.status_code == 422 and "Cached evidence" in response.json()["detail"]


@pytest.mark.parametrize("change", ["timestamp", "geometry"])
def test_changed_review_scope_cannot_reuse_an_old_cached_frame(
    client, app, indexed, monkeypatch, change
):
    review, _ = saved_review(client, app, indexed, monkeypatch)
    with app.state.services.db.session.begin() as session:
        record = session.get(ReviewRecord, review["id"])
        modified = {**record.response}
        if change == "timestamp":
            modified["frames"] = [dict(f) for f in modified["frames"]]
            modified["frames"][0]["at_ms"] = 9000
        else:
            modified["provenance"] = {**modified["provenance"], "max_edge": 768}
        record.response = modified
    assert client.get(review["frames"][0]["frame_url"]).status_code == 422


def test_legacy_review_backfills_verified_cache_and_video_deletion_removes_it(
    client, app, indexed, monkeypatch
):
    review, source = saved_review(client, app, indexed, monkeypatch)
    for path in (source.parent / "review_frames").glob("*.jpg"):
        path.unlink()
    response = client.get(review["frames"][0]["frame_url"])
    assert response.status_code == 200
    assert len(list((source.parent / "review_frames").glob("*.jpg"))) == 1
    assert client.delete(f"/api/v1/videos/{indexed[0]['id']}").status_code == 204
    assert not source.parent.exists()
