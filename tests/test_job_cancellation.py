import threading

import pytest

from agentx.domain.contracts import Box, Detection
from agentx.storage.models import IndexRun


def test_queued_cancel_is_durable_idempotent_and_cannot_rewrite_completed_memory(
    client, app, indexed
):
    video, original = indexed
    queued = client.post(f"/api/v1/videos/{video['id']}/runs", json={}).json()
    for _ in range(2):
        response = client.post(f"/api/v1/runs/{queued['id']}/cancel")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"
    app.state.services.indexer.process(queued["id"])
    cancelled = client.get(f"/api/v1/runs/{queued['id']}").json()
    assert cancelled["status"] == "cancelled" and cancelled["observation_count"] == 0
    assert client.get(f"/api/v1/runs/{queued['id']}/state").status_code == 422
    assert client.post(f"/api/v1/runs/{original['id']}/cancel").status_code == 409
    assert client.post("/api/v1/runs/missing/cancel").status_code == 404
    assert client.get(f"/api/v1/runs/{original['id']}").json()["status"] == "complete"


@pytest.mark.parametrize("provider_fails_after_cancel", [False, True])
def test_running_cancel_preserves_batches_blocks_edits_and_releases_worker(
    client, app, indexed, monkeypatch, provider_fails_after_cancel
):
    video, original = indexed
    entered, release = threading.Event(), threading.Event()

    class BlockingDetector:
        provenance = {"adapter": "controlled-blocking-test"}

        def __init__(self, objects, *args):
            self.objects, self.calls = objects, 0

        def detect(self, frame, at_ms):
            self.calls += 1
            if self.calls == 12:
                entered.set()
                assert release.wait(10), "Test did not release its controlled inference call."
                if provider_fails_after_cancel:
                    raise ValueError("Provider ended after cancellation.")
            return [Detection(o["id"], Box.model_validate(o["box"]), 0.99) for o in self.objects]

    monkeypatch.setattr("agentx.services.indexing.ReferenceDetector", BlockingDetector)
    run = client.post(f"/api/v1/videos/{video['id']}/runs", json={}).json()
    indexer = app.state.services.indexer
    worker = threading.Thread(target=indexer.process, args=(run["id"],))
    worker.start()
    try:
        assert entered.wait(10)
        # A duplicate attempt to claim the running job must not remove its cancellation signal.
        indexer.process(run["id"])
        response = client.post(f"/api/v1/runs/{run['id']}/cancel")
        assert response.status_code == 200 and response.json()["status"] == "cancelling"
        assert client.post(f"/api/v1/runs/{run['id']}/cancel").json()["status"] == "cancelling"
        assert client.delete(f"/api/v1/videos/{video['id']}").status_code == 409
        assert client.post(f"/api/v1/videos/{video['id']}/runs", json={}).status_code == 422
    finally:
        release.set()
        worker.join(timeout=10)
    assert not worker.is_alive()
    stopped = client.get(f"/api/v1/runs/{run['id']}").json()
    assert stopped["status"] == "cancelled"
    assert stopped["observation_count"] == 20
    assert stopped["error"] is None
    assert run["id"] not in indexer.cancel_requests
    assert client.get(f"/api/v1/runs/{original['id']}").json()["status"] == "complete"
    assert client.post(f"/api/v1/videos/{video['id']}/runs", json={}).status_code == 202


def test_restart_acknowledges_persisted_cancel_intent(client, app, indexed):
    video, original = indexed
    run = client.post(f"/api/v1/videos/{video['id']}/runs", json={}).json()
    indexer = app.state.services.indexer
    with app.state.services.db.session.begin() as session:
        session.get(IndexRun, run["id"]).status = "cancelling"
    indexer.start()
    indexer.close()
    assert client.get(f"/api/v1/runs/{run['id']}").json()["status"] == "cancelled"
    assert client.get(f"/api/v1/runs/{original['id']}").json()["status"] == "complete"
