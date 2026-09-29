import hashlib
import os

import pytest

from agentx.domain.contracts import IndexRequest, Question
from agentx.services.selection import find_objects
from agentx.services.source_integrity import SourceIntegrity


def test_integrity_cache_detects_same_size_replacement_and_restored_mtime(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    original = b"a recording with a fixed size"
    source.write_bytes(original)
    expected = hashlib.sha256(original).hexdigest()
    verifier = SourceIntegrity(capacity=2)
    calls = []
    digest = hashlib.file_digest

    def count(stream, algorithm):
        calls.append(stream.name)
        return digest(stream, algorithm)

    monkeypatch.setattr(hashlib, "file_digest", count)
    assert verifier.problem(source, expected) is None
    assert verifier.problem(source, expected) is None
    assert len(calls) == 1
    before = source.stat()
    source.write_bytes(b"X" + original[1:])
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert verifier.problem(source, expected) == "evidence_changed"
    assert verifier.problem(source, expected) == "evidence_changed"
    assert len(calls) == 2
    source.write_bytes(original)
    assert verifier.problem(source, expected) is None
    assert len(calls) == 3
    assert len(verifier._cache) == 2
    source.unlink()
    assert verifier.problem(source, expected) == "evidence_missing"


def test_mutated_source_invalidates_live_claims_frames_and_new_work(client, app, indexed):
    video, run = indexed
    s = app.state.services
    context = s.memory.context(run["id"], 11000)
    _, recording, objects, _ = context
    answer = s.workflow.answer(
        run["id"], Question(text="Where is Red toolkit?", at_ms=11000, use_provider=False)
    )
    evidence = answer["evidence"][0]
    source = s.catalog.require_source(recording)
    original = source.read_bytes()
    # Still decodable footage, but no longer the immutable import identified by its hash.
    source.write_bytes(original + b"changed")
    for intent in ("location", "history", "last_seen"):
        result = s.workflow.answer(
            run["id"],
            Question(text="Where is Red toolkit?", at_ms=11000, intent=intent, use_provider=False),
        )
        assert result["states"][0]["status"] == "unknown"
        assert result["states"][0]["reason"] == "evidence_changed"
        assert result["states"][0]["current_zone"] is None
        assert not result["evidence"]
        assert "no usable visual evidence" in result["answer"]
        assert "Recorded changes" not in result["answer"]
    with pytest.raises(ValueError, match="no longer matches"):
        s.memory.evidence_for(evidence["observation_id"], 11000)
    with pytest.raises(ValueError, match="no longer matches"):
        find_objects(s.memory, *context, {"filters": [{"kind": "state", "status": "visible"}]})
    with pytest.raises(ValueError, match="no longer matches"):
        s.indexer.enqueue(video["id"], IndexRequest())
    with pytest.raises(ValueError, match="no longer matches"):
        s.reviews.create(run["id"], Question(text="Review", at_ms=11000))
    for url in (evidence["frame_url"], f"/api/v1/videos/{video['id']}/frame?at_ms=1000"):
        response = client.get(url)
        assert response.status_code == 422
        assert "no longer matches" in response.json()["detail"]
    registered = client.post(
        f"/api/v1/videos/{video['id']}/objects",
        json={"name": "New object", "at_ms": 0, "box": objects[0].box},
    )
    assert registered.status_code == 422
    source.write_bytes(original)
    assert s.memory.object_state(*context[:2], objects[0], 11000).evidence
    restored = client.get(evidence["frame_url"])
    assert restored.status_code == 200
    assert restored.headers["cache-control"] == "no-store"


def test_queued_index_rechecks_source_before_loading_models(app, indexed):
    video, _ = indexed
    s = app.state.services
    run = s.indexer.enqueue(video["id"], IndexRequest(backend="reference"))
    with s.db.session() as session:
        from agentx.storage.models import Video

        recording = session.get(Video, video["id"])
    source = s.catalog.require_source(recording)
    source.write_bytes(source.read_bytes() + b"changed after enqueue")
    s.indexer.process(run.id)
    with s.db.session() as session:
        from agentx.storage.models import IndexRun

        failed = session.get(IndexRun, run.id)
        assert failed.status == "failed"
        assert "no longer matches" in failed.error
        assert failed.observation_count == 0
