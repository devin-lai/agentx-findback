import io
import json
from zipfile import ZipFile

import pytest

from agentx.services.bundle_verify import verify
from agentx.storage.models import IndexRun, QueryRecord, Video


def ask(client, run, text="Show the history of Red toolkit", at_ms=7000):
    response = client.post(
        f"/api/v1/runs/{run['id']}/questions",
        json={"text": text, "at_ms": at_ms, "use_provider": False},
    )
    assert response.status_code == 200, response.text
    return response.json()


def extract(response, directory):
    assert response.status_code == 200, response.text[:500] if response.status_code != 200 else ""
    with ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(directory)
    return json.loads((directory / "report.json").read_text())


def test_offline_report_preserves_answer_cutoff_original_frames_and_old_version(
    client, app, indexed, tmp_path
):
    video, run = indexed
    answer = ask(client, run)
    regions = video["regions"]
    regions[0]["name"] = "Renamed after the answer"
    client.put(f"/api/v1/videos/{video['id']}/regions", json=regions)
    s = app.state.services
    with s.db.session.begin() as session:
        item = session.get(IndexRun, run["id"])
        item.provenance = {
            **item.provenance,
            "model_path": "/private/model",
            "endpoint": "http://private:secret@host",
            "unexpected_secret": "secret",
        }
        source = s.catalog.path(session.get(Video, video["id"]).source_path)
    response = client.get(f"/api/v1/questions/{answer['id']}/bundle")
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["cache-control"] == "no-store"
    output = tmp_path / "extracted"
    report = extract(response, output)
    assert report["answer"]["answer"] == answer["answer"]
    assert report["answer"]["as_of_ms"] == 7000
    assert report["source"]["sha256"] == video["sha256"]
    assert report["source"]["is_fixture"] is True
    assert report["run"]["regions"][0]["name"] == "Left area"
    assert {f["at_ms"] for f in report["frames"]} == {
        7000,
        *(e["at_ms"] for e in answer["evidence"]),
    }
    assert report["context_frame"]["role"] == "context_only"
    assert report["context_frame"]["at_ms"] == 7000
    assert "box" not in report["context_frame"]
    assert all(f["at_ms"] <= 7000 for f in report["frames"])
    for ref in report["answer"]["evidence"]:
        original = next(
            e for e in answer["evidence"] if e["observation_id"] == ref["observation_id"]
        )
        assert (output / ref["frame_url"]).read_bytes() == client.get(original["frame_url"]).content
        assert "media_url" not in ref
    content = (output / "report.html").read_text()
    assert "Generated software fixture" in content
    assert "not confirmed" in content
    assert "Renamed after the answer" not in content
    assert "http://private" not in content and "/private/model" not in content
    assert "unexpected_secret" not in json.dumps(report)
    assert verify(output, source)["status"] == "verified"
    assert not list(s.settings.data_dir.glob("evidence-*"))
    # The verifier also works as a standalone file without AgentX installed.
    import subprocess
    import sys

    checked = subprocess.run(
        [sys.executable, "-I", str(output / "verify.py")], capture_output=True, text=True
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr


def test_unknown_object_report_contains_no_frames_or_invented_location(client, indexed, tmp_path):
    _, run = indexed
    answer = ask(client, run, "Where is my passport?")
    report = extract(client.get(f"/api/v1/questions/{answer['id']}/bundle"), tmp_path / "report")
    assert report["frames"] == []
    assert report["answer"]["states"] == []
    assert "could not match" in report["answer"]["answer"]


@pytest.mark.parametrize(
    "fault",
    [
        "source_changed",
        "source_missing",
        "future_citation",
        "forged_answer",
        "wrong_cutoff",
        "wrong_run",
        "wrong_object",
    ],
)
def test_changed_evidence_and_invalid_saved_answers_never_produce_a_bundle(
    client, app, indexed, fault
):
    video, run = indexed
    answer = ask(client, run)
    s = app.state.services
    with s.db.session.begin() as session:
        source = s.catalog.path(session.get(Video, video["id"]).source_path)
        record = session.get(QueryRecord, answer["id"])
        response = json.loads(json.dumps(record.response))
        if fault == "source_changed":
            with source.open("ab") as stream:
                stream.write(b"changed")
        elif fault == "source_missing":
            source.unlink()
        elif fault == "future_citation":
            response["evidence"][0]["at_ms"] = 11000
        elif fault == "forged_answer":
            response["answer"] = "The toolkit is inside a drawer."
        elif fault == "wrong_cutoff":
            response["as_of_ms"] = 9000
        elif fault == "wrong_run":
            response["run_id"] = "another-run"
        elif fault == "wrong_object":
            response["states"][0]["object_id"] = video["objects"][1]["id"]
        record.response = response
    result = client.get(f"/api/v1/questions/{answer['id']}/bundle")
    assert result.status_code in {404, 422}, result.text
    assert not list(s.settings.data_dir.glob("evidence-*"))


def test_frame_timestamp_mismatch_is_rejected(client, indexed, monkeypatch):
    from agentx.services import evidence_bundle

    _, run = indexed
    answer = ask(client, run)
    original = evidence_bundle.read_frame

    def shifted(path, at_ms):
        pts, frame = original(path, at_ms)
        return pts + (1 if at_ms != 7000 else 0), frame

    monkeypatch.setattr(evidence_bundle, "read_frame", shifted)
    response = client.get(f"/api/v1/questions/{answer['id']}/bundle")
    assert response.status_code == 422 and "timestamp" in response.text


def test_export_limits_busy_auth_and_cleanup(client, app, indexed, monkeypatch):
    _, run = indexed
    answer = ask(client, run)
    url = f"/api/v1/questions/{answer['id']}/bundle"
    s = app.state.services
    with s.bundles.lock:
        assert client.get(url).status_code == 429
    with monkeypatch.context() as m:
        m.setattr("agentx.services.evidence_bundle.MAX_FRAMES", 0)
        assert client.get(url).status_code == 413
    with monkeypatch.context() as m:
        m.setattr("agentx.services.evidence_bundle.MAX_BYTES", 1)
        assert client.get(url).status_code == 413
    s.settings.api_token = "test-token"
    assert client.get(url).status_code == 401
    assert client.get(url, headers={"Authorization": "Bearer test-token"}).status_code == 200
    assert not list(s.settings.data_dir.glob("evidence-*"))


def test_verifier_detects_changed_missing_extra_files_and_wrong_source(client, indexed, tmp_path):
    _, run = indexed
    answer = ask(client, run)
    output = tmp_path / "report"
    report = extract(client.get(f"/api/v1/questions/{answer['id']}/bundle"), output)
    frame = output / report["frames"][0]["file"]
    original = frame.read_bytes()
    frame.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="Integrity"):
        verify(output)
    frame.unlink()
    with pytest.raises(ValueError, match="missing"):
        verify(output)
    frame.write_bytes(original)
    extra = output / "extra.txt"
    extra.write_text("unexpected")
    with pytest.raises(ValueError, match="unexpected"):
        verify(output)
    extra.unlink()
    with pytest.raises(ValueError, match="original video"):
        verify(output, frame)
    assert verify(output)["status"] == "verified"
    # Finder and Python add these when the report is opened; they are not report content.
    (output / ".DS_Store").write_bytes(b"finder")
    (output / "__pycache__").mkdir()
    (output / "__pycache__" / "verify.cpython-312.pyc").write_bytes(b"cache")
    assert verify(output)["status"] == "verified"
    (output / ".DS_Store").unlink()
    (output / ".DS_Store").symlink_to(frame)
    with pytest.raises(ValueError, match="unexpected"):
        verify(output)


def test_user_and_model_text_are_escaped_in_offline_html(client, app, indexed, tmp_path):
    _, run = indexed
    payload = '<img src="https://attacker.test/x" onerror="alert(1)"><script>alert(1)</script>'
    answer = ask(client, run, f"Where is Red toolkit? {payload}")
    with app.state.services.db.session.begin() as session:
        row = session.get(QueryRecord, answer["id"])
        response = json.loads(json.dumps(row.response))
        response["tools"].append({"name": "draft", "thought": payload})
        row.response = response
    output = tmp_path / "report"
    extract(client.get(f"/api/v1/questions/{answer['id']}/bundle"), output)
    html = (output / "report.html").read_text()
    assert payload not in html
    assert "&lt;script&gt;" in html
    assert "<script" not in html
    assert "default-src 'none'" in html


def test_a_recovered_region_is_labelled_in_the_offline_report():
    """An offline reader has no other way to tell a recovered region from a directly read one."""
    from agentx.services.evidence_bundle import render_report

    def report(scene_reference):
        return {
            "answer": {
                "question": "Where is Blue remote?",
                "answer": "At video 00:11.0, Blue remote is visible in Center area.",
                "as_of_ms": 11000,
                "intent": "location",
                "states": [],
                "events": [],
                "skills": [],
                "planner": "local",
                "planner_model": None,
                "elapsed_seconds": 0.1,
                "tools": [],
                "warnings": [],
                "evidence": [
                    {
                        "observation_id": "obs-1",
                        "at_ms": 11000,
                        "box": {"x1": 0.2, "y1": 0.4, "x2": 0.3, "y2": 0.6},
                        "zone": "center",
                        "scene_reference": scene_reference,
                        "frame_url": "frames/obs-1.jpg",
                    }
                ],
            },
            "source": {
                "title": "bumped",
                "sha256": "0" * 64,
                "duration_ms": 12000,
                "is_fixture": True,
            },
            "run": {
                "id": "run-1",
                "backend": "reference",
                "regions": [{"id": "center", "name": "Center area"}],
                "provenance": {},
                "created_at": "2000-01-01T00:00:00+00:00",
            },
            "context_frame": None,
            "frames": [{"file": "frames/obs-1.jpg"}],
            "created_at": "2000-01-01T00:00:00+00:00",
            "scope": "test",
            "limitations": "test",
        }

    recovered = render_report(report("compensated"))
    assert "Camera moved · region recovered from the registration view" in recovered
    unavailable = render_report(report("unavailable"))
    assert "no region could be named for this frame" in unavailable
    # A frame read at the registered pose carries no camera claim at all.
    assert "Camera moved" not in render_report(report("registered"))
