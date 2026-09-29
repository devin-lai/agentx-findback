import json
import sys
from types import SimpleNamespace

import pytest

from agentx.domain.contracts import Box
from agentx.vision.cosmos_detector import normalize_box
from agentx.vision.identity import AppearanceMatcher as RealMatcher


def test_normalize_box_accepts_unit_and_thousand_scales():
    assert normalize_box([101, 336, 226, 500]) == Box(x1=0.101, y1=0.336, x2=0.226, y2=0.5)
    assert normalize_box([0.51, 0.62, 0.66, 0.81]) == Box(x1=0.51, y1=0.62, x2=0.66, y2=0.81)
    with pytest.raises(ValueError):
        normalize_box([0.6, 0.6, 0.5, 0.5])
    with pytest.raises(ValueError):
        normalize_box([float("nan"), 0, 1, 1])


def scripted_cosmos(fail_at_ms=None):
    """Answer the fixture's ground truth per category from the video time; never sees pixels."""

    def chat(base, key, model, messages, **kwargs):
        if isinstance(messages[-1]["content"], str):  # the bounded retry after invalid output
            return "still not json"
        texts = [c["text"] for c in messages[-1]["content"] if c["type"] == "text"]
        at_ms = int(next(t for t in texts if t.startswith("video_time_ms=")).split("=")[1])
        category = next(t for t in texts if t.startswith("Category")).split(": ", 1)[1]
        if fail_at_ms is not None and at_ms == fail_at_ms:
            return "garbage"
        if category.startswith("remote"):
            # A degenerate extra box is ignored; the valid one survives.
            return json.dumps({"boxes": [[532, 652, 667, 832], [0.9, 0.9, 0.1, 0.1]]})
        if at_ms < 3000:
            return json.dumps({"boxes": [[101, 347, 242, 525]]})
        if at_ms < 6000:
            return json.dumps({"boxes": [[0.75, 0.35, 0.89, 0.53]]})
        if at_ms < 8000:
            return json.dumps({"boxes": []})
        return json.dumps({"boxes": [[437, 347, 578, 525]]})

    return chat


def calls_per_frame(export):
    return sorted({o["at_ms"] for o in export["observations"]})


def test_cosmos_backend_builds_memory_from_per_frame_localizations(client, app, monkeypatch):
    s = app.state.services
    video = client.post("/api/v1/demo").json()
    assert (
        client.post(f"/api/v1/videos/{video['id']}/runs", json={"backend": "cosmos"}).status_code
        == 422
    )
    s.settings.cosmos_base_url = "http://cosmos.invalid/v1"
    monkeypatch.setattr(s.providers, "_chat", scripted_cosmos(fail_at_ms=5000))
    run = client.post(
        f"/api/v1/videos/{video['id']}/runs", json={"backend": "cosmos", "sample_fps": 5}
    ).json()
    assert run["config"]["sample_fps"] == 1.0  # clamped to the Cosmos indexing rate
    s.indexer.process(run["id"])
    run = client.get(f"/api/v1/runs/{run['id']}").json()
    assert run["status"] == "complete", run
    assert run["provenance"]["adapter"] == "cosmos-grounded"
    assert run["provenance"]["reference_crop"] and run["provenance"]["learned_model"]
    toolkit = next(o for o in video["objects"] if o["name"] == "Red toolkit")
    states = {x["name"]: x for x in client.get(f"/api/v1/runs/{run['id']}/state?at_ms=7000").json()}
    assert (
        states["Red toolkit"]["status"] == "last_seen" and states["Red toolkit"]["zone"] == "right"
    )
    assert (
        states["Blue remote"]["status"] == "visible" and states["Blue remote"]["zone"] == "center"
    )
    export = client.get(f"/api/v1/runs/{run['id']}/export").json()
    observations = [o for o in export["observations"] if o["object_id"] == toolkit["id"]]
    assert len(observations) == 12  # one per second of the twelve-second fixture
    by_time = {o["at_ms"]: o for o in observations}
    assert by_time[0]["visible"] and by_time[0]["zone"] == "left"
    assert by_time[0]["box"] == {"x1": 0.101, "y1": 0.347, "x2": 0.242, "y2": 0.525}
    assert by_time[4000]["visible"] and by_time[4000]["zone"] == "right"
    # The scripted failure at 5 s becomes a missing observation, never an invented box.
    assert not by_time[5000]["visible"] and by_time[5000]["reason"] == "not_detected"
    assert len(calls_per_frame(export)) == 12
    assert not by_time[6000]["visible"]
    assert by_time[9000]["zone"] == "center"
    assert all(o["score"] is None for o in observations)
    kinds = [(e["kind"], e["zone"]) for e in export["events"] if e["object_id"] == toolkit["id"]]
    assert kinds[:2] == [("appeared", "left"), ("moved", "right")]
    assert ("reappeared", "center") in kinds
    answer = s.workflow.answer(
        run["id"],
        __import__("agentx.domain.contracts", fromlist=["Question"]).Question(
            text="Where was Red toolkit last seen?", at_ms=7000, use_provider=False
        ),
    )
    assert "Right area" in answer["answer"] and answer["evidence"][0]["at_ms"] == 4000


def test_cosmos_candidates_are_assigned_by_identity(client, app, monkeypatch):
    """Look-alikes: the model lists both instances; DINOv2 assigns each registered object."""
    import numpy as np

    from agentx.vision import cosmos_detector

    class FakeMatcher:
        threshold, margin = 0.6, 0.05
        provenance = {"identity_model": "fake"}

        def __init__(self, settings, device):
            pass

        def embed(self, crops):
            out = []
            for c in crops:
                b, g, r = (float(x) for x in c.reshape(-1, 3).mean(axis=0))
                out.append(np.array([1.0, 0.0]) if r > b else np.array([0.0, 1.0]))
            return np.stack(out)

        def assign(self, references, candidates):
            return RealMatcher.assign(self, references, candidates)

    monkeypatch.setattr("agentx.vision.identity.AppearanceMatcher", FakeMatcher)
    # This contract test uses synthetic embeddings, so isolate the hardware probe too.
    # The core contributor environment intentionally does not install optional PyTorch.
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    s = app.state.services
    s.settings.cosmos_base_url = "http://cosmos.invalid/v1"
    s.settings.identity_enabled = True
    video = client.post("/api/v1/demo").json()
    # Register a second "remote": a look-alike of the blue one (same category, same crop).
    remote = next(o for o in video["objects"] if o["name"] == "Blue remote")
    twin = client.post(
        f"/api/v1/videos/{video['id']}/objects",
        json={"name": "Twin remote", "label": "remote", "at_ms": 0, "box": remote["box"]},
    ).json()

    def lists_every_instance(base, key, model, messages, **kwargs):
        texts = [c["text"] for c in messages[-1]["content"] if c["type"] == "text"]
        category = next(t for t in texts if t.startswith("Category")).split(": ", 1)[1]
        if category.startswith("remote"):
            # The blue remote plus the red toolkit's box, which is not a remote at all.
            return json.dumps({"boxes": [[532, 652, 667, 832], [101, 347, 242, 525]]})
        return json.dumps({"boxes": [[101, 347, 242, 525]]})

    monkeypatch.setattr(s.providers, "_chat", lists_every_instance)
    run = client.post(f"/api/v1/videos/{video['id']}/runs", json={"backend": "cosmos"}).json()
    s.indexer.process(run["id"])
    run = client.get(f"/api/v1/runs/{run['id']}").json()
    assert run["status"] == "complete", run
    assert run["provenance"]["adapter"] == "cosmos-grounded-dinov2"
    states = {x["name"]: x for x in client.get(f"/api/v1/runs/{run['id']}/state?at_ms=2000").json()}
    assert states["Red toolkit"]["status"] == "visible" and states["Red toolkit"]["zone"] == "left"
    # Blue remote and its twin have identical references: one candidate matches both equally,
    # so neither may silently win.
    assert states["Blue remote"]["status"] == "unknown"
    assert states["Blue remote"]["reason"] == "identity_ambiguous"
    assert states["Twin remote"]["status"] == "unknown"
    export = client.get(f"/api/v1/runs/{run['id']}/export").json()
    twin_obs = [o for o in export["observations"] if o["object_id"] == twin["id"]]
    assert twin_obs and all(o["identity_score"] is not None for o in twin_obs)
    assert cosmos_detector.iou(Box(x1=0, y1=0, x2=1, y2=1), Box(x1=0.5, y1=0.5, x2=1, y2=1)) == 0.25
    assert (
        len(
            cosmos_detector.distinct(
                [Box(x1=0, y1=0, x2=0.5, y2=0.5), Box(x1=0.01, y1=0, x2=0.5, y2=0.5)]
            )
        )
        == 1
    )


def test_cosmos_backend_requires_identity_for_shared_categories(client, app):
    s = app.state.services
    s.settings.cosmos_base_url = "http://cosmos.invalid/v1"
    video = client.post("/api/v1/demo").json()
    remote = next(o for o in video["objects"] if o["name"] == "Blue remote")
    client.post(
        f"/api/v1/videos/{video['id']}/objects",
        json={"name": "Twin remote", "label": "remote", "at_ms": 0, "box": remote["box"]},
    )
    run = client.post(f"/api/v1/videos/{video['id']}/runs", json={"backend": "cosmos"}).json()
    s.indexer.process(run["id"])
    run = client.get(f"/api/v1/runs/{run['id']}").json()
    assert run["status"] == "failed" and "identity" in run["error"]


def test_parse_boxes_tolerates_model_quirks():
    from agentx.vision.cosmos_detector import parse_boxes

    numbered = parse_boxes('{"box_1": [101, 211, 205, 562], "box_2": [0.36, 0.61, 0.46, 0.9]}')
    assert [(round(b.x1, 2), round(b.y2, 2)) for b in numbered] == [(0.1, 0.56), (0.36, 0.9)]
    # Repeated keys (a grammar-constrained habit) are all kept.
    repeated = parse_boxes('{"boxes": [101, 211, 205, 562], "boxes": [362, 610, 462, 905]}')
    assert len(repeated) == 2
    # A looping flat list collapses to its distinct boxes.
    flat = parse_boxes(
        '{"boxes": [0.41, 0.61, 0.52, 0.91, 0.52, 0.91, 0.52, 0.91, 0.41, 0.61, 0.52, 0.91]}'
    )
    assert len(flat) == 1
    assert parse_boxes("{}") == []
    assert parse_boxes('{"box_1": "none", "note": "x"}') == []
    with pytest.raises(ValueError):
        parse_boxes("no json here")
