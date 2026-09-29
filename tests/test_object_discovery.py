"""Registration discovery proposes candidates; it must never become memory by itself."""

import json

import pytest

from agentx.domain.contracts import Box
from agentx.services.discovery import Discovery, parse_named_boxes


def test_parse_named_boxes_reads_names_and_scales():
    payload = json.dumps(
        {
            "object_1": {"name": "blue coffee mug", "box": [101, 336, 226, 500]},
            "object_2": {"name": "silver scissors", "box": [0.51, 0.62, 0.66, 0.81]},
        }
    )
    assert parse_named_boxes(payload) == [
        ("blue coffee mug", Box(x1=0.101, y1=0.336, x2=0.226, y2=0.5)),
        ("silver scissors", Box(x1=0.51, y1=0.62, x2=0.66, y2=0.81)),
    ]


def test_parse_named_boxes_skips_malformed_entries_without_inventing_one():
    payload = json.dumps(
        {
            "object_1": {"name": "good mug", "box": [0.1, 0.1, 0.4, 0.4]},
            "object_2": {"name": "no box"},
            "object_3": {"box": [0.1, 0.1, 0.2, 0.2]},
            "object_4": {"name": "inverted", "box": [0.9, 0.9, 0.1, 0.1]},
            "object_5": {"name": "short", "box": [0.1, 0.2]},
        }
    )
    assert parse_named_boxes(payload) == [("good mug", Box(x1=0.1, y1=0.1, x2=0.4, y2=0.4))]


def test_parse_named_boxes_rejects_non_json():
    with pytest.raises(ValueError):
        parse_named_boxes("the desk has a mug on it")


def named(name):
    return type("Obj", (), {"name": name})()


def discovery(settings):
    return Discovery(catalog=None, providers=None, settings=settings)


def test_bounded_proposals_drop_tiny_boxes_and_near_duplicates(app):
    service = discovery(app.state.services.settings)
    candidates = [
        ("cup", Box(x1=0.1, y1=0.1, x2=0.3, y2=0.3), 0.95, "cup"),
        # A near-duplicate of the first box must not become a second proposal.
        ("cup", Box(x1=0.11, y1=0.11, x2=0.31, y2=0.31), 0.81, "cup"),
        # Too small to be a usable visual reference.
        ("mouse", Box(x1=0.5, y1=0.5, x2=0.505, y2=0.505), 0.77, "mouse"),
        ("book", Box(x1=0.6, y1=0.2, x2=0.9, y2=0.5), 0.66, "book"),
    ]
    proposals = service._bound(candidates, [], width=640, height=360)
    assert [p["suggested_name"] for p in proposals] == ["Cup", "Book"]
    assert proposals[0]["score"] == 0.95


def test_bounded_proposals_never_reuse_a_registered_name(app):
    service = discovery(app.state.services.settings)
    candidates = [
        ("cup", Box(x1=0.1, y1=0.1, x2=0.3, y2=0.3), 0.9, "cup"),
        ("cup", Box(x1=0.6, y1=0.6, x2=0.8, y2=0.8), 0.8, "cup"),
    ]
    proposals = service._bound(candidates, [named("Cup")], width=640, height=360)
    assert [p["suggested_name"] for p in proposals] == ["Cup 2", "Cup 3"]


def test_suggestions_require_a_configured_backend(client):
    video = client.post("/api/v1/demo").json()
    result = client.post(
        f"/api/v1/videos/{video['id']}/suggestions", json={"at_ms": 0, "backend": "cosmos"}
    )
    assert result.status_code == 503, result.text


def test_suggestions_reject_a_timestamp_outside_the_video(client):
    video = client.post("/api/v1/demo").json()
    result = client.post(
        f"/api/v1/videos/{video['id']}/suggestions",
        json={"at_ms": video["duration_ms"] + 1, "backend": "cosmos"},
    )
    assert result.status_code == 422, result.text


def test_cosmos_suggestions_propose_without_registering_anything(client, app):
    s = app.state.services
    s.settings.cosmos_base_url = "http://127.0.0.1:65535/v1"
    s.settings.cosmos_model = "cosmos-test"
    listing = json.dumps(
        {
            "object_1": {"name": "red toolkit", "box": [0.1, 0.34, 0.24, 0.52]},
            "object_2": {"name": "blue remote", "box": [0.53, 0.65, 0.67, 0.83]},
        }
    )
    s.providers.describe_frame = lambda system, image, **kwargs: listing
    s.providers.provenance = lambda: {"backend": "http", "endpoint": "test"}
    video = client.post("/api/v1/demo").json()

    result = client.post(
        f"/api/v1/videos/{video['id']}/suggestions", json={"at_ms": 1000, "backend": "cosmos"}
    )
    assert result.status_code == 200, result.text
    body = result.json()
    # The fixture already registers both names, so a proposal must not collide with them.
    assert [p["suggested_name"] for p in body["proposals"]] == ["Red toolkit 2", "Blue remote 2"]
    # Open-vocabulary names are not COCO categories, so they stay on reference tracking.
    assert {p["label"] for p in body["proposals"]} == {"custom"}
    assert body["backend"] == "cosmos"
    assert body["limits"]

    # The decisive property: proposing registered nothing and changed no memory.
    assert [o["name"] for o in client.get(f"/api/v1/videos/{video['id']}").json()["objects"]] == [
        o["name"] for o in video["objects"]
    ]


def test_capabilities_reports_discovery_backends(client):
    body = client.get("/api/v1/capabilities").json()
    assert set(body["discovery"]) == {"rtdetr", "cosmos"}
    assert body["discovery"]["cosmos"] is False


def test_shared_proposer_is_cached_per_configuration(app, monkeypatch):
    """A cached model must not outlive the settings it was built from."""
    from agentx.vision import rtdetr

    built = []

    class FakeProposer:
        def __init__(self, settings):
            built.append(
                (settings.detector_model, settings.detector_revision, settings.model_device)
            )

    monkeypatch.setattr(rtdetr.RTDetrProposer, "_cache", None)
    monkeypatch.setattr(rtdetr.RTDetrProposer, "_cache_key", None)
    monkeypatch.setattr(rtdetr.RTDetrProposer, "__init__", FakeProposer.__init__)

    settings = app.state.services.settings
    first = rtdetr.RTDetrProposer.shared(settings)
    assert rtdetr.RTDetrProposer.shared(settings) is first
    assert len(built) == 1

    settings.model_device = "cpu"
    second = rtdetr.RTDetrProposer.shared(settings)
    assert second is not first
    assert len(built) == 2


def configured_cosmos(app):
    s = app.state.services
    s.settings.cosmos_base_url = "http://cosmos.invalid/v1"
    s.settings.cosmos_model = "cosmos-test"
    return s


@pytest.mark.parametrize(
    "failure",
    [
        # The inference service rejected the request and explained why, naming itself.
        ValueError(
            "cosmos-test rejected the request with HTTP 500: upstream http://10.0.0.5:8002 failed"
        ),
        # The service could not be reached at all.
        ConnectionError("All connection attempts failed"),
    ],
)
def test_a_backend_failure_is_a_sanitized_gateway_error(client, app, failure):
    """A model or service failure is not the caller's mistake, and its text is not publishable."""
    s = configured_cosmos(app)

    def fail(*args, **kwargs):
        raise failure

    s.providers.describe_frame = fail
    video = client.post("/api/v1/demo").json()

    result = client.post(
        f"/api/v1/videos/{video['id']}/suggestions", json={"at_ms": 1000, "backend": "cosmos"}
    )

    assert result.status_code == 502, result.text
    detail = result.json()["detail"]
    assert "Nothing was registered." in detail
    assert "10.0.0.5" not in detail and "cosmos.invalid" not in detail
    assert "HTTP 500" not in detail and "connection" not in detail.casefold()


def test_an_unparseable_listing_is_a_gateway_error_not_a_client_error(client, app):
    s = configured_cosmos(app)
    s.providers.describe_frame = lambda *a, **k: "Sure! I can see a mug and a book."
    video = client.post("/api/v1/demo").json()

    result = client.post(
        f"/api/v1/videos/{video['id']}/suggestions", json={"at_ms": 1000, "backend": "cosmos"}
    )

    assert result.status_code == 502, result.text


def test_missing_original_evidence_is_reported_to_the_caller(client, app):
    """A data problem the person can act on keeps its own message, ahead of the backend call."""
    s = configured_cosmos(app)
    s.providers.describe_frame = lambda *a, **k: pytest.fail("the source is checked first")
    video = client.post("/api/v1/demo").json()
    with s.db.session() as session:
        from agentx.storage.models import Video

        s.catalog.path(session.get(Video, video["id"]).source_path).unlink()

    result = client.post(
        f"/api/v1/videos/{video['id']}/suggestions", json={"at_ms": 1000, "backend": "cosmos"}
    )

    assert result.status_code == 422 and "Original evidence is missing." in result.text


def test_a_busy_cosmos_runtime_still_asks_the_caller_to_retry(client, app):
    from agentx.agents.providers import ReviewBusy

    s = configured_cosmos(app)

    def busy(*args, **kwargs):
        raise ReviewBusy("A Cosmos request is already running. Try again when it completes.")

    s.providers.describe_frame = busy
    video = client.post("/api/v1/demo").json()

    result = client.post(
        f"/api/v1/videos/{video['id']}/suggestions", json={"at_ms": 1000, "backend": "cosmos"}
    )

    assert result.status_code == 409 and "already running" in result.json()["detail"]
