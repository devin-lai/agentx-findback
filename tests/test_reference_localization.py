import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from agentx.agents.localization import normalize_integer_report
from agentx.agents.providers import Providers
from agentx.config import Settings

spec = importlib.util.spec_from_file_location(
    "grounding_research", Path(__file__).parents[1] / "scripts/eval/evaluate_grounded_reviews.py"
)
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


@pytest.mark.parametrize("coordinate", [0, 1, 500, 1000])
def test_integer_units_match_frozen_research_and_do_not_mutate_raw(coordinate):
    raw = dict(present=True, x=coordinate, y=1000, note="visible")
    assert normalize_integer_report(raw) == json.loads(
        research.normalize_integer_points(json.dumps(raw))
    )
    assert normalize_integer_report(raw)["x"] == coordinate / 1000
    assert raw["x"] == coordinate and raw["y"] == 1000


@pytest.mark.parametrize("bad", [0.5, 500.0, "500", True, -1, 1001, None])
def test_ambiguous_or_out_of_range_coordinate_is_rejected(bad):
    with pytest.raises(ValueError):
        normalize_integer_report(dict(present=True, x=bad, y=500, note="visible"))


@pytest.mark.parametrize(
    "patch",
    [{"present": 1}, {"present": False}, {"note": None}, {"extra": 1}],
)
def test_integer_schema_rejects_coercion_and_nonnull_absence(patch):
    with pytest.raises(ValueError):
        normalize_integer_report(dict(present=True, x=500, y=500, note="visible") | patch)
    absent = dict(present=False, x=None, y=None, note="unclear")
    assert normalize_integer_report(absent) == absent


def test_application_requests_match_frozen_adapter_including_retry(monkeypatch):
    frame = np.zeros((600, 900, 3), dtype=np.uint8)
    frame[100:200, 100:200] = 230
    monkeypatch.setattr("agentx.agents.providers.frames", lambda *a, **kw: [(2000, frame)])
    settings = Settings(cosmos_max_edge=768, _env_file=None)
    provider = Providers(settings)
    baseline_calls, integer_calls = [], []

    def baseline(messages, **kwargs):
        baseline_calls.append(copy.deepcopy(messages))
        return '{"present":true,"x":0.25,"y":0.75,"note":"visible"}'

    monkeypatch.setattr(provider, "_generate", baseline)
    original = provider.review(
        "unused", 2000, "Where?", ["Skill policy."], target="cup", reference=b"ref"
    )
    settings.cosmos_max_edge = 448
    settings.cosmos_reference_grounding = "integer_1000"
    settings.cosmos_reference_max_edge = 768

    def integer(messages, **kwargs):
        integer_calls.append(copy.deepcopy(messages))
        assert kwargs == {"max_tokens": 200, "json_mode": True}
        return (
            '{"present":true,"x":0.25,"y":0.75,"note":"wrong units"}'
            if len(integer_calls) == 1
            else '{"present":true,"x":250,"y":750,"note":"visible"}'
        )

    monkeypatch.setattr(provider, "_generate", integer)
    result = provider.review(
        "unused", 2000, "Where?", ["Skill policy."], target="cup", reference=b"ref"
    )
    assert integer_calls[0] == research.variant_messages(baseline_calls[0], "integer_points")
    repair = {
        "role": "user",
        "content": 'Invalid output. Return only {"present":bool,"x":number|null,"y":number|null,"note":string}.',
    }
    assert integer_calls[1] == research.variant_messages(
        baseline_calls[0] + [repair], "integer_points"
    )
    assert result["frames"] == original["frames"]
    assert result["frame_reports"] == original["frame_reports"]
    provenance = result["provenance"]
    assert provenance["attempts"] == 2 and provenance["max_edge"] == 768
    assert provenance["coordinate_contract"] == "reference_integer_1000_v1"
    assert (
        provenance["prompt_sha256"]
        == hashlib.sha256(integer_calls[0][0]["content"].encode()).hexdigest()
    )
    assert not result["authoritative"]


def test_override_is_scoped_to_reference_grounding(monkeypatch):
    frame = np.zeros((600, 900, 3), dtype=np.uint8)
    monkeypatch.setattr("agentx.agents.providers.frames", lambda *a, **kw: [(0, frame)])
    provider = Providers(
        Settings(
            cosmos_reference_grounding="integer_1000", cosmos_reference_max_edge=768, _env_file=None
        )
    )
    calls = []

    def generate(messages, **kwargs):
        calls.append(copy.deepcopy(messages))
        return '{"present":true,"x":0.25,"y":0.75,"note":"visible"}'

    monkeypatch.setattr(provider, "_generate", generate)
    result = provider.review("unused", 0, "Where?", target="cup")
    assert result["provenance"]["max_edge"] == 448
    assert result["provenance"]["coordinate_contract"] == "normalized_v1"
    assert "Image 1 is only" not in calls[0][0]["content"]
    image = next(p for p in calls[0][1]["content"] if p["type"] == "image_url")
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(image["image_url"]["url"].split(",")[1]), np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert max(decoded.shape[:2]) == 448
    monkeypatch.setattr(
        provider,
        "_generate",
        lambda *a, **kw: (
            '{"summary":"Visible at 0 ms.","evidence_frame_ids":[0],"uncertainty":"Only one frame."}'
        ),
    )
    freeform = provider.review("unused", 0, "What changed?")
    assert freeform["provenance"]["max_edge"] == 448
    assert freeform["frames"] == result["frames"]


def test_reference_adapter_uses_base_presence_and_keeps_freeform_on_base(monkeypatch):
    provider = Providers(
        Settings(
            cosmos_backend="http",
            cosmos_base_url="http://review/v1",
            cosmos_model="base",
            cosmos_reference_adapter_model="v1",
            _env_file=None,
        )
    )
    evidence = [{"id": 0, "at_ms": 0}, {"id": 1, "at_ms": 1000}]
    monkeypatch.setattr(
        provider,
        "_sample_frames",
        lambda *a, **kw: (evidence, [b"frame"] * 2, [None] * 2),
    )
    monkeypatch.setattr(provider, "_sample", lambda *a, **kw: (evidence, [b"frame"] * 2))
    monkeypatch.setattr(
        provider,
        "_provenance",
        lambda: {"served_model": {"id": "base", "parent": None, "adapter": False}},
    )
    monkeypatch.setattr(
        provider,
        "served_model",
        lambda model=None: {"id": model, "parent": "base", "adapter": True},
    )
    calls = []

    def generate(messages, **kwargs):
        model = kwargs.get("model", "base")
        calls.append(model)
        if "review ordered visual evidence" in messages[0]["content"]:
            return '{"summary":"The bottle is visible.","evidence_frame_ids":[0],"uncertainty":"One sample."}'
        frame_id = next(
            p["text"] for p in messages[1]["content"] if p.get("text", "").startswith("frame_id=")
        )
        if model == "base" and frame_id.startswith("frame_id=1"):
            return '{"present":false,"x":null,"y":null,"note":"not visible"}'
        x = 0.8 if model == "v1" else 0.1
        return json.dumps({"present": True, "x": x, "y": 0.5, "note": "visible"})

    monkeypatch.setattr(provider, "_generate", generate)
    result = provider.review("unused", 1000, "Where?", target="bottle", reference=b"ref")
    assert result["model"] == "v1"
    assert [row["present"] for row in result["frame_reports"]] == [True, False]
    assert result["frame_reports"][0]["x"] == 0.8
    assert result["frame_reports"][1]["presence_veto"]["adapter_queried"] is False
    assert calls.count("base") == 2 and calls.count("v1") == 1
    assert result["provenance"]["served_model"]["id"] == "v1"
    assert result["provenance"]["served_presence_model"]["id"] == "base"
    freeform = provider.review("unused", 1000, "What changed?")
    assert freeform["model"] == "base" and calls[-1] == "base"


def test_base_point_rule_uses_the_adapter_only_as_an_identity_veto(monkeypatch):
    provider = Providers(
        Settings(
            cosmos_backend="http",
            cosmos_base_url="http://review/v1",
            cosmos_model="base",
            cosmos_reference_adapter_model="v3",
            cosmos_reference_point_source="base",
            _env_file=None,
        )
    )
    evidence = [{"id": i, "at_ms": i * 1000} for i in range(3)]
    monkeypatch.setattr(
        provider, "_sample_frames", lambda *a, **kw: (evidence, [b"frame"] * 3, [None] * 3)
    )
    monkeypatch.setattr(
        provider,
        "_provenance",
        lambda: {"served_model": {"id": "base", "parent": None, "adapter": False}},
    )
    monkeypatch.setattr(
        provider, "served_model", lambda model=None: {"id": model, "parent": "base"}
    )

    def generate(messages, **kwargs):
        model = kwargs.get("model", "base")
        frame_id = next(
            p["text"] for p in messages[1]["content"] if p.get("text", "").startswith("frame_id=")
        )
        if model == "v3" and frame_id.startswith("frame_id=2"):
            return '{"present":false,"x":null,"y":null,"note":"a different object"}'
        x = 0.8 if model == "v3" else 0.1
        return json.dumps({"present": True, "x": x, "y": 0.5, "note": "visible"})

    monkeypatch.setattr(provider, "_generate", generate)
    result = provider.review("unused", 2000, "Where?", target="bottle", reference=b"ref")
    reports = result["frame_reports"]
    # Both present: the base model's point. Adapter says absent: the frame is not the object.
    assert [(r["present"], r["x"]) for r in reports] == [(True, 0.1), (True, 0.1), (False, None)]
    assert reports[2]["presence_veto"] == {
        "base_present": True,
        "adapter_queried": True,
        "adapter_present": False,
    }
    assert result["provenance"]["fusion_rule"] == "base_present_and_adapter_present_then_base_point"


def test_integer_review_persists_exact_frame_across_setting_changes(
    client, app, indexed, monkeypatch
):
    video, run = indexed
    services = app.state.services
    settings = services.settings
    settings.cosmos_base_url = "http://example.invalid/v1"
    settings.cosmos_reference_grounding = "integer_1000"
    settings.cosmos_reference_max_edge = 768
    monkeypatch.setattr(
        services.providers,
        "_generate",
        lambda *a, **kw: '{"present":true,"x":1,"y":1000,"note":"visible"}',
    )
    response = client.post(
        f"/api/v1/runs/{run['id']}/review",
        json={"text": "Where?", "at_ms": 0, "object_id": video["objects"][0]["id"]},
    )
    assert response.status_code == 200, response.text
    review = response.json()
    assert review["frame_reports"][0]["x"] == 0.001
    assert review["provenance"]["coordinate_divisor"] == 1000
    settings.cosmos_reference_grounding = "normalized"
    settings.cosmos_reference_max_edge = None
    settings.cosmos_max_edge = 224
    saved = client.get(f"/api/v1/runs/{run['id']}/reviews").json()[0]
    assert saved == review
    image = client.get(saved["frames"][0]["frame_url"])
    assert image.status_code == 200
    assert hashlib.sha256(image.content).hexdigest() == saved["frames"][0]["sha256"]


def test_grounding_prompt_matches_the_deployed_contract_bytes():
    """Fine-tuning data and serving share these builders; the bytes must not drift.

    The hashes are the system prompts recorded by the deployed Cosmos-Reason2-8B service in the
    earlier model bake-off receipts (integer reference contract and normalized contract).
    """
    from agentx.agents.providers import grounding_messages, grounding_system
    from agentx.agents.skills import SkillRegistry

    body, _ = SkillRegistry().load("review-visual-evidence")
    integer = grounding_system([body], integer_points=True)
    normalized = grounding_system([body], integer_points=False)
    assert hashlib.sha256(integer.encode()).hexdigest() == (
        "613f71a92239587f2fdfda8a4d53229e714d5cf69aea8b785f1d1fd1e4f482ad"
    )
    assert hashlib.sha256(normalized.encode()).hexdigest() == (
        "71560ff6e5dad16dca59a1a1be8771c976b927bcaff2059658ef8dca0441e731"
    )
    entry = {"id": 3, "at_ms": 4000}
    messages = grounding_messages(
        integer, b"frame", entry, "Registered cup", b"ref", integer_points=True
    )
    texts = [part.get("text") for part in messages[1]["content"]]
    images = [part["image_url"]["url"] for part in messages[1]["content"] if "image_url" in part]
    assert texts[0] == "IDENTITY REFERENCE ONLY (not the source frame):"
    assert "frame_id=3; video_time_ms=4000" in texts
    assert images == [
        "data:image/jpeg;base64," + base64.b64encode(b"ref").decode(),
        "data:image/jpeg;base64," + base64.b64encode(b"frame").decode(),
    ]
    plain = grounding_messages(normalized, b"frame", entry, "cup", None, integer_points=False)
    assert [p["type"] for p in plain[1]["content"]] == ["text", "image_url", "text"]


def test_review_provenance_names_a_lora_adapter_and_its_base(monkeypatch):
    import httpx

    calls = []

    def get(self, url, headers=None):
        calls.append(url)
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={
                "data": [
                    {"id": "nvidia/Cosmos-Reason2-8B", "root": "/m", "parent": None},
                    {
                        "id": "findback-grounding-v1",
                        "root": "/runs/adapter",
                        "parent": "nvidia/Cosmos-Reason2-8B",
                    },
                ]
            },
        )

    monkeypatch.setattr(httpx.Client, "get", get)
    settings = Settings(
        cosmos_base_url="http://review/v1", cosmos_model="findback-grounding-v1", _env_file=None
    )
    provider = Providers(settings)
    served = provider._provenance()["served_model"]
    assert served == {
        "id": "findback-grounding-v1",
        "parent": "nvidia/Cosmos-Reason2-8B",
        "adapter": True,
    }
    provider._provenance()
    assert calls == ["http://review/v1/models"]  # cached after one successful answer

    base = Providers(
        Settings(
            cosmos_base_url="http://review/v1",
            cosmos_model="nvidia/Cosmos-Reason2-8B",
            _env_file=None,
        )
    )
    assert base.served_model() == {
        "id": "nvidia/Cosmos-Reason2-8B",
        "parent": None,
        "adapter": False,
    }


def test_unreachable_review_service_is_not_asked_on_every_page_load(monkeypatch):
    import httpx

    calls = []

    def down(self, url, headers=None):
        calls.append(url)
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx.Client, "get", down)
    provider = Providers(
        Settings(cosmos_base_url="http://review/v1", cosmos_model="m", _env_file=None)
    )
    assert provider.served_model() is None
    assert provider.served_model() is None
    assert "served_model" not in provider._provenance()
    assert len(calls) == 1
