"""Protocol-level tests for the standard-library FindBack MCP server (no network)."""

import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "findback_mcp", Path(__file__).parents[1] / "integrations/mcp/findback_mcp.py"
)
mcp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp)


def request(method, params=None, ident=1):
    return {"jsonrpc": "2.0", "id": ident, "method": method, "params": params or {}}


def test_initialize_negotiates_a_supported_protocol_version():
    reply = mcp.handle(request("initialize", {"protocolVersion": "2025-03-26"}))
    assert reply["result"]["protocolVersion"] == "2025-03-26"
    assert reply["result"]["capabilities"] == {"tools": {"listChanged": False}}
    unknown = mcp.handle(request("initialize", {"protocolVersion": "1999-01-01"}))
    assert unknown["result"]["protocolVersion"] == mcp.PROTOCOL_VERSIONS[0]


def test_notifications_get_no_reply_and_unknown_methods_are_errors():
    assert mcp.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert mcp.handle(request("bogus"))["error"]["code"] == -32601


def test_tools_are_listed_with_schemas_and_validated_before_any_call(monkeypatch):
    tools = mcp.handle(request("tools/list"))["result"]["tools"]
    assert [t["name"] for t in tools] == ["list_recordings", "ask_memory", "get_evidence_frame"]
    called = []
    monkeypatch.setattr(mcp, "call", lambda name, args: called.append(name) or [])
    assert mcp.handle(request("tools/call", {"name": "nope"}))["error"]["code"] == -32602
    missing = mcp.handle(request("tools/call", {"name": "ask_memory", "arguments": {}}))
    assert "recording, question" in missing["error"]["message"]
    assert called == []


def test_server_errors_are_tool_errors_with_the_token_redacted(monkeypatch):
    monkeypatch.setenv("FINDBACK_TOKEN", "fbk-secret-123")  # gitleaks:allow - synthetic fixture
    mcp.client()  # registers the token for redaction

    def fail(name, args):
        raise mcp.fb.CliError("upstream said Bearer fbk-secret-123 is invalid")

    monkeypatch.setattr(mcp, "call", fail)
    reply = mcp.handle(request("tools/call", {"name": "list_recordings", "arguments": {}}))
    text = reply["result"]["content"][0]["text"]
    assert reply["result"]["isError"] is True
    assert "fbk-secret-123" not in json.dumps(reply)
    assert "upstream said" in text


class FakeApi:
    def __init__(self, video):
        self.video, self.asked = video, []

    def json(self, method, path, body=None):
        if path == "/api/v1/videos":
            return [self.video]
        self.asked.append((path, body))
        return {
            "id": "q1",
            "run_id": path.split("/")[4],
            "question": body["text"],
            "answer": "ok",
            "intent": "location",
            "as_of_ms": 0,
            "states": [],
            "evidence": [],
            "warnings": [],
        }


def fake_video(live_status, runs):
    return {
        "id": "v1",
        "title": "Bench",
        "live_status": live_status,
        "duration_ms": 9000,
        "objects": [{"id": "o1", "name": "Drill"}],
        "regions": [],
        "runs": [
            {"object_ids": ["o1"], "regions": [], **run} for run in runs
        ],  # the server lists runs newest first
    }


def test_ask_memory_uses_the_newest_complete_run(monkeypatch):
    api = FakeApi(
        fake_video(
            None,
            [
                {"id": "new", "status": "complete", "created_at": "2000-01-02T02:00:00"},
                {"id": "old", "status": "complete", "created_at": "2000-01-01T02:00:00"},
            ],
        )
    )
    monkeypatch.setattr(mcp, "client", lambda: api)
    mcp.call("ask_memory", {"recording": "Bench", "question": "Where is Drill?"})
    assert api.asked[0][0] == "/api/v1/runs/new/questions"


def test_ask_memory_follows_a_live_recording(monkeypatch):
    runs = [{"id": "live", "status": "running", "processed_ms": 4200, "created_at": "2000-01-02"}]
    api = FakeApi(fake_video("recording", runs))
    monkeypatch.setattr(mcp, "client", lambda: api)
    mcp.call("ask_memory", {"recording": "Bench", "question": "Where is Drill?", "at_end": True})
    assert api.asked == [
        ("/api/v1/runs/live/questions", {"text": "Where is Drill?", "at_ms": 4200})
    ]
    listed = json.loads(mcp.call("list_recordings", {})[0]["text"])
    assert listed[0]["live_status"] == "recording"
    # A running run of an ordinary upload is still not asked.
    api = FakeApi(fake_video(None, runs))
    monkeypatch.setattr(mcp, "client", lambda: api)
    try:
        mcp.call("ask_memory", {"recording": "Bench", "question": "Where is Drill?"})
    except mcp.fb.CliError:
        pass
    else:
        raise AssertionError("an unfinished upload run must not be asked")


def test_a_token_with_a_newline_is_stripped_and_unexpected_errors_stay_redacted(monkeypatch):
    monkeypatch.setenv("FINDBACK_TOKEN", "fbk-secret-456\n")  # gitleaks:allow - synthetic fixture
    assert mcp.client().token == "fbk-secret-456"  # gitleaks:allow - synthetic fixture

    def boom(name, args):
        raise RuntimeError("header rejected: Bearer fbk-secret-456")

    monkeypatch.setattr(mcp, "call", boom)
    reply = mcp.handle(request("tools/call", {"name": "list_recordings", "arguments": {}}))
    assert reply["result"]["isError"] is True
    assert "fbk-secret-456" not in json.dumps(reply)


def test_batches_and_bare_values_get_an_error_instead_of_ending_the_session():
    assert mcp.handle([request("ping")])["error"]["code"] == -32600
    assert mcp.handle("ping")["error"]["code"] == -32600
