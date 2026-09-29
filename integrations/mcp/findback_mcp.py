#!/usr/bin/env python3
"""FindBack as a Model Context Protocol (MCP) server, standard library only.

MCP connects an agent to tools; the `findback-video-memory` Skill teaches the agent how to use
them. This server speaks MCP's JSON-RPC over stdio and reuses the Skill's own HTTP client
(`skills/findback-video-memory/scripts/findback.py`), so token redaction, the no-redirect policy
and the loopback proxy bypass are identical in both.

Tools:

* `list_recordings` — recordings, registered objects and which memory runs are current;
* `ask_memory` — ask a question about one recording at a cutoff (or at its end); the reply is the
  server's answer rendered from verified memory, plus the cited evidence frame as an image;
* `get_evidence_frame` — the original frame of one observation, as an image.

Configuration comes from the environment only: `FINDBACK_URL` (default http://127.0.0.1:9000) and
optional `FINDBACK_TOKEN`, which is never returned to the client.

Example client configuration (Claude Desktop, Cursor and other MCP clients use the same shape):

    {"mcpServers": {"findback": {"command": "python3",
      "args": ["/path/to/AgentX/integrations/mcp/findback_mcp.py"],
      "env": {"FINDBACK_URL": "http://127.0.0.1:9000"}}}}
"""

import base64
import importlib.util
import json
import os
import sys
from pathlib import Path

SKILL_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "findback-video-memory"
    / "scripts"
    / "findback.py"
)
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "findback", "version": "1.0.0"}
NOTICE = (
    "Answers describe the recording at the given time, never where anything is now. "
    "'last_seen' means the position at the cutoff is not confirmed."
)


def load_client_module():
    # Never write bytecode into the signed Skill directory: an extra file breaks its signature.
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("findback_skill_client", SKILL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fb = load_client_module()

TOOLS = [
    {
        "name": "list_recordings",
        "description": "List FindBack recordings with their registered objects and memory runs "
        "(a run marked current covers every registered object).",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "ask_memory",
        "description": "Ask where a registered object was, or what happened to it, in one "
        "recording at a cutoff time. Returns the answer from verified memory, its status "
        "(visible, last_seen, unknown, not_observed), the evidence time and area, and the "
        "evidence frame. A recording cannot show where anything is now; for a live camera "
        "recording, at_end asks about the newest moment memory has processed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "recording": {"type": "string", "description": "Recording title or id"},
                "question": {"type": "string", "description": "The user's question"},
                "at_ms": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Cutoff in milliseconds of video time",
                },
                "at_end": {
                    "type": "boolean",
                    "description": "Ask at the end of the recording instead of at_ms",
                },
            },
            "required": ["recording", "question"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_evidence_frame",
        "description": "Return the original video frame of one observation as an image.",
        "inputSchema": {
            "type": "object",
            "properties": {"observation_id": {"type": "string"}},
            "required": ["observation_id"],
            "additionalProperties": False,
        },
    },
]


def client():
    # A token read from a file often ends in a newline, which no HTTP header may contain.
    token = os.environ.get("FINDBACK_TOKEN", "").strip() or None
    fb._secrets[:] = [token] if token else []  # every error text is redacted with it
    return fb.Client(os.environ.get("FINDBACK_URL", "http://127.0.0.1:9000"), token, timeout=180)


def frame_content(api, observation_id: str) -> dict:
    _, _, content = api.request(
        "GET",
        f"/api/v1/observations/{fb._segment(observation_id, 'observation ID')}/frame",
        accept="image/jpeg",
    )
    return {"type": "image", "data": base64.b64encode(content).decode(), "mimeType": "image/jpeg"}


def find_recording(api, key: str) -> dict:
    videos = api.json("GET", "/api/v1/videos")
    matches = [v for v in videos if key in (v["id"], v.get("title"), v.get("original_name"))]
    if len(matches) != 1:
        titles = ", ".join(repr(v.get("title")) for v in videos) or "none"
        raise fb.CliError(f"No single recording named {key!r}. Recordings: {titles}.")
    return matches[0]


def call(name: str, arguments: dict) -> list[dict]:
    api = client()
    if name == "list_recordings":
        videos = api.json("GET", "/api/v1/videos")
        views = [
            {
                **fb.video_view(v),
                **({"live_status": v["live_status"]} if v.get("live_status") else {}),
            }
            for v in videos
        ]
        return [{"type": "text", "text": json.dumps(views, indent=1)}]
    if name == "ask_memory":
        video = find_recording(api, str(arguments["recording"]))
        # A capture that is still recording is asked through the run following it; its answers
        # carry the server's live-recording warning.
        live = video.get("live_status") == "recording"
        runs = [
            r
            for r in video.get("runs", [])
            if (r["status"] == "complete" or (live and r["status"] == "running"))
            and fb._covers(r, video)
        ]
        if not runs:
            raise fb.CliError("That recording has no complete memory run covering its objects.")
        # The server lists runs newest first; choose by creation time, as the Skill CLI does.
        run = max(runs, key=lambda r: r.get("created_at") or "")
        body = {"text": str(arguments["question"])[:1000]}
        if arguments.get("at_end"):
            body["at_ms"] = run.get("processed_ms") or video["duration_ms"]
        elif arguments.get("at_ms") is not None:
            body["at_ms"] = int(arguments["at_ms"])
        answer = api.json("POST", f"/api/v1/runs/{run['id']}/questions", body=body)
        regions = {r["id"]: r["name"] for r in video.get("regions", [])}
        view = fb.answer_view(answer, regions)
        view["notice"] = NOTICE
        content = [{"type": "text", "text": json.dumps(view, indent=1)}]
        if view["evidence"]:
            content.append(frame_content(api, view["evidence"]["observation_id"]))
        return content
    return [frame_content(api, str(arguments["observation_id"]))]


def handle(message) -> dict | None:
    if not isinstance(message, dict):
        # Batches and bare values are not supported; answer instead of crashing.
        return error(None, -32600, "Invalid request: send one JSON-RPC object per line.")
    method, ident = message.get("method"), message.get("id")
    if ident is None:
        return None  # notifications (for example notifications/initialized) need no reply
    if method == "initialize":
        requested = (message.get("params") or {}).get("protocolVersion")
        version = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        result = {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": NOTICE,
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = message.get("params") or {}
        name, arguments = params.get("name", ""), params.get("arguments") or {}
        tool = next((t for t in TOOLS if t["name"] == name), None)
        if tool is None:
            return error(ident, -32602, f"Unknown tool: {name!r}")
        missing = [k for k in tool["inputSchema"].get("required", []) if k not in arguments]
        if missing:
            return error(ident, -32602, f"Missing arguments: {', '.join(missing)}")
        try:
            result = {"content": call(name, arguments)}
        except fb.CliError as exc:
            text = fb.redact(exc.message + (f" {exc.hint}" if getattr(exc, "hint", None) else ""))
            result = {"content": [{"type": "text", "text": text}], "isError": True}
        except Exception as exc:
            # Never let one bad call end the session, and never echo the token in its message.
            text = fb.redact(f"The FindBack call failed: {type(exc).__name__}: {exc}")
            result = {"content": [{"type": "text", "text": text[:500]}], "isError": True}
    else:
        return error(ident, -32601, f"Method not found: {method}")
    return {"jsonrpc": "2.0", "id": ident, "result": result}


def error(ident, code: int, text: str) -> dict:
    return {"jsonrpc": "2.0", "id": ident, "error": {"code": code, "message": text}}


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            reply = error(None, -32700, "Parse error")
        else:
            reply = handle(message)
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
