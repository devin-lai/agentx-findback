#!/usr/bin/env python3
"""Command-line client for an AgentX FindBack server, used by the findback-video-memory skill.

Standard library only (Python 3.10+). Configuration comes from the environment:

  FINDBACK_URL    server base URL, default http://127.0.0.1:9000
  FINDBACK_TOKEN  optional access token, sent as a Bearer header; never printed

Every command prints one JSON document on stdout and human hints on stderr.
Exit codes: 0 ok, 2 usage, 3 server or HTTP error, 4 timeout.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

VERSION = "1.1.3"
DEFAULT_URL = "http://127.0.0.1:9000"
EXIT_OK, EXIT_USAGE, EXIT_SERVER, EXIT_TIMEOUT = 0, 2, 3, 4
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
BACKENDS = ("reference", "rtdetr", "cosmos", "sam2")
FINISHED = {"complete", "failed", "cancelled"}
ACTIVE = {"queued", "running", "cancelling"}
IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
BEARER = re.compile(r"(?i)(bearer\s+)[^\s\"',;]+")
REDACTED = "[REDACTED]"
HINTS = {
    401: "The server requires an access token. Set FINDBACK_TOKEN in this process's "
    "environment; never paste the token into a chat or onto a command line.",
    404: "Not found. Check the ID with the `videos` or `video` command.",
    409: "The resource is busy. Wait for the current job to finish, then retry.",
    413: "The request exceeds the server's limits.",
    415: "Unsupported media type. Upload an MP4, MOV, MKV, WebM or AVI video.",
    422: "The server rejected the request; the detail says why. Nothing was changed.",
    429: "The server is busy. Retry after the Retry-After interval.",
    502: "A model backend failed. No memory was changed.",
    503: "Unavailable: the requested backend is not configured, or the server needs "
    "AGENTX_API_TOKEN before it accepts non-loopback clients.",
}

_secrets: list[str] = []


class CliError(Exception):
    """A failure reported as a JSON error document with a specific exit code."""

    def __init__(self, message, code=EXIT_SERVER, *, status=None, hint=None, data=None):
        super().__init__(message)
        self.message, self.code, self.status, self.hint, self.data = (
            message,
            code,
            status,
            hint,
            data,
        )


class UsageParser(argparse.ArgumentParser):
    """argparse that also reports usage errors as JSON on stdout."""

    def error(self, message):
        self.print_usage(sys.stderr)
        emit({"error": {"exit_code": EXIT_USAGE, "message": message}})
        sys.stderr.write(f"{self.prog}: error: {message}\n")
        raise SystemExit(EXIT_USAGE)


# --- Output and redaction ------------------------------------------------------------------


def redact(text: str) -> str:
    for secret in _secrets:
        text = text.replace(secret, REDACTED)
    return BEARER.sub(lambda m: m.group(1) + REDACTED, text)


def scrub(value):
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def emit(payload, pretty=False) -> None:
    text = json.dumps(
        scrub(payload),
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )
    sys.stdout.write(redact(text) + "\n")
    sys.stdout.flush()


def note(message: str) -> None:
    sys.stderr.write(redact(message) + "\n")
    sys.stderr.flush()


def clock(ms) -> str | None:
    """Video time as the server writes it: mm:ss.s."""
    if ms is None:
        return None
    seconds = ms / 1000
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


# --- HTTP -----------------------------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow redirects: urllib would forward the Authorization header to the target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise CliError(
            f"Refused to follow an HTTP {code} redirect from {req.full_url}. The access token "
            "is never forwarded; point FINDBACK_URL at the FindBack server itself.",
            status=code,
        )


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class Client:
    def __init__(self, base_url: str, token: str | None, timeout: float):
        parts = urllib.parse.urlsplit(base_url.strip())
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise CliError(
                "FINDBACK_URL must be an http(s) URL such as http://127.0.0.1:9000.", EXIT_USAGE
            )
        if parts.username or parts.password:
            raise CliError(
                "FINDBACK_URL must not contain credentials; set FINDBACK_TOKEN instead.",
                EXIT_USAGE,
            )
        path = parts.path.rstrip("/")
        path = path[: -len("/api")] if path.endswith("/api") else path
        self.base = urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, "", ""))
        self.token = token
        self.timeout = timeout
        handlers: list = [_NoRedirect()]
        if _loopback(parts.hostname):
            handlers.append(urllib.request.ProxyHandler({}))  # loopback never goes via a proxy
        elif token and parts.scheme == "http":
            note(
                "Warning: FINDBACK_TOKEN is being sent over unencrypted HTTP to a non-loopback "
                "host. Prefer HTTPS or an SSH tunnel to 127.0.0.1."
            )
        self.opener = urllib.request.build_opener(*handlers)

    def request(self, method, path, *, query=None, body=None, data=None, headers=None, accept=None):
        url = self.base + path + ("?" + urllib.parse.urlencode(query) if query else "")
        sent = {"Accept": accept or "application/json", "User-Agent": f"findback-skill/{VERSION}"}
        sent.update(headers or {})
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            sent["Content-Type"] = "application/json"
        if self.token:
            sent["Authorization"] = "Bearer " + self.token
        request = urllib.request.Request(url, data=data, method=method, headers=sent)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as exc:
            raise _http_error(exc, method, path) from None
        except TimeoutError:
            raise self._timeout(method, path) from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise self._timeout(method, path) from None
            raise self._unreachable(exc.reason) from None
        except OSError as exc:
            raise self._unreachable(exc) from None

    def _timeout(self, method, path):
        return CliError(
            f"{method} {path} timed out after {self.timeout:g} s.",
            EXIT_TIMEOUT,
            hint="Raise --request-timeout for slow model backends, or check the server's load.",
        )

    def _unreachable(self, reason):
        return CliError(
            f"Cannot reach FindBack at {self.base}: {reason}",
            hint="Check FINDBACK_URL and that the server is running (`agentx serve`). "
            "See references/troubleshooting.md.",
        )

    def json(self, method, path, **kwargs):
        _, _, raw = self.request(method, path, **kwargs)
        try:
            return json.loads(raw)
        except ValueError:
            raise CliError(
                f"{method} {path} did not return JSON; is this a FindBack server?"
            ) from None


def _http_error(exc: urllib.error.HTTPError, method: str, path: str) -> CliError:
    try:
        raw = exc.read(65536)
    except OSError:
        raw = b""
    try:
        payload = json.loads(raw)
        detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
    except ValueError:
        detail = raw.decode("utf-8", "replace").strip()[:500]
    if isinstance(detail, list):  # FastAPI validation errors
        detail = "; ".join(
            ".".join(str(p) for p in item.get("loc", [])[1:]) + f": {item.get('msg')}"
            if isinstance(item, dict)
            else str(item)
            for item in detail
        )
    hint = HINTS.get(exc.code)
    retry = exc.headers.get("Retry-After") if exc.headers else None
    if retry and hint:
        hint += f" Retry-After: {retry} s."
    return CliError(
        f"HTTP {exc.code} from {method} {path}: {detail or exc.reason}",
        status=exc.code,
        hint=hint,
    )


def _segment(value: str, what: str) -> str:
    if not IDENTIFIER.match(value or ""):
        raise CliError(f"Invalid {what}: {value!r}.", EXIT_USAGE)
    return urllib.parse.quote(value, safe="")


# --- Compact views --------------------------------------------------------------------------


def run_view(run: dict, video: dict | None = None) -> dict:
    view = {
        "id": run["id"],
        "video_id": run.get("video_id"),
        "status": run["status"],
        "backend": run.get("backend"),
        "created_at": run.get("created_at"),
        "processed_ms": run.get("processed_ms"),
        "observation_count": run.get("observation_count"),
        "elapsed_seconds": run.get("elapsed_seconds"),
        "object_count": len(run.get("object_ids") or []),
        "error": run.get("error"),
    }
    skill = (run.get("provenance") or {}).get("skill")
    if isinstance(skill, dict) and skill.get("name"):
        view["skill"] = f"{skill['name']}@{str(skill.get('sha256', ''))[:12]}"
    if video is not None:
        view["current"] = _covers(run, video)
    return view


def _covers(run: dict, video: dict) -> bool:
    """A run answers for the video's current inventory only if it indexed exactly those objects
    with the same region names; later registrations are outside an older run's scope."""
    return set(run.get("object_ids") or []) == {o["id"] for o in video.get("objects", [])} and (
        run.get("regions") == video.get("regions")
    )


def video_view(video: dict) -> dict:
    return {
        "id": video["id"],
        "title": video.get("title"),
        "original_name": video.get("original_name"),
        "is_fixture": video.get("is_fixture"),
        "duration_ms": video.get("duration_ms"),
        "duration": clock(video.get("duration_ms")),
        "size": f"{video.get('width')}x{video.get('height')}",
        "created_at": video.get("created_at"),
        "regions": [{"id": r["id"], "name": r["name"]} for r in video.get("regions", [])],
        "objects": [
            {
                "id": o["id"],
                "name": o["name"],
                "label": o.get("label"),
                "registered_at_ms": o.get("registered_at_ms"),
            }
            for o in video.get("objects", [])
        ],
        "runs": [run_view(r, video) for r in video.get("runs", [])],
    }


def _evidence(item: dict, regions: dict) -> dict:
    return {
        "observation_id": item["observation_id"],
        "at_ms": item["at_ms"],
        "time": clock(item["at_ms"]),
        "zone": item.get("zone"),
        "zone_name": regions.get(item.get("zone")),
        "scene_reference": item.get("scene_reference"),
    }


def _tool(tool: dict) -> str:
    result = tool.get("result")
    name = str(tool.get("name", "?"))
    return f"{name}: {result}" if isinstance(result, (str, int, float)) else name


def answer_view(answer: dict, regions: dict) -> dict:
    states = answer.get("states") or []
    primary = states[0] if len(states) == 1 else None
    evidence = answer.get("evidence") or []
    view = {
        "question_id": answer["id"],
        "run_id": answer["run_id"],
        "question": answer["question"],
        "intent": answer["intent"],
        "cutoff_ms": answer["as_of_ms"],
        "cutoff": clock(answer["as_of_ms"]),
        "answer": answer["answer"],
        "status": primary["status"] if primary else None,
        "object": {"id": primary["object_id"], "name": primary["name"]} if primary else None,
        "reason": primary.get("reason") if primary else None,
        "evidence": _evidence(evidence[0], regions) if evidence else None,
        "warnings": answer.get("warnings", []),
        "planner": answer.get("planner"),
        "planner_model": answer.get("planner_model"),
        "skills": [f"{s['name']}@{s['sha256'][:12]}" for s in answer.get("skills", [])],
        "tools": [_tool(t) for t in answer.get("tools", []) if t.get("name") != "load_skill"],
        "elapsed_seconds": answer.get("elapsed_seconds"),
    }
    if len(evidence) > 1:
        view["other_evidence"] = [_evidence(e, regions) for e in evidence[1:]]
    if answer.get("events"):
        view["events"] = [
            {
                "time": clock(e["at_ms"]),
                "kind": e["kind"],
                "zone_name": regions.get(e.get("zone")),
                "observation_id": e.get("observation_id"),
            }
            for e in answer["events"]
        ]
    if len(states) > 1:
        view["states"] = [
            {"object_id": s["object_id"], "name": s["name"], "status": s["status"]} for s in states
        ]
    return view


# --- Files ----------------------------------------------------------------------------------


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(content)
    partial.replace(path)
    return path.resolve()


def save_frame(client: Client, observation_id: str, out: Path) -> dict:
    _, headers, content = client.request(
        "GET",
        f"/api/v1/observations/{_segment(observation_id, 'observation ID')}/frame",
        accept="image/jpeg",
    )
    if not content.startswith(b"\xff\xd8"):
        raise CliError("The server did not return a JPEG evidence frame.")
    target = (
        out
        if out.suffix.lower() in {".jpg", ".jpeg"}
        else out / (f"findback-evidence-{observation_id}.jpg")
    )
    path = _write(target, content)
    frame_ms = headers.get("X-Frame-Time-Ms")
    frame_ms = int(frame_ms) if frame_ms and frame_ms.isdigit() else None
    return {
        "observation_id": observation_id,
        "frame_time_ms": frame_ms,
        "frame_time": clock(frame_ms),
        "path": str(path),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "media": f"MEDIA:{path}",
    }


class _Multipart:
    """A streamed multipart/form-data body with one file field, so large videos stay on disk."""

    def __init__(self, path: Path):
        self.boundary = "findback-" + uuid.uuid4().hex
        name = path.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
        self.head = (
            f"--{self.boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        self.tail = f"\r\n--{self.boundary}--\r\n".encode()
        self.path = path
        self.length = len(self.head) + path.stat().st_size + len(self.tail)

    def __iter__(self):
        yield self.head
        with self.path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                yield chunk
        yield self.tail


# --- Commands -------------------------------------------------------------------------------


def cmd_health(client: Client, args) -> dict:
    health = client.json("GET", "/api/health")
    auth = client.json("GET", "/api/auth/status")
    summary = {
        "url": client.base,
        "health": health,
        "auth_required": bool(auth.get("required")),
        "token_configured": bool(client.token),
    }
    try:
        capabilities = client.json("GET", "/api/v1/capabilities")
    except CliError as exc:
        exc.data = summary
        raise
    summary.update(
        {
            "index_backends": {
                "reference": bool(capabilities.get("reference", True)),
                "rtdetr": bool(capabilities.get("rtdetr_installed")),
                "cosmos": bool(capabilities.get("cosmos")),
                "sam2": bool(capabilities.get("sam2")),
            },
            "suggest_backends": capabilities.get("discovery", {}),
            "planner": capabilities.get("planner"),
            "planner_model": capabilities.get("planner_model"),
            "cosmos_model": capabilities.get("cosmos_model"),
            "capabilities": capabilities,
        }
    )
    return summary


def cmd_videos(client: Client, args) -> list | dict:
    videos = client.json("GET", "/api/v1/videos")
    return videos if args.full else [video_view(v) for v in videos]


def cmd_video(client: Client, args) -> dict:
    video = client.json("GET", f"/api/v1/videos/{_segment(args.video_id, 'video ID')}")
    return video if args.full else video_view(video)


def cmd_upload(client: Client, args) -> dict:
    path = Path(args.file).expanduser()
    if not path.is_file():
        raise CliError(f"No such file: {path}", EXIT_USAGE)
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        raise CliError("Upload an MP4, MOV, MKV, WebM or AVI video.", EXIT_USAGE)
    body = _Multipart(path)
    video = client.json(
        "POST",
        "/api/v1/videos",
        data=iter(body),
        headers={
            "Content-Type": f"multipart/form-data; boundary={body.boundary}",
            "Content-Length": str(body.length),
        },
    )
    note("Uploaded. Register the objects the user names, then build memory with `index`.")
    return video_view(video)


def cmd_demo(client: Client, args) -> dict:
    video = client.json("POST", "/api/v1/demo", query={"scene": args.scene})
    note(
        "Created a generated, labeled software fixture with two registered objects. It checks "
        "the workflow; it is not real footage or an accuracy benchmark."
    )
    return video_view(video)


def cmd_suggest(client: Client, args) -> dict:
    video_id = _segment(args.video_id, "video ID")
    result = client.json(
        "POST",
        f"/api/v1/videos/{video_id}/suggestions",
        body={"at_ms": args.at_ms, "backend": args.backend},
    )
    note(
        "Suggestions are advisory and nothing was registered. Ask the user which objects to "
        "register and what to call them."
    )
    return {"video_id": args.video_id, **result}


def _box(text: str) -> list[float]:
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError:
        values = []
    if len(values) != 4:
        raise CliError("--box needs four comma-separated numbers: x1,y1,x2,y2.", EXIT_USAGE)
    return values


def cmd_register(client: Client, args) -> dict:
    video_id = _segment(args.video_id, "video ID")
    x1, y1, x2, y2 = _box(args.box)
    if args.pixels:
        video = client.json("GET", f"/api/v1/videos/{video_id}")
        width, height = video["width"], video["height"]
        x1, x2, y1, y2 = x1 / width, x2 / width, y1 / height, y2 / height
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise CliError(
            "The box must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 in normalized frame "
            "coordinates; pass --pixels for pixel coordinates.",
            EXIT_USAGE,
        )
    obj = client.json(
        "POST",
        f"/api/v1/videos/{video_id}/objects",
        body={
            "name": args.name,
            "label": args.label,
            "at_ms": args.at_ms,
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        },
    )
    note("Registered. Existing runs do not include this object; build memory again with `index`.")
    return {k: obj.get(k) for k in ("id", "name", "label", "registered_at_ms", "box")}


def wait_for_run(client: Client, run_id: str, timeout: float, poll: float) -> dict:
    deadline = time.monotonic() + timeout
    status = None
    while True:
        run = client.json("GET", f"/api/v1/runs/{_segment(run_id, 'run ID')}")
        if run["status"] != status:
            status = run["status"]
            note(f"run {run_id}: {status}")
        if status in FINISHED:
            return run
        if time.monotonic() >= deadline:
            raise CliError(
                f"Run {run_id} was still {status} after {timeout:g} s.",
                EXIT_TIMEOUT,
                hint=f"The run continues on the server. Resume with: run {run_id} --wait",
                data={"run": run_view(run)},
            )
        time.sleep(min(poll, max(0.05, deadline - time.monotonic())))


def _finished(run: dict) -> int:
    if run["status"] in {"failed", "cancelled"}:
        note("The run did not complete; see its error. See references/troubleshooting.md.")
        return EXIT_SERVER
    return EXIT_OK


def cmd_index(client: Client, args):
    video_id = _segment(args.video_id, "video ID")
    if args.reuse:
        video = client.json("GET", f"/api/v1/videos/{video_id}")
        same = [
            r for r in video.get("runs", []) if r["backend"] == args.backend and _covers(r, video)
        ]
        run = next((r for r in same if r["status"] == "complete"), None) or next(
            (r for r in same if r["status"] in ACTIVE), None
        )
        if run is not None:
            note(f"Reusing run {run['id']} ({run['status']}); no new indexing was started.")
            if args.wait and run["status"] not in FINISHED:
                run = wait_for_run(client, run["id"], args.wait_timeout, args.poll)
            return {"reused": True, "run": run_view(run)}, _finished(run)
    body = {"backend": args.backend}
    if args.sample_fps is not None:
        body["sample_fps"] = args.sample_fps
    run = client.json("POST", f"/api/v1/videos/{video_id}/runs", body=body)
    if args.wait:
        run = wait_for_run(client, run["id"], args.wait_timeout, args.poll)
    return {"reused": False, "run": run_view(run)}, _finished(run)


def cmd_run(client: Client, args):
    run = client.json("GET", f"/api/v1/runs/{_segment(args.run_id, 'run ID')}")
    if args.wait and run["status"] not in FINISHED:
        run = wait_for_run(client, args.run_id, args.wait_timeout, args.poll)
    return run_view(run), _finished(run)


def _title_key(title: str) -> str:
    return " ".join(title.split()).casefold()


def _unwrap_title(title: str) -> str:
    """Tolerate balanced quote characters passed literally by an agent's tool call."""
    value = title.strip()
    pairs = {'"': '"', "'": "'", "“": "”", "‘": "’"}
    while len(value) >= 2 and value[0] in pairs and value[-1] == pairs[value[0]]:
        value = value[1:-1].strip()
    return value


def resolve_recording(client: Client, title: str) -> tuple[str, dict]:
    """The latest complete, current run of the recording with exactly this title.

    A title that merely starts the same way ("Controlled desk fixture" versus "Controlled desk
    fixture, bumped camera") is a different recording, so only an exact match (ignoring case
    and spacing) is accepted; otherwise the error lists the titles to choose from.
    """
    videos = client.json("GET", "/api/v1/videos")
    matches = [v for v in videos if _title_key(v.get("title") or "") == _title_key(title)]
    if not matches:
        unwrapped = _unwrap_title(title)
        if unwrapped != title:
            matches = [
                v for v in videos if _title_key(v.get("title") or "") == _title_key(unwrapped)
            ]
    if not matches:
        titles = "; ".join(f'"{v.get("title")}"' for v in videos) or "none"
        raise CliError(
            f'No recording is titled exactly "{title}". Recordings: {titles}. Use one of these '
            "titles, or ask the user which recording they mean.",
            EXIT_USAGE,
        )
    if len(matches) > 1:
        ids = ", ".join(v["id"] for v in matches)
        raise CliError(
            f'{len(matches)} recordings are titled "{title}" ({ids}); ask the user which one, then '
            "use its run ID.",
            EXIT_USAGE,
        )
    video = matches[0]
    runs = [r for r in video.get("runs", []) if r["status"] == "complete" and _covers(r, video)]
    if not runs:
        raise CliError(
            f'"{video.get("title")}" has no complete memory for its current objects. Build it with '
            f"`index {video['id']} --reuse --wait` if the user wants that.",
            EXIT_USAGE,
        )
    run = max(runs, key=lambda r: r.get("created_at") or "")
    return run["id"], video


def cmd_ask(client: Client, args):
    if args.video and args.run_id:
        raise CliError("Give either a run ID or --video, not both.", EXIT_USAGE)
    video = None
    if args.video:
        run_id, video = resolve_recording(client, args.video)
    elif args.run_id:
        run_id = _segment(args.run_id, "run ID")
    else:
        raise CliError("Give the run ID, or the recording's exact title with --video.", EXIT_USAGE)
    text = args.question.strip()
    if not text or len(text) > 1000:
        raise CliError("The question must be 1-1000 characters.", EXIT_USAGE)
    body = {"text": text, "use_provider": not args.no_planner, "use_skills": not args.no_skills}
    if args.at_ms is not None:
        body["at_ms"] = args.at_ms
    if args.object_id:
        body["object_id"] = args.object_id
    if args.intent:
        body["intent"] = args.intent
    answer = client.json("POST", f"/api/v1/runs/{run_id}/questions", body=body)
    if args.full:
        return answer, EXIT_OK
    try:
        run = client.json("GET", f"/api/v1/runs/{run_id}")
        regions = {r["id"]: r["name"] for r in run.get("regions", [])}
        if video is None and run.get("video_id"):
            video = client.json("GET", f"/api/v1/videos/{_segment(run['video_id'], 'video ID')}")
    except CliError:
        regions = {}
    view = {"recording": (video or {}).get("title"), **answer_view(answer, regions)}
    for warning in view["warnings"]:
        note(f"Server warning: {warning}")
    code = EXIT_OK
    if args.save_frame:
        if view["evidence"] is None:
            view["evidence_frame"] = None
            note("No evidence frame: this answer cites no visible observation.")
        else:
            try:
                frame = save_frame(client, view["evidence"]["observation_id"], args.save_frame)
                view["evidence_frame"], view["media"] = frame["path"], frame["media"]
                note(f"Evidence frame saved: {frame['path']}")
            except CliError as exc:
                view["evidence_frame"], view["frame_error"] = None, exc.message
                code = exc.code
    return view, code


def cmd_frame(client: Client, args) -> dict:
    frame = save_frame(client, args.observation_id, args.out)
    note(f"Evidence frame saved: {frame['path']}")
    return frame


def cmd_report(client: Client, args) -> dict:
    question_id = _segment(args.question_id, "question ID")
    out = args.out.expanduser()
    if out.exists() and not args.force:
        raise CliError(f"{out} exists; pass --force to replace it.", EXIT_USAGE)
    _, _, content = client.request(
        "GET", f"/api/v1/questions/{question_id}/bundle", accept="application/zip"
    )
    if not content.startswith(b"PK\x03\x04"):
        raise CliError("The server did not return a ZIP evidence report.")
    path = _write(out, content)
    note("Extract the ZIP and open report.html; it carries its own hash verifier.")
    return {
        "question_id": args.question_id,
        "path": str(path),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


# --- Arguments ------------------------------------------------------------------------------


def _non_negative(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("expected an integer number of milliseconds") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def _fps(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a number") from None
    if not 1 <= value <= 15:
        raise argparse.ArgumentTypeError("must be between 1 and 15")
    return value


def _positive(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a number of seconds") from None
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def _common(parser: argparse.ArgumentParser, suppress: bool) -> None:
    """Options accepted before or after the command; a later copy never resets an earlier one."""
    default = argparse.SUPPRESS if suppress else None
    parser.add_argument("--url", default=default, help="Server URL; overrides FINDBACK_URL.")
    parser.add_argument(
        "--request-timeout",
        type=_positive,
        default=argparse.SUPPRESS if suppress else 60.0,
        help="Seconds to wait for one HTTP response (default 60).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=argparse.SUPPRESS if suppress else False,
        help="Indent the JSON output.",
    )


def _waiting(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wait", action="store_true", help="Poll until the run finishes.")
    parser.add_argument(
        "--timeout",
        dest="wait_timeout",
        type=_positive,
        default=600.0,
        help="Seconds to wait for the run (default 600); exit 4 when exceeded.",
    )
    parser.add_argument("--poll", type=_positive, default=1.0, help="Seconds between polls.")


def build_parser() -> argparse.ArgumentParser:
    parser = UsageParser(
        prog="findback.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _common(parser, suppress=False)
    parser.add_argument("--version", action="version", version=f"findback.py {VERSION}")
    commands = parser.add_subparsers(dest="command", required=True, parser_class=UsageParser)

    def add(name, handler, help_text):
        command = commands.add_parser(name, help=help_text, description=help_text)
        _common(command, suppress=True)
        command.set_defaults(handler=handler)
        return command

    add("health", cmd_health, "Server health, authentication and capabilities.")
    command = add("videos", cmd_videos, "List recordings with their objects and runs.")
    command.add_argument("--full", action="store_true", help="Print the server's full response.")
    command = add("video", cmd_video, "Show one recording with its objects and runs.")
    command.add_argument("video_id")
    command.add_argument("--full", action="store_true", help="Print the server's full response.")
    command = add("upload", cmd_upload, "Upload a video file the user named.")
    command.add_argument("file")
    command = add("demo", cmd_demo, "Create the generated controlled sample (two objects).")
    command.add_argument("--scene", choices=("fixed", "bumped"), default="fixed")
    command = add("suggest", cmd_suggest, "Advisory registration candidates for one frame.")
    command.add_argument("video_id")
    command.add_argument("--at-ms", type=_non_negative, required=True, help="Frame time.")
    command.add_argument("--backend", choices=("rtdetr", "cosmos"), default="rtdetr")
    command = add("register", cmd_register, "Register an object the user named and located.")
    command.add_argument("video_id")
    command.add_argument("--name", required=True, help="Unique name within the recording.")
    command.add_argument("--label", default="custom", help="Category, e.g. remote or cup.")
    command.add_argument("--at-ms", type=_non_negative, required=True, help="Frame time.")
    command.add_argument("--box", required=True, help="x1,y1,x2,y2 normalized to 0-1.")
    command.add_argument("--pixels", action="store_true", help="The box is in pixels.")
    command = add("index", cmd_index, "Build memory for a recording; each build is a new run.")
    command.add_argument("video_id")
    command.add_argument("--backend", choices=BACKENDS, default="reference")
    command.add_argument("--sample-fps", type=_fps, help="Sampling rate, 1-15 (server default 5).")
    command.add_argument(
        "--reuse", action="store_true", help="Reuse a run that already covers these objects."
    )
    _waiting(command)
    command = add("run", cmd_run, "Show one run, optionally waiting for it to finish.")
    command.add_argument("run_id")
    _waiting(command)
    command = add("ask", cmd_ask, "Ask about a registered object as of a cutoff.")
    command.add_argument("run_id", nargs="?", help="A run ID; omit it when using --video.")
    command.add_argument("question")
    command.add_argument(
        "--video",
        metavar="TITLE",
        help="The recording's exact title; asks its latest complete memory.",
    )
    cutoff = command.add_mutually_exclusive_group(required=True)
    cutoff.add_argument("--at-ms", type=_non_negative, help="Cutoff in video milliseconds.")
    cutoff.add_argument("--at-end", action="store_true", help="Cutoff at the end of the run.")
    command.add_argument("--object-id", help="Select a registered object explicitly.")
    command.add_argument("--intent", choices=("location", "last_seen", "history"))
    command.add_argument("--no-planner", action="store_true", help="Force the local resolver.")
    command.add_argument("--no-skills", action="store_true", help="Withhold the server Skills.")
    command.add_argument("--save-frame", type=Path, metavar="DIR", help="Save the evidence frame.")
    command.add_argument("--full", action="store_true", help="Print the server's full response.")
    command = add("frame", cmd_frame, "Save the original frame of one observation.")
    command.add_argument("observation_id")
    command.add_argument("--out", type=Path, required=True, help="Directory or .jpg path.")
    command = add("report", cmd_report, "Download a portable evidence report (ZIP).")
    command.add_argument("question_id")
    command.add_argument("--out", type=Path, required=True, help="Destination .zip file.")
    command.add_argument("--force", action="store_true", help="Replace an existing file.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    token = os.environ.get("FINDBACK_TOKEN", "").strip()
    _secrets[:] = [token] if token else []
    try:
        base = args.url or os.environ.get("FINDBACK_URL") or DEFAULT_URL
        client = Client(base, token or None, args.request_timeout)
        result = args.handler(client, args)
        payload, code = result if isinstance(result, tuple) else (result, EXIT_OK)
    except CliError as exc:
        error = {"exit_code": exc.code, "message": exc.message}
        if exc.status is not None:
            error["http_status"] = exc.status
        emit({"error": error, **({"partial": exc.data} if exc.data else {})}, args.pretty)
        if exc.hint:
            note(exc.hint)
        return exc.code
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # a traceback must not carry request details into a transcript
        if os.environ.get("FINDBACK_DEBUG"):
            raise
        emit({"error": {"exit_code": EXIT_SERVER, "message": f"{type(exc).__name__}: {exc}"}})
        return EXIT_SERVER
    emit(payload, args.pretty)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
