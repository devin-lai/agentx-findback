import json

import typer
import uvicorn

from agentx.config import Settings
from agentx.storage.database import Database

app = typer.Typer(no_args_is_help=True, help="AgentX FindBack: evidence-backed visual memory.")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 9000):
    """Run the API, one durable index worker and the built web client."""
    settings = Settings()
    if host not in {"127.0.0.1", "localhost", "::1"} and not settings.api_token:
        raise typer.BadParameter("Set AGENTX_API_TOKEN before binding to a network interface.")
    # Idle keep-alive clients must not block a graceful stop of the single node process.
    uvicorn.run(
        "agentx.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        workers=1,
        timeout_graceful_shutdown=10,
    )


@app.command()
def migrate():
    """Apply committed database migrations."""
    settings = Settings()
    settings.prepare()
    db = Database(settings.db_url)
    db.migrate()
    db.engine.dispose()
    typer.echo("Database migrations applied.")


@app.command()
def doctor(
    probe: bool = typer.Option(
        False,
        "--probe",
        help="Probe the planner request shapes and the configured Cosmos model service.",
    ),
):
    """Report local prerequisites without printing credentials."""
    import importlib.util

    from agentx.vision.video import ffmpeg_executable

    settings = Settings()
    typer.echo(
        json.dumps(
            {
                "ffmpeg": bool(ffmpeg_executable()),
                "planner_configured": settings.planner_available,
                "planner_model": settings.planner_endpoint[2] or None,
                "cosmos_configured": settings.cosmos_available,
                "cosmos_backend": settings.cosmos_backend,
                "cosmos_model": settings.cosmos_model if settings.cosmos_available else None,
                "sam2_checkpoint_ready": settings.sam2_available,
                "identity_enabled": settings.identity_enabled,
                "access_token_configured": bool(settings.api_token),
                "agent_token_configured": bool(settings.api_token and settings.agent_token),
                "neural_vision_installed": all(
                    importlib.util.find_spec(n) is not None
                    for n in ("torch", "transformers", "trackers")
                ),
                "web_built": (settings.web_dist / "index.html").is_file(),
                **({"planner_probe": probe_planner(settings)} if probe else {}),
                **({"cosmos_probe": probe_cosmos(settings)} if probe else {}),
            },
            indent=2,
        )
    )


def probe_planner(settings: Settings) -> dict:
    """One real planner request with the application's own settings.

    Configuration alone does not prove an endpoint accepts what this client sends: an
    inference server can be reachable and still reject the selection request because it was
    started without a reasoning parser. Both shapes are probed, because the tool loop and
    typed object selection do not send the same fields.
    """
    from agentx.agents.providers import PlannerUnavailable, Providers

    if not settings.planner_available:
        return {"configured": False}
    providers = Providers(settings)
    messages = [
        {"role": "system", "content": 'Reply with the JSON object {"ok":true} and nothing else.'},
        {"role": "user", "content": "ready?"},
    ]
    results: dict = {"configured": True, "model": settings.planner_endpoint[2]}
    for name, schema in (
        ("tool_loop", None),
        (
            "object_selection",
            {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        ),
    ):
        try:
            providers.planner_chat(messages, response_schema=schema)
            results[name] = "ok"
        except PlannerUnavailable as exc:
            results[name] = f"failed: {exc}"
    return results


def probe_cosmos(settings: Settings) -> dict:
    """Check that the configured Cosmos name is served, without loading weights or logging secrets.

    The HTTP model listing establishes reachability and the served name. It does not
    establish that visual generation is accurate or that a review request will succeed.
    """
    if not settings.cosmos_available:
        return {"configured": False}
    if settings.cosmos_backend == "transformers":
        return {"configured": True, "backend": "transformers", "checkpoint": "present"}

    from agentx.agents.providers import Providers

    providers = Providers(settings)
    result = {
        "configured": True,
        "backend": "http",
        "base_model": "ok" if providers.served_model() else "unreachable_or_name_mismatch",
    }
    if settings.cosmos_reference_adapter_model:
        result["reference_adapter"] = (
            "ok"
            if providers.served_model(settings.cosmos_reference_adapter_model)
            else "unreachable_or_name_mismatch"
        )
    return result


@app.command("live-push")
def live_push(
    source: str = typer.Argument(
        help="Camera index (0), RTSP/HTTP stream URL or a video file played at its own pace."
    ),
    server: str = typer.Option("http://127.0.0.1:9000", help="FindBack server base URL."),
    title: str = typer.Option("Live camera", help="Name of the new recording."),
    fps: float = typer.Option(5.0, min=0.5, max=30, help="Capture rate sent to the server."),
    seconds: float = typer.Option(0, min=0, help="Stop after this long; 0 runs to the end."),
    max_edge: int = typer.Option(1280, min=320, max=4096, help="Longest side of sent frames."),
    stop: bool = typer.Option(True, help="Seal the recording when the source ends."),
):
    """Feed a camera, stream or file into a live FindBack recording, frame by frame.

    Uses the operator token from AGENTX_API_TOKEN when the server requires one. A file is
    replayed in real time, so memory follows it exactly as it would follow a camera.
    """
    import os
    import time
    import urllib.error
    import urllib.request

    import cv2

    token = os.environ.get("AGENTX_API_TOKEN", "")

    def call(path: str, data: bytes, content_type: str) -> dict:
        request = urllib.request.Request(server.rstrip("/") + path, data=data, method="POST")
        request.add_header("Content-Type", content_type)
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read() or b"{}")

    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise typer.BadParameter(f"Could not open {source}.")
    is_file = not source.isdigit() and "://" not in source
    video = call(
        "/api/v1/live",
        json.dumps({"title": title, "fps": fps, "source": "push"}).encode(),
        "application/json",
    )
    typer.echo(json.dumps({"video_id": video["id"], "title": video["title"]}))
    started = time.monotonic()
    next_send = 0.0
    sent = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            elapsed = time.monotonic() - started
            if is_file:
                # Pace a file by its own timestamps, as a camera would deliver it.
                position = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if position > elapsed:
                    time.sleep(position - elapsed)
                elapsed = max(elapsed, position)
            if seconds and elapsed > seconds:
                break
            if elapsed < next_send:
                continue
            # Keep a fixed schedule; after a stall, restart it instead of bursting to catch up.
            next_send = next_send + 1 / fps if elapsed - next_send < 1 / fps else elapsed + 1 / fps
            scale = min(1.0, max_edge / max(image.shape[:2]))
            if scale < 1:
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            encoded_ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not encoded_ok:
                continue
            result = call(f"/api/v1/live/{video['id']}/frames", encoded.tobytes(), "image/jpeg")
            sent += 1
            if sent % max(1, round(fps * 10)) == 0:
                typer.echo(json.dumps({"sent": sent, **result}))
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if stop:
            try:
                sealed = call(f"/api/v1/live/{video['id']}/stop", b"", "application/json")
            except urllib.error.HTTPError as exc:
                # 409: the server already sealed it (idle timeout or maximum duration).
                typer.echo(json.dumps({"stop": exc.code, "frames_sent": sent}))
                return
            typer.echo(
                json.dumps(
                    {
                        "sealed": sealed["id"],
                        "sha256": sealed["sha256"],
                        "duration_ms": sealed["duration_ms"],
                        "frames_sent": sent,
                    }
                )
            )
