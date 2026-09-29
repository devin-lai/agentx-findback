"""The temporary OpenClaw proxy must preserve the read-and-ask boundary."""

from __future__ import annotations

import importlib.util
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "integrations/openclaw/read_only_proxy.py"
SPEC = importlib.util.spec_from_file_location("openclaw_proxy", MODULE_PATH)
assert SPEC and SPEC.loader
proxy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proxy)


def start_server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_proxy_allows_reads_and_existing_run_questions_only():
    requests = []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, format_string, *args):
            return

        def do_GET(self):
            requests.append(("GET", self.path, self.headers.get("Authorization")))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            requests.append(("POST", self.path, self.headers.get("Authorization")))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

    upstream = start_server(Upstream)
    gateway = start_server(
        proxy.make_handler("test-read-token", f"http://127.0.0.1:{upstream.server_port}")
    )
    base = f"http://127.0.0.1:{gateway.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/v1/videos", timeout=3) as response:
            assert response.status == 200
            assert b"test-read-token" not in response.read()
        path = "/api/v1/runs/" + "a" * 32 + "/questions"
        question = urllib.request.Request(
            base + path,
            data=b'{"question":"where?"}',
            method="POST",
        )
        with urllib.request.urlopen(question, timeout=3) as response:
            assert response.status == 200
        for rejected in ("/api/v1/videos", "/api/v1/runs/not-an-id/questions"):
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(
                    urllib.request.Request(base + rejected, data=b"{}", method="POST"),
                    timeout=3,
                )
            assert error.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base + "/not-api", timeout=3)
        assert error.value.code == 403
    finally:
        gateway.shutdown()
        upstream.shutdown()
        gateway.server_close()
        upstream.server_close()
    assert requests == [
        ("GET", "/api/v1/videos", "Bearer test-read-token"),
        ("POST", path, "Bearer test-read-token"),
    ]


def test_proxy_rejects_non_loopback_upstream():
    with pytest.raises(ValueError, match="loopback"):
        proxy.make_handler("test-read-token", "https://example.com")


def test_proxy_does_not_forward_token_through_upstream_redirect():
    redirected_requests = []

    class RedirectTarget(BaseHTTPRequestHandler):
        def log_message(self, format_string, *args):
            return

        def do_GET(self):
            redirected_requests.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()

    target = start_server(RedirectTarget)

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, format_string, *args):
            return

        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{target.server_port}/capture")
            self.end_headers()

    upstream = start_server(Upstream)
    gateway = start_server(
        proxy.make_handler("test-read-token", f"http://127.0.0.1:{upstream.server_port}")
    )
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(
                f"http://127.0.0.1:{gateway.server_port}/api/v1/redirect", timeout=3
            )
        assert error.value.code == 302
        assert redirected_requests == []
    finally:
        gateway.shutdown()
        upstream.shutdown()
        target.shutdown()
        gateway.server_close()
        upstream.server_close()
        target.server_close()
