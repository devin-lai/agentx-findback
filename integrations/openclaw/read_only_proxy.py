"""Short-lived FindBack read-and-ask proxy for a sandboxed OpenClaw tool container.

The token stays in this host process. Bind to a Docker bridge address only on a
trusted single-user host: other local containers can reach this experimental proxy.
"""

from __future__ import annotations

import argparse
import re
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import dotenv_values

QUESTION_PATH = re.compile(r"/api/v1/runs/[0-9a-f]{32}/questions\Z")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def make_handler(token: str, upstream: str) -> type[BaseHTTPRequestHandler]:
    """Build a handler with its token captured only in the host-side closure."""
    if not token:
        raise ValueError("a read-and-ask token is required")
    parsed_upstream = urllib.parse.urlsplit(upstream)
    if (
        parsed_upstream.scheme not in {"http", "https"}
        or parsed_upstream.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed_upstream.username
        or parsed_upstream.password
        or parsed_upstream.path not in {"", "/"}
        or parsed_upstream.query
        or parsed_upstream.fragment
    ):
        raise ValueError("upstream must be a loopback HTTP origin")
    upstream = upstream.rstrip("/")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format_string, *args):
            # Do not persist request paths, questions or response content.
            return

        def do_GET(self):
            self.forward("GET")

        def do_HEAD(self):
            self.forward("HEAD")

        def do_POST(self):
            self.forward("POST")

        def forward(self, method: str) -> None:
            parsed = urllib.parse.urlsplit(self.path)
            if parsed.scheme or parsed.netloc or not parsed.path.startswith("/api/"):
                self.send_error(403, "Only FindBack API paths are available")
                return
            if method == "POST" and not QUESTION_PATH.fullmatch(parsed.path):
                self.send_error(403, "Only existing-run questions are writable")
                return
            body = None
            if method == "POST":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = 0
                if length < 1 or length > 65536:
                    self.send_error(413, "Question body must be 1-65536 bytes")
                    return
                body = self.rfile.read(length)
            target = upstream + urllib.parse.urlunsplit(("", "", parsed.path, parsed.query, ""))
            headers = {
                "Authorization": "Bearer " + token,
                "Accept": self.headers.get("Accept", "application/json"),
            }
            if body is not None:
                headers["Content-Type"] = "application/json"
            request = urllib.request.Request(target, data=body, headers=headers, method=method)
            try:
                response = opener.open(request, timeout=120)
            except urllib.error.HTTPError as error:
                response = error
            except (TimeoutError, urllib.error.URLError):
                self.send_error(502, "FindBack upstream unavailable")
                return
            with response:
                payload = response.read()
                self.send_response(response.status)
                for header in ("Content-Type", "Content-Disposition", "X-Frame-Time-Ms"):
                    value = response.headers.get(header)
                    if value:
                        self.send_header(header, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if method != "HEAD":
                    self.wfile.write(payload)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=19001)
    parser.add_argument("--upstream", default="http://127.0.0.1:9000")
    args = parser.parse_args(argv)
    token = dotenv_values(args.env_file).get("AGENTX_AGENT_TOKEN")
    if not token:
        parser.error("AGENTX_AGENT_TOKEN is missing from --env-file")
    server = ThreadingHTTPServer(
        (args.listen_host, args.listen_port), make_handler(str(token), args.upstream)
    )
    server.daemon_threads = True
    print("FindBack read-and-ask proxy ready", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
