"""Governance checks for the portable Agent Skill and the internal prompt-contract Skills.

Covers the Agent Skills frontmatter contract, the governance files of every skill directory, the
portable skill's command-line client (argument parsing, token redaction, redirect refusal and an
end-to-end run against a real local server), the routing encoded in the internal Skills' eval
sets, and the integrity of each directory's detached signature.
"""

import base64
import hashlib
import http.server
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agentx.api.app import create_app
from agentx.config import Settings

ROOT = Path(__file__).parents[1]
PORTABLE = ROOT / "skills" / "findback-video-memory"
AUDITOR = ROOT / "skills" / "findback-evidence-audit"
INTERNAL = sorted(p for p in (ROOT / "src/agentx/skills").iterdir() if (p / "SKILL.md").is_file())
SKILL_DIRS = [
    *sorted(p for p in (ROOT / "skills").iterdir() if (p / "SKILL.md").is_file()),
    *INTERNAL,
]
CLI = PORTABLE / "scripts" / "findback.py"
PUBLIC_KEY = ROOT / "skills" / "findback-skills.pub"
SPEC_FIELDS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
# The internal SKILL.md files predate this package and keep a top-level `version`. Saved answers
# cite the SHA-256 of their exact bytes, so they are never edited to drop it.
INTERNAL_EXTRA_FIELDS = {"version"}
CARD_SECTIONS = (
    "## Description:",
    "## Owner",
    "## Third-Party Community Consideration",
    "### License/Terms of Use:",
    "## Use Case:",
    "### Deployment Geography for Use:",
    "## Requirements / Dependencies:",
    "**Requires API Key or External Credential:**",
    "**Credential Type(s):**",
    "## Known Risks and Mitigations:",
    "## Reference(s):",
    "## Skill Output:",
    "## Skill Version(s):",
    "## Ethical Considerations:",
)
CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TOKEN = "findback-test-token-7c1e93d2"


def skill_id(path: Path) -> str:
    return path.name


def frontmatter(path: Path) -> tuple[dict, str]:
    """A strict reader for the flat `key: value` frontmatter these files use (no YAML library)."""
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", text, re.DOTALL)
    assert match, f"{path} must start with a --- frontmatter block"
    fields: dict = {}
    parent = None
    for line in match.group(1).splitlines():
        nested = re.match(r"^  ([A-Za-z0-9_-]+):[ \t]*(.*)$", line)
        if nested and parent is not None:
            fields[parent][nested[1]] = nested[2].strip().strip('"')
            continue
        top = re.match(r"^([A-Za-z0-9_-]+):[ \t]*(.*)$", line)
        assert top, f"{path}: unexpected frontmatter line {line!r}"
        key, value = top[1], top[2].strip()
        assert key not in fields, f"{path}: duplicate key {key}"
        if value:
            assert not re.search(r":\s|\s#", value), f"{path}: {key} is not a plain YAML scalar"
            fields[key], parent = value.strip('"'), None
        else:
            fields[key], parent = {}, key
    return fields, match.group(2)


def load_module(path: Path, name: str):
    """Import a script without writing __pycache__ into a signed directory."""
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


@pytest.fixture(scope="module")
def cli():
    return load_module(CLI, "findback_cli")


def evals(skill_dir: Path) -> dict:
    return json.loads((skill_dir / "evals" / "evals.json").read_text())


# --- Frontmatter and layout -----------------------------------------------------------------


@pytest.mark.parametrize("skill_dir", SKILL_DIRS, ids=skill_id)
def test_skill_frontmatter_follows_the_agent_skills_contract(skill_dir):
    fields, body = frontmatter(skill_dir / "SKILL.md")
    allowed = SPEC_FIELDS | (INTERNAL_EXTRA_FIELDS if skill_dir in INTERNAL else set())
    assert set(fields) <= allowed, set(fields) - allowed
    name, description = fields["name"], fields["description"]
    assert name == skill_dir.name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name) and len(name) <= 64
    assert 1 <= len(description) <= 1024
    assert len(fields.get("compatibility", "")) <= 500
    assert isinstance(fields.get("metadata", {}), dict)
    assert body.strip()


def test_portable_skill_is_a_short_routing_document_with_negative_triggers():
    fields, body = frontmatter(PORTABLE / "SKILL.md")
    assert body.strip().startswith("# ")
    assert fields["license"] == "Apache-2.0"
    assert fields["metadata"]["version"] and fields["metadata"]["author"]
    description = fields["description"]
    assert "Not for" in description
    for negative in ("live location", "people", "summarizing", "unseen object", "editing"):
        assert negative in description
    assert len((PORTABLE / "SKILL.md").read_text().splitlines()) < 100
    assert len(body) / 4 < 5000, "keep the body under roughly 5K tokens"
    for heading in ("## Required questions", "## Routing", "## Output contract", "## Safety"):
        assert heading in body
    assert "MEDIA:" in body and "FINDBACK_TOKEN" in body
    for link in re.findall(r"\]\(([^)#]+)\)", body):
        assert (PORTABLE / link).is_file(), link
    for path in ("scripts/findback.py", "references/api.md", "references/answer-semantics.md"):
        assert (PORTABLE / path).is_file()
    assert (PORTABLE / "references/troubleshooting.md").is_file()


def test_internal_skill_bytes_match_the_hashes_their_governance_files_cite():
    for skill_dir in INTERNAL:
        digest = hashlib.sha256((skill_dir / "SKILL.md").read_bytes()).hexdigest()
        assert evals(skill_dir)["skill_sha256"] == digest, skill_dir.name
        assert digest in (skill_dir / "skill-card.md").read_text(), skill_dir.name
        benchmark = (skill_dir / "BENCHMARK.md").read_text()
        assert digest in benchmark, skill_dir.name


# --- Governance files -----------------------------------------------------------------------


@pytest.mark.parametrize("skill_dir", SKILL_DIRS, ids=skill_id)
def test_every_skill_has_a_card_an_eval_set_and_a_benchmark(skill_dir):
    card = (skill_dir / "skill-card.md").read_text()
    for section in CARD_SECTIONS:
        assert section in card, section
    assert "AgentX FindBack team" in card
    assert "not owned or developed by NVIDIA" in card and "not an NVIDIA-verified" in card
    assert (skill_dir / "BENCHMARK.md").is_file()

    data = evals(skill_dir)
    assert data["skill_name"] == skill_dir.name
    cases = data["evals"]
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
    for case in cases:
        assert CASE_ID.match(case["id"]), case["id"]
        assert case["prompt"].strip() and case["expected_output"].strip()
        assert case["assertions"] and all(isinstance(a, str) for a in case["assertions"])
        assert isinstance(case["should_trigger"], bool)
        expected = skill_dir.name if case["should_trigger"] else None
        assert case["expected_skill"] == expected, case["id"]
    negatives = [case for case in cases if not case["should_trigger"]]
    assert negatives, "an eval set needs cases where the skill must not be used"
    if skill_dir == PORTABLE:
        assert len(cases) >= 12 and len(negatives) >= 4


def test_portable_benchmark_is_measured_and_pins_the_dataset():
    text = (PORTABLE / "BENCHMARK.md").read_text()
    digest = hashlib.sha256((PORTABLE / "evals/evals.json").read_bytes()).hexdigest()
    assert f"sha256:{digest}" in text, "BENCHMARK.md must cite the current evals.json digest"
    glance = text.split("## Results at a Glance", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in glance.splitlines() if line.startswith("| ") and "---" not in line]
    assert len(rows) >= 5
    for row in rows[1:]:
        cells = [cell.strip().strip("*") for cell in row.strip("|").split("|")]
        assert re.fullmatch(r"\d+/\d+", cells[1]), row
        assert re.fullmatch(r"\d+/\d+", cells[2]), row
    assert "1/15" in glance and "15/15" in glance


def run_checks(case: dict, transcript: str, reply: str, env: dict | None = None) -> bool:
    """Apply an eval case's deterministic checks, as documented in evals.json."""
    for check in case["checks"]:
        kind = check["type"]
        if kind == "transcript_not_contains_env":
            value = (env or {}).get(check["env"])
            if value and value in transcript + reply:
                return False
            continue
        text = f"{transcript}\n{reply}" if kind.startswith("transcript") else reply
        found = re.search(check["pattern"], text) is not None
        if found == kind.endswith("_not_matches"):
            return False
    return True


def test_portable_eval_checks_separate_good_and_bad_transcripts():
    cases = {case["id"]: case for case in evals(PORTABLE)["evals"]}
    for case in cases.values():
        for check in case["checks"]:
            if "pattern" in check:
                re.compile(check["pattern"])
    ask = 'python3 scripts/findback.py ask 0123 "Where was the Red toolkit last seen?" --at-ms 7000 --save-frame ./ev\n'
    reply = (
        "The last supported observation of Red toolkit at or before 00:07.0 is at 00:05.8, in "
        "Right area. Its position at the requested time is not confirmed. Evidence frame: "
        "/tmp/ev/findback-evidence-a9ade00b38bb487a89ad6132b8638246.jpg"
    )
    env = {"FINDBACK_TOKEN": "s3cret-value"}
    explicit = cases["fvm-001-last-seen-explicit"]
    assert run_checks(explicit, ask, reply, env)
    assert not run_checks(explicit, ask, reply + " It was probably put in the top drawer.", env)
    assert not run_checks(explicit, ask + "echo s3cret-value\n", reply, env)
    assert not run_checks(explicit, ask.replace("7000", "9000"), reply, env)

    media = cases["fvm-006-openclaw-media-line"]
    call = 'python3 scripts/findback.py ask 0123 "Where is the Blue remote?" --at-ms 4000 --save-frame /tmp/ev\n'
    good = "The Blue remote is visible at 00:04.0 in the Center area.\nMEDIA:/tmp/ev/findback-evidence-1.jpg\n"
    assert run_checks(media, call, good)
    assert not run_checks(
        media,
        call,
        good.replace(
            "MEDIA:/tmp/ev/findback-evidence-1.jpg", "`MEDIA:/tmp/ev/findback-evidence-1.jpg`"
        ),
    )
    assert not run_checks(media, call, "Here it is: MEDIA:/tmp/ev/findback-evidence-1.jpg")

    which = cases["fvm-007-ask-which-recording"]
    listing = "python3 scripts/findback.py videos\n"
    assert run_checks(
        which, listing, "Which recording do you mean: the fixture or the bumped camera one?"
    )
    assert not run_checks(
        which, listing + 'python3 scripts/findback.py ask 0123 "x" --at-end\n', "00:11.8"
    )

    negative = cases["fvm-101-neg-live-location"]
    assert run_checks(negative, "", "I cannot know where your phone is right now.")
    assert not run_checks(negative, "python3 scripts/findback.py videos\n", "Checking recordings.")


# --- The command-line client ----------------------------------------------------------------


def test_cli_parses_every_command(cli):
    parser = cli.build_parser()
    args = parser.parse_args(
        ["ask", "run1", "Where was it?", "--at-ms", "7000", "--no-planner", "--save-frame", "out"]
    )
    assert (args.command, args.run_id, args.at_ms, args.at_end) == ("ask", "run1", 7000, False)
    assert args.no_planner and not args.no_skills and args.save_frame == Path("out")
    assert parser.parse_args(["ask", "run1", "q", "--at-end"]).at_end
    index = parser.parse_args(["index", "video1", "--wait", "--timeout", "30"])
    assert (index.backend, index.wait, index.wait_timeout, index.poll) == ("reference", True, 30, 1)
    assert index.request_timeout == 60 and not index.reuse
    assert parser.parse_args(["--url", "http://a:1", "health"]).url == "http://a:1"
    assert parser.parse_args(["health", "--url", "http://b:2"]).url == "http://b:2"
    both = parser.parse_args(["--request-timeout", "5", "videos", "--pretty"])
    assert both.request_timeout == 5 and both.pretty
    register = parser.parse_args(
        ["register", "v", "--name", "Mug", "--at-ms", "0", "--box", "0.1,0.2,0.3,0.4", "--pixels"]
    )
    assert register.label == "custom" and register.pixels
    assert parser.parse_args(["report", "q1", "--out", "r.zip"]).force is False
    assert parser.parse_args(["demo"]).scene == "fixed"
    by_title = parser.parse_args(
        ["ask", "--video", '"Controlled desk fixture"', "Where was it?", "--at-ms", "7000"]
    )
    assert by_title.run_id is None and by_title.video == '"Controlled desk fixture"'


def test_cli_resolves_only_the_named_recording_and_tolerates_wrapped_quotes(cli):
    class Catalog:
        def json(self, method, path):
            assert (method, path) == ("GET", "/api/v1/videos")
            common = {"regions": [], "objects": []}
            return [
                {
                    **common,
                    "id": "fixed",
                    "title": "Controlled desk fixture",
                    "runs": [
                        {
                            "id": "fixed-old",
                            "status": "complete",
                            "object_ids": [],
                            "regions": [],
                            "created_at": "2026-01-01",
                        },
                        {
                            "id": "fixed-new",
                            "status": "complete",
                            "object_ids": [],
                            "regions": [],
                            "created_at": "2026-01-02",
                        },
                        {
                            "id": "fixed-stale",
                            "status": "complete",
                            "object_ids": ["new-object-not-in-video"],
                            "regions": [],
                            "created_at": "2026-01-04",
                        },
                    ],
                },
                {
                    **common,
                    "id": "bumped",
                    "title": "Controlled desk fixture, bumped camera",
                    "runs": [
                        {
                            "id": "bumped-run",
                            "status": "complete",
                            "object_ids": [],
                            "regions": [],
                            "created_at": "2026-01-03",
                        }
                    ],
                },
            ]

    catalog = Catalog()
    for title in (
        "Controlled desk fixture",
        '"Controlled desk fixture"',
        "'\"Controlled desk fixture\"'",
        "“Controlled desk fixture”",
    ):
        run_id, recording = cli.resolve_recording(catalog, title)
        assert (run_id, recording["id"]) == ("fixed-new", "fixed")
    with pytest.raises(cli.CliError, match="No recording is titled exactly"):
        cli.resolve_recording(catalog, "Controlled desk")

    class DuplicateCatalog(Catalog):
        def json(self, method, path):
            videos = super().json(method, path)
            return [*videos, {**videos[0], "id": "other-video"}]

    with pytest.raises(cli.CliError, match="2 recordings are titled"):
        cli.resolve_recording(DuplicateCatalog(), "Controlled desk fixture")


@pytest.mark.parametrize(
    "argv",
    [
        ["ask", "run1", "Where?"],  # the cutoff is required: never guessed
        ["ask", "run1", "Where?", "--at-ms", "5", "--at-end"],
        ["ask", "run1", "Where?", "--at-ms", "-5"],
        ["index", "video1", "--backend", "yolo"],
        ["register", "v", "--name", "Mug", "--at-ms", "0"],
        ["index", "video1", "--sample-fps", "40"],
        ["unknown-command"],
    ],
)
def test_cli_usage_errors_exit_2_with_a_json_error(cli, capsys, argv):
    with pytest.raises(SystemExit) as raised:
        cli.main(argv)
    assert raised.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"]["exit_code"] == 2


def test_cli_rejects_invalid_boxes_and_identifiers_before_any_request(cli, monkeypatch, capsys):
    monkeypatch.setenv("FINDBACK_URL", "http://127.0.0.1:9")
    assert cli.main(["register", "v1", "--name", "M", "--at-ms", "0", "--box", "10,20,30,40"]) == 2
    assert "normalized" in json.loads(capsys.readouterr().out)["error"]["message"]
    assert cli.main(["register", "v1", "--name", "M", "--at-ms", "0", "--box", "0.1,0.2"]) == 2
    capsys.readouterr()
    assert cli.main(["frame", "../../etc", "--out", "x"]) == 2
    assert "Invalid observation ID" in json.loads(capsys.readouterr().out)["error"]["message"]
    monkeypatch.setenv("FINDBACK_URL", "http://user:pw@127.0.0.1:9")
    assert cli.main(["health"]) == 2
    assert "credentials" in json.loads(capsys.readouterr().out)["error"]["message"]


class Hostile(http.server.BaseHTTPRequestHandler):
    """Echoes the Authorization header back in errors, and redirects /api/health elsewhere."""

    redirect_to = ""
    seen: list = []

    def do_GET(self):  # noqa: N802
        type(self).seen.append((self.path, self.headers.get("Authorization")))
        if self.path == "/api/health" and self.redirect_to:
            self.send_response(302)
            self.send_header("Location", self.redirect_to + "/api/health")
            self.end_headers()
            return
        body = json.dumps({"detail": f"rejected {self.headers.get('Authorization')}"}).encode()
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def serve(handler):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_cli_redacts_the_token_from_every_output(cli, monkeypatch, capsys):
    server, url = serve(type("Echo", (Hostile,), {"seen": []}))
    try:
        monkeypatch.setenv("FINDBACK_URL", url)
        monkeypatch.setenv("FINDBACK_TOKEN", TOKEN)
        assert cli.main(["videos"]) == 3
        output = capsys.readouterr()
        error = json.loads(output.out)["error"]
        assert error["http_status"] == 401
        assert TOKEN not in output.out + output.err
        assert "Bearer [REDACTED]" in error["message"]
        assert "FINDBACK_TOKEN" in output.err  # the hint names the variable, never its value
    finally:
        server.shutdown()


def test_cli_refuses_redirects_so_the_token_is_never_forwarded(cli, monkeypatch, capsys):
    target_handler = type("Target", (Hostile,), {"seen": []})
    target, target_url = serve(target_handler)
    origin, origin_url = serve(type("Origin", (Hostile,), {"seen": [], "redirect_to": target_url}))
    try:
        monkeypatch.setenv("FINDBACK_URL", origin_url)
        monkeypatch.setenv("FINDBACK_TOKEN", TOKEN)
        assert cli.main(["health"]) == 3
        output = capsys.readouterr()
        assert "redirect" in json.loads(output.out)["error"]["message"]
        assert TOKEN not in output.out + output.err
        assert target_handler.seen == []
    finally:
        origin.shutdown()
        target.shutdown()


def test_cli_reports_an_unreachable_server_as_exit_3(cli, monkeypatch, capsys):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv("FINDBACK_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("FINDBACK_TOKEN", TOKEN)
    assert cli.main(["health"]) == 3
    output = capsys.readouterr()
    assert "Cannot reach FindBack" in json.loads(output.out)["error"]["message"]
    assert TOKEN not in output.out + output.err


# --- End to end against a real server --------------------------------------------------------


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def agentx_command() -> list[str]:
    script = Path(sys.executable).with_name("agentx")
    if script.is_file():
        return [str(script)]
    return [sys.executable, "-c", "from agentx.cli import app; app()"]


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    """`agentx serve` on a free port with a temporary data directory and no planner or Cosmos.

    It runs from the data directory, so no `.env` is read, with every AGENTX_ variable of the
    calling environment removed.
    """
    data = tmp_path_factory.mktemp("findback-server")
    port = free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AGENTX_", "FINDBACK_"))}
    env.update(
        {
            "AGENTX_DATA_DIR": str(data),
            "AGENTX_API_TOKEN": TOKEN,
            "AGENTX_WEB_DIST": str(data / "no-web"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    log_path = data / "server.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [*agentx_command(), "serve", "--host", "127.0.0.1", "--port", str(port)],
            cwd=data,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        url = f"http://127.0.0.1:{port}"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 90
        try:
            while True:
                if process.poll() is not None:
                    pytest.fail("agentx serve exited:\n" + log_path.read_text()[-4000:])
                try:
                    with opener.open(url + "/api/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    pytest.fail("agentx serve did not start:\n" + log_path.read_text()[-4000:])
                time.sleep(0.25)
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def run_cli(url: str, *args: str, token: str = TOKEN, cwd: Path) -> tuple[int, dict, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AGENTX_", "FINDBACK_"))}
    env.update({"FINDBACK_URL": url, "FINDBACK_TOKEN": token, "PYTHONDONTWRITEBYTECODE": "1"})
    result = subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert token not in result.stdout + result.stderr
    return result.returncode, json.loads(result.stdout), result.stderr


def test_cli_end_to_end_against_a_local_server(live_server, tmp_path):
    code, health, _ = run_cli(live_server, "health", cwd=tmp_path)
    assert code == 0, health
    assert health["auth_required"] and health["token_configured"]
    assert health["planner"] is False and health["index_backends"]["cosmos"] is False

    code, video, _ = run_cli(live_server, "demo", cwd=tmp_path)
    assert code == 0, video
    assert video["is_fixture"] and {"Red toolkit", "Blue remote"} <= {
        o["name"] for o in video["objects"]
    }

    code, index, _ = run_cli(
        live_server, "index", video["id"], "--wait", "--timeout", "240", cwd=tmp_path
    )
    assert code == 0, index
    run = index["run"]
    assert run["status"] == "complete" and run["backend"] == "reference" and not index["reused"]
    assert run["skill"].startswith("observe-object-events@")
    code, again, _ = run_cli(live_server, "index", video["id"], "--reuse", "--wait", cwd=tmp_path)
    assert code == 0 and again["reused"] and again["run"]["id"] == run["id"]

    frames = tmp_path / "frames"
    code, answer, stderr = run_cli(
        live_server,
        "ask",
        run["id"],
        "Where was the Red toolkit last seen?",
        "--at-ms",
        "7000",
        "--save-frame",
        str(frames),
        cwd=tmp_path,
    )
    assert code == 0, answer
    assert answer["status"] == "last_seen" and answer["cutoff_ms"] == 7000
    evidence = answer["evidence"]
    assert re.fullmatch(r"[0-9a-f]{32}", evidence["observation_id"])
    assert 5000 <= evidence["at_ms"] < 6000 and evidence["zone_name"] == "Right area"
    assert "not confirmed" in answer["answer"] and answer["planner"] == "local"
    frame = Path(answer["evidence_frame"])
    assert frame.is_absolute() and frame.parent == frames.resolve()
    assert frame.read_bytes().startswith(b"\xff\xd8")
    assert answer["media"] == f"MEDIA:{frame}"
    assert "Evidence frame saved" in stderr

    # How a small local agent phrased the same question: wrapped in quotes, cutoff restated.
    code, restated, _ = run_cli(
        live_server,
        "ask",
        run["id"],
        '"Where was the Red toolkit last seen at 00:07?"',
        "--at-ms",
        "7000",
        cwd=tmp_path,
    )
    assert code == 0 and restated["status"] == "last_seen", restated
    assert restated["evidence"]["observation_id"] == evidence["observation_id"]

    code, by_title, _ = run_cli(
        live_server,
        "ask",
        "--video",
        f'"{video["title"]}"',
        "Where was the Red toolkit last seen?",
        "--at-ms",
        "7000",
        cwd=tmp_path,
    )
    assert code == 0 and by_title["run_id"] == run["id"], by_title
    assert by_title["recording"] == video["title"]

    report = tmp_path / "evidence.zip"
    code, saved, _ = run_cli(
        live_server, "report", answer["question_id"], "--out", str(report), cwd=tmp_path
    )
    assert code == 0, saved
    with zipfile.ZipFile(report) as archive:
        assert "report.html" in archive.namelist()

    audited = subprocess.run(
        [sys.executable, str(AUDITOR / "scripts/audit_report.py"), str(report)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert audited.returncode == 0, audited.stdout + audited.stderr
    assert json.loads(audited.stdout)["cutoff_ms"] == 7000
    tampered = tmp_path / "tampered-evidence.zip"
    with zipfile.ZipFile(report) as original, zipfile.ZipFile(tampered, "w") as changed:
        for info in original.infolist():
            content = original.read(info.filename)
            if info.filename.startswith("frames/"):
                content = content[:-1] + bytes([content[-1] ^ 1])
            changed.writestr(info, content)
    refused = subprocess.run(
        [sys.executable, str(AUDITOR / "scripts/audit_report.py"), str(tampered)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode == 1
    assert "Integrity check failed" in json.loads(refused.stdout)["error"]

    code, error, _ = run_cli(live_server, "videos", token="wrong-token-0000", cwd=tmp_path)
    assert code == 3 and error["error"]["http_status"] == 401


# --- Internal Skill routing -------------------------------------------------------------------

PLANNER_MODEL = "routing-test-planner"
VALID_REVIEW = (
    '{"summary":"The sampled frames show the desk.","evidence_frame_ids":[0],'
    '"uncertainty":"Only sampled frames are known."}'
)


def stub_model(base, key, model, messages, **kwargs):
    """No real model: the planner is unreachable and Cosmos answers minimally."""
    if model == PLANNER_MODEL:
        raise ValueError("The routing test has no planner.")
    content = messages[-1]["content"]
    texts = [part.get("text", "") for part in content] if isinstance(content, list) else [content]
    if any(str(text).startswith("frame_id=") for text in texts):
        return '{"present":false,"x":null,"y":null,"note":"not visible"}'
    return VALID_REVIEW


@pytest.fixture(scope="module")
def routing(tmp_path_factory):
    settings = Settings(
        data_dir=tmp_path_factory.mktemp("routing"),
        database_url="",
        api_token="",
        planner_base_url="",
        planner_model="",
        stepfun_api_key="",
        stepfun_model="",
        cosmos_base_url="",
        cosmos_backend="http",
        cosmos_model_path="",
        worker_enabled=False,
        web_dist=Path("not-built"),
        _env_file=None,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        scenes = {}
        for scene in ("fixed", "bumped"):
            video = client.post("/api/v1/demo", params={"scene": scene}).json()
            queued = client.post(f"/api/v1/videos/{video['id']}/runs", json={}).json()
            app.state.services.indexer.process(queued["id"])
            scenes[scene] = (video, client.get(f"/api/v1/runs/{queued['id']}").json())
        yield app, client, scenes


INTERNAL_CASES = [
    pytest.param(skill_dir.name, case, id=case["id"])
    for skill_dir in INTERNAL
    for case in evals(skill_dir)["evals"]
]


@pytest.mark.parametrize(("skill", "case"), INTERNAL_CASES)
def test_internal_skill_routing_matches_its_eval_set(routing, monkeypatch, skill, case):
    app, client, scenes = routing
    services = app.state.services
    spec = case["routing"]
    video, run = scenes[spec["scene"]]
    monkeypatch.setattr(services.providers, "_chat", stub_model)
    if spec["surface"] == "index":
        loaded = [run["provenance"]["skill"]["name"]]
    else:
        body = {"text": case["prompt"], "at_ms": spec["at_ms"], "use_skills": spec["use_skills"]}
        if spec.get("select"):
            body["object_id"] = next(
                o["id"] for o in video["objects"] if o["name"] == spec["select"]
            )
        if spec["surface"] == "question":
            planner = spec["planner"]
            monkeypatch.setattr(
                services.settings,
                "planner_base_url",
                "http://planner.invalid/v1" if planner else "",
            )
            monkeypatch.setattr(
                services.settings, "planner_model", PLANNER_MODEL if planner else ""
            )
            body["use_provider"] = spec["use_provider"]
            response = client.post(f"/api/v1/runs/{run['id']}/questions", json=body)
        else:
            monkeypatch.setattr(services.settings, "cosmos_base_url", "http://cosmos.invalid/v1")
            response = client.post(f"/api/v1/runs/{run['id']}/review", json=body)
        assert response.status_code == 200, response.text
        loaded = [trace["name"] for trace in response.json()["skills"]]
    assert (skill in loaded) is case["should_trigger"], loaded


# --- Signatures and packaging ------------------------------------------------------------------


@pytest.mark.parametrize("skill_dir", SKILL_DIRS, ids=skill_id)
def test_signature_manifest_matches_the_directory(skill_dir):
    """Integrity without model-signing: every file is covered, unchanged, and nothing was added.

    The ECDSA signature itself is checked by `scripts/ops/sign_skills.py verify`, which needs the
    separately installed model-signing package.
    """
    bundle = json.loads((skill_dir / "skill.oms.sig").read_text())
    assert bundle["mediaType"] == "application/vnd.dev.sigstore.bundle.v0.3+json"
    key_hint = hashlib.sha256(PUBLIC_KEY.read_bytes()).hexdigest()
    assert bundle["verificationMaterial"]["publicKey"]["hint"] == key_hint
    envelope = bundle["dsseEnvelope"]
    assert envelope["payloadType"] == "application/vnd.in-toto+json" and envelope["signatures"]
    statement = json.loads(base64.b64decode(envelope["payload"]))
    assert statement["predicateType"] == "https://model_signing/signature/v1.0"
    signed = {r["name"]: r["digest"] for r in statement["predicate"]["resources"]}
    present = {
        path.relative_to(skill_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(skill_dir.rglob("*"))
        if path.is_file() and path.name != "skill.oms.sig"
    }
    assert present == signed, "a skill file changed or was added after signing; sign again"


def test_signing_helpers_find_every_skill_and_refuse_machine_local_files(tmp_path):
    signing = load_module(ROOT / "scripts" / "ops" / "sign_skills.py", "sign_skills_under_test")
    assert [p.name for p in signing.skill_dirs(ROOT)] == [p.name for p in SKILL_DIRS]
    skill = tmp_path / "demo-skill"
    (skill / "scripts" / "__pycache__").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo-skill\ndescription: x\n---\n# Demo\n")
    (skill / "scripts" / "__pycache__" / "tool.cpython-312.pyc").write_bytes(b"\0")
    (skill / ".DS_Store").write_bytes(b"\0")
    assert signing.stray_files(skill) == [
        ".DS_Store",
        "scripts/__pycache__",
        "scripts/__pycache__/tool.cpython-312.pyc",
    ]


@pytest.mark.skipif(
    not os.environ.get("FINDBACK_SIGNING_PYTHON"),
    reason="set FINDBACK_SIGNING_PYTHON to an interpreter with model-signing installed",
)
def test_every_signature_verifies_cryptographically():
    result = subprocess.run(
        [os.environ["FINDBACK_SIGNING_PYTHON"], str(ROOT / "scripts/ops/sign_skills.py"), "verify"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{len(SKILL_DIRS)}/{len(SKILL_DIRS)} signatures verified" in result.stdout


def test_release_bundle_ships_the_portable_skill_but_never_a_private_key(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts" / "ops"))
    packager = load_module(
        ROOT / "scripts" / "ops" / "package_release.py", "package_release_under_test"
    )
    for name in [
        "skills/findback-evidence-audit/SKILL.md",
        "skills/findback-evidence-audit/skill.oms.sig",
        "skills/findback-evidence-audit/scripts/audit_report.py",
        "skills/findback-video-memory/SKILL.md",
        "skills/findback-video-memory/skill.oms.sig",
        "skills/findback-video-memory/scripts/findback.py",
        "skills/findback-skills.pub",
        "skills/findback-video-memory/scripts/__pycache__/findback.cpython-312.pyc",
        "skills/signing.key",
        ".private/signing/findback-skills-signing.key",
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    found = {p.relative_to(tmp_path).as_posix() for p in packager.release_files(tmp_path)}
    assert found == {
        "skills/findback-evidence-audit/SKILL.md",
        "skills/findback-evidence-audit/skill.oms.sig",
        "skills/findback-evidence-audit/scripts/audit_report.py",
        "skills/findback-video-memory/SKILL.md",
        "skills/findback-video-memory/skill.oms.sig",
        "skills/findback-video-memory/scripts/findback.py",
        "skills/findback-skills.pub",
    }


def test_git_ignores_the_private_signing_key_but_not_the_published_files(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_bytes((ROOT / ".gitignore").read_bytes())

    def ignored(path):
        return (
            subprocess.run(
                ["git", "check-ignore", "--no-index", "-q", path], cwd=tmp_path, check=False
            ).returncode
            == 0
        )

    assert ignored(".private/signing/findback-skills-signing.key")
    for path in (
        "skills/findback-skills.pub",
        "skills/findback-video-memory/skill.oms.sig",
        "skills/findback-video-memory/evals/evals.json",
        "src/agentx/skills/verify-location-answer/skill.oms.sig",
    ):
        assert not ignored(path), path
