"""Measure native Skill discovery with FindBack and three NVIDIA VSS Skills together.

This deliberately stops at catalog routing. Recordings and VSS services are absent, so
no final-answer or end-to-end video capability claim can follow from the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

NAMES = (
    "findback-video-memory",
    "vss-ask-video",
    "vss-search-archive",
    "vss-summarize-video",
)
PATTERN = re.compile(
    r"(?P<name>findback-video-memory|vss-(?:ask-video|search-archive|summarize-video))/SKILL\.md"
)
VISIBLE = ("SKILL.md", "skill-card.md", "scripts", "references", "assets")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_skill(source: Path, target: Path) -> None:
    target.mkdir(parents=True)
    for name in VISIBLE:
        item = source / name
        if item.is_dir():
            shutil.copytree(item, target / name, ignore=shutil.ignore_patterns("__pycache__"))
        elif item.is_file():
            shutil.copy2(item, target / name)


def parse_claude(stream: str) -> dict:
    pending: dict[str, str] = {}
    attempted: list[str] = []
    loaded: list[str] = []
    reply = ""
    usage: dict = {}
    for line in stream.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            reply = str(event.get("result") or "")
            usage = {
                "num_turns": event.get("num_turns"),
                "duration_ms": event.get("duration_ms"),
                "total_cost_usd": event.get("total_cost_usd"),
                "is_error": event.get("is_error"),
            }
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("name") == "Skill":
                name = (block.get("input") or {}).get("skill")
                if isinstance(name, str):
                    attempted.append(name)
                    pending[str(block.get("id") or "")] = name
            elif block.get("type") == "tool_result":
                name = pending.get(str(block.get("tool_use_id") or ""))
                if name and not block.get("is_error"):
                    loaded.append(name)
    return {"attempted": attempted, "loaded": loaded, "reply": reply[:1000], "usage": usage}


def parse_codex(stream: str) -> dict:
    loaded: list[str] = []
    commands: list[str] = []
    reply = ""
    usage: dict = {}
    for line in stream.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.completed":
            usage = event.get("usage") or {}
        item = event.get("item") or {}
        if event.get("type") != "item.completed":
            continue
        if item.get("type") == "command_execution":
            command = str(item.get("command") or "")
            commands.append(command)
            for match in PATTERN.finditer(command):
                loaded.append(match["name"])
        elif item.get("type") == "agent_message":
            reply = str(item.get("text") or "")
    return {
        "attempted": loaded,
        "loaded": loaded,
        "commands": commands,
        "reply": reply[:1000],
        "usage": usage,
    }


def run_one(agent: str, case: dict, workspace: Path, timeout: int) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FINDBACK_", "AGENTX_"))}
    if agent == "claude":
        command = [
            "claude",
            "-p",
            case["prompt"],
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            "sonnet",
            "--max-turns",
            "6",
            "--permission-mode",
            "default",
            "--allowedTools",
            "Skill,Read,Glob,Grep",
            "--disallowedTools",
            "Bash,Edit,Write,WebFetch,WebSearch",
        ]
    else:
        command = [
            "codex",
            "exec",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "--ignore-user-config",
            "-s",
            "read-only",
            "-C",
            str(workspace),
            case["prompt"],
        ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stream, stderr, code = completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired as exc:
        stream = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr, code = "timeout", -1
    parsed = parse_claude(stream) if agent == "claude" else parse_codex(stream)
    chosen = parsed["loaded"][0] if parsed["loaded"] else "none"
    return {
        "id": case["id"],
        "expected": case["expected"],
        "chosen": chosen,
        "passed": chosen == case["expected"],
        "all_loaded": parsed["loaded"],
        "attempted": parsed["attempted"],
        "seconds": round(time.monotonic() - started, 3),
        "exit_code": code,
        "stderr_tail": stderr[-600:],
        "reply": parsed["reply"],
        "usage": parsed["usage"],
        "stream": stream,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("claude", "codex"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--nvidia-skills-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append", help="Pilot one or more case IDs")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Preserve existing reports: choose a new output path")
    project = Path(__file__).resolve().parents[2]
    protocol = json.loads(args.protocol.read_text())
    cases_path = project / protocol["case_file"]
    if sha(cases_path) != protocol["case_file_sha256"]:
        raise ValueError("Case file differs from the frozen protocol")
    sources = {
        "findback-video-memory": project / "skills/findback-video-memory",
        **{name: args.nvidia_skills_root / name for name in NAMES[1:]},
    }
    for name, expected in protocol["skill_sha256"].items():
        if sha(sources[name] / "SKILL.md") != expected:
            raise ValueError(f"Skill {name} differs from the frozen protocol")
    cases = json.loads(cases_path.read_text())["cases"]
    if args.case_id:
        ids = set(args.case_id)
        cases = [case for case in cases if case["id"] in ids]
        if len(cases) != len(ids):
            raise ValueError("Unknown case ID")
    args.output.mkdir(parents=True)
    report = {
        "status": "running",
        "agent": args.agent,
        "protocol_sha256": sha(args.protocol),
        "created_at": datetime.now(UTC).isoformat(),
        "cli_version": subprocess.run(
            ["claude" if args.agent == "claude" else "codex", "--version"],
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "rows": [],
    }
    with tempfile.TemporaryDirectory(prefix=f"agentx-vss-native-{args.agent}-") as parent:
        for case in cases:
            workspace = Path(parent) / case["id"]
            skill_dir = workspace / (
                ".claude/skills" if args.agent == "claude" else ".agents/skills"
            )
            for name, source in sources.items():
                install_skill(source, skill_dir / name)
            result = run_one(args.agent, case, workspace, args.timeout)
            stream = result.pop("stream")
            (args.output / f"{case['id']}.jsonl").write_text(stream)
            report["rows"].append(result)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(case["id"], case["expected"], "->", result["chosen"], flush=True)
    report["status"] = "complete"
    report["correct"] = sum(row["passed"] for row in report["rows"])
    report["cases"] = len(report["rows"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("NATIVE_ROUTING_COMPLETE", report["correct"], report["cases"], flush=True)


if __name__ == "__main__":
    main()
