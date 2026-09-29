"""Stop explicit live-location requests without a recording before launching OpenClaw.

Usage:
    python integrations/openclaw/run_guarded.py --message-file prompt.txt -- \
      openclaw agent exec --config openclaw.json --message-file prompt.txt --json

This is a narrow host-side guard for an observed outer-agent failure. Other prompts are
passed to the supplied OpenClaw command without modifying its arguments or environment.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

LIVE_QUERY = re.compile(
    r"(?:\b(?:where(?:\s+(?:is|are)|['’]s)|locate|find)\b.{0,120}"
    r"\b(?:right\s+now|now|currently|live\s+location|at\s+(?:this|the)\s+moment)\b"
    r"|\bcurrent\s+location\s+of\b)",
    re.IGNORECASE | re.DOTALL,
)
RECORDING_CONTEXT = re.compile(
    r"\b(?:in|from|of|at\s+the\s+end\s+of)\s+"
    r"(?:(?:the|this|that|my|a|named)\s+)?"
    r"(?:video|recording|clip|footage)\b",
    re.IGNORECASE,
)

REFUSAL = (
    "A saved recording can show a last supported sighting, but it cannot establish "
    "a device's current location. Name a recording if you want me to check it."
)


def needs_live_location_guard(prompt: str) -> bool:
    """Catch clear current-world location requests with no stated recording context."""
    return bool(LIVE_QUERY.search(prompt) and not RECORDING_CONTEXT.search(prompt))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message-file", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("an OpenClaw command is required after --")
    prompt = args.message_file.read_text(encoding="utf-8")
    if needs_live_location_guard(prompt):
        print(
            json.dumps(
                {
                    "ok": True,
                    "status": "ok",
                    "guarded": True,
                    "guard": "live_location_without_recording",
                    "final": REFUSAL,
                    "payloads": [{"text": REFUSAL, "mediaUrl": None}],
                    "model": None,
                    "provider": None,
                }
            )
        )
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
