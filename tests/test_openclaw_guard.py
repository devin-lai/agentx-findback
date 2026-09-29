"""The host guard blocks unsupported live location without a false video refusal."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "integrations/openclaw/run_guarded.py"
SPEC = importlib.util.spec_from_file_location("openclaw_guard", MODULE_PATH)
assert SPEC and SPEC.loader
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def test_live_location_without_recording_is_guarded(tmp_path, capsys):
    prompt = tmp_path / "request.txt"
    prompt.write_text("Where is my phone right now? I have not provided a recording.")
    marker = tmp_path / "command-ran"
    result = guard.main(
        [
            "--message-file",
            str(prompt),
            "--",
            "python3",
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
        ]
    )
    answer = json.loads(capsys.readouterr().out)
    assert result == 0
    assert answer["guarded"] is True
    assert "current location" in answer["final"]
    assert not marker.exists()


def test_recording_relative_question_is_forwarded(tmp_path):
    prompt = tmp_path / "request.txt"
    prompt.write_text(
        'Where is the Black remote now in the recording "Composite desk - three remotes"?'
    )
    marker = tmp_path / "command-ran"
    result = guard.main(
        [
            "--message-file",
            str(prompt),
            "--",
            "python3",
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
        ]
    )
    assert result == 0
    assert marker.read_text() == "ran"


def test_past_sighting_question_is_not_a_live_location_request():
    assert not guard.needs_live_location_guard(
        'Where was the remote last seen at 00:07 in the "Controlled desk fixture" video?'
    )


def test_no_recording_live_location_stays_guarded_after_prior_evidence_exists(tmp_path):
    (tmp_path / "findback-evidence.jpg").write_bytes(b"prior evidence")
    assert guard.needs_live_location_guard("Can you find my phone right now?")
    assert guard.needs_live_location_guard("Where's my phone now?")
    assert guard.needs_live_location_guard("What is the current location of my phone?")
