import copy
import importlib.util
import sqlite3
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "saved_memory_evaluation", Path(__file__).parents[1] / "scripts/eval/evaluate_saved_memory.py"
)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_snapshot_never_writes_original(tmp_path):
    source, destination = tmp_path / "source.db", tmp_path / "snapshot.db"
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE sample (value TEXT)")
        db.execute("INSERT INTO sample VALUES ('original')")
    before = source.read_bytes()
    digest = evaluation.snapshot_database(source, destination)
    assert len(digest) == 64
    with sqlite3.connect(destination) as db:
        db.execute("DELETE FROM sample")
    assert source.read_bytes() == before
    with pytest.raises(ValueError):
        evaluation.snapshot_database(source, destination)
    with pytest.raises(ValueError):
        evaluation.snapshot_database(tmp_path / "missing.db", tmp_path / "new.db")


def test_saved_memory_scorer_rejects_wrong_state_scope_and_future():
    dataset = {"object_id": "cup", "run_id": "run", "video_id": "video"}
    case = {
        "at_ms": 30000,
        "expected_resolution": "unique",
        "expected_intent": "location",
        "expected_status": "unknown",
        "expected_current_zone": None,
        "expected_last_ms": 23600,
    }
    answer = {
        "states": [
            {
                "object_id": "cup",
                "status": "unknown",
                "current_zone": None,
                "last_observed_ms": 23600,
            }
        ],
        "evidence": [{"run_id": "run", "video_id": "video", "at_ms": 23600}],
        "answer": "Current location is unconfirmed.",
        "intent": "location",
        "as_of_ms": 30000,
    }
    assert all(evaluation.check_answer(answer, case, dataset).values())
    wrong = copy.deepcopy(answer)
    wrong["states"][0].update(object_id="other", status="visible", current_zone="center")
    wrong["evidence"][0].update(run_id="other", at_ms=31000)
    checks = evaluation.check_answer(wrong, case, dataset)
    assert not checks["object"] and not checks["state"] and not checks["current_zone"]
    assert not checks["source_scope"] and not checks["causal_evidence"]
    assert not evaluation.check_answer(answer, dict(case, expected_resolution="none"), dataset)[
        "no_unsupported_claim"
    ]
