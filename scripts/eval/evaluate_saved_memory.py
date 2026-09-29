"""Evaluate frozen query labels on a read-only snapshot of an existing SQLite memory.

This tests language and evidence contracts over saved observations. It does not
re-score perception, create footage, or establish real-world/user accuracy.
"""

import argparse
import hashlib
import json
import shutil
import sqlite3
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from agentx.api.app import create_app
from agentx.config import Settings
from agentx.domain.contracts import Question
from agentx.services.answering import AMBIGUOUS_MATCH, NO_MATCH


def snapshot_database(source: Path, destination: Path) -> str:
    if not source.is_file() or destination.exists():
        raise ValueError("Use an existing database and a new snapshot destination.")
    with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as original:
        with sqlite3.connect(destination) as snapshot:
            original.backup(snapshot)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def check_answer(answer: dict, case: dict, dataset: dict) -> dict[str, bool]:
    states, evidence = answer["states"], answer["evidence"]
    state = states[0] if len(states) == 1 else None
    expected_id = dataset["object_id"] if case["expected_resolution"] == "unique" else None
    resolution = (
        "unique"
        if state
        else "ambiguous"
        if answer["answer"] == AMBIGUOUS_MATCH
        else "none"
        if answer["answer"] == NO_MATCH
        else "unknown"
    )
    checks = {
        "object": [s["object_id"] for s in states] == ([expected_id] if expected_id else []),
        "resolution": resolution == case["expected_resolution"],
        "intent": answer["intent"] == case["expected_intent"],
        "cutoff": answer["as_of_ms"] == case["at_ms"],
        "causal_evidence": all(e["at_ms"] <= case["at_ms"] for e in evidence),
        "source_scope": all(
            e["video_id"] == dataset["video_id"] and e["run_id"] == dataset["run_id"]
            for e in evidence
        ),
    }
    if expected_id:
        checks.update(
            state=bool(state and state["status"] == case["expected_status"]),
            current_zone=bool(state and state["current_zone"] == case["expected_current_zone"]),
            last_observed=bool(state and state["last_observed_ms"] == case["expected_last_ms"]),
            has_evidence=bool(evidence),
        )
    else:
        checks["no_unsupported_claim"] = not states and not evidence
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing report.")
    raw = args.cases.read_bytes()
    dataset = json.loads(raw)
    source = Path(__file__).resolve().parents[2] / "src"
    report = {
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": __doc__,
        "dataset": dataset,
        "cases_sha256": hashlib.sha256(raw).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.rglob("*.py"))
        },
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    with tempfile.TemporaryDirectory(prefix="agentx-saved-memory-") as directory:
        root = Path(directory)
        report["database_snapshot_sha256"] = snapshot_database(args.database, root / "memory.db")
        settings = Settings(
            data_dir=root / "data",
            database_url=f"sqlite:///{root / 'memory.db'}",
            worker_enabled=False,
        )
        if not args.local_only and not settings.planner_available:
            raise SystemExit("Configure a planner or use --local-only.")
        s = create_app(settings).state.services
        s.db.migrate()
        report["planner_model"] = settings.planner_endpoint[2]
        report["selection_thinking_budget"] = settings.planner_selection_thinking_budget
        calls = []
        original_chat = s.providers.planner_chat

        def record_call(messages, **kwargs):
            receipt = {
                "messages_sha256": hashlib.sha256(
                    json.dumps(messages, sort_keys=True).encode()
                ).hexdigest(),
                "schema_requested": bool(kwargs.get("response_schema")),
            }
            started = time.monotonic()
            try:
                result = original_chat(messages, **kwargs)
                receipt["response"] = result
                return result
            except Exception as exc:
                receipt["error_type"] = type(exc).__name__
                raise
            finally:
                receipt["seconds"] = round(time.monotonic() - started, 4)
                calls.append(receipt)

        s.providers.planner_chat = record_call
        try:
            run, video, objects, _ = s.memory.context(dataset["run_id"], 0)
            if video.id != dataset["video_id"] or [o.id for o in objects] != [dataset["object_id"]]:
                raise ValueError(
                    "Saved run or single-object inventory does not match the protocol."
                )
            original_data = args.data_dir.resolve()
            for relative in {
                video.source_path,
                video.media_path,
                *(o.reference_path for o in objects),
            }:
                original = (original_data / relative).resolve()
                if not original.is_relative_to(original_data):
                    raise ValueError("Stored media escapes the supplied data directory.")
                destination = s.catalog.path(relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, destination)
            s.catalog.require_source(video)
            report["source_sha256"] = video.sha256
            report["run_provenance"] = run.provenance
            for case in dataset["cases"]:
                if case["expected_resolution"] == "unique":
                    state = s.memory.states(run.id, case["at_ms"])[0]
                    if (state.status.value, state.current_zone, state.last_observed_ms) != (
                        case["expected_status"],
                        case["expected_current_zone"],
                        case["expected_last_ms"],
                    ):
                        raise ValueError(
                            "Saved observations differ from the frozen expected memory."
                        )
            for number, case in enumerate(dataset["cases"]):
                for provider in [False] if args.local_only else [False, True]:
                    calls.clear()
                    answer = s.workflow.answer(
                        run.id,
                        Question(text=case["text"], at_ms=case["at_ms"], use_provider=provider),
                    )
                    checks = check_answer(answer, case, dataset)
                    checks["object_scope"] = all(
                        s.memory.observation(e["observation_id"])[0].object_id
                        == dataset["object_id"]
                        for e in answer["evidence"]
                    )
                    if case["expected_resolution"] == "unique":
                        canonical = s.workflow.local(
                            run.id,
                            Question(
                                text=case["text"],
                                object_id=dataset["object_id"],
                                intent=case["expected_intent"],
                            ),
                            *s.memory.context(run.id, case["at_ms"]),
                        )
                        checks["grounded_text"] = answer["answer"] == canonical["answer"]
                    row = {
                        "number": number,
                        "case": case,
                        "provider_requested": provider,
                        "passed": all(checks.values()),
                        "checks": checks,
                        "answer": answer,
                        "planner_calls": list(calls),
                    }
                    report["rows"].append(row)
                    save()
                    print(
                        json.dumps(
                            {
                                "number": number,
                                "provider": provider,
                                "passed": row["passed"],
                                "failed": [key for key, value in checks.items() if not value],
                            }
                        ),
                        flush=True,
                    )
        finally:
            s.db.engine.dispose()
    rows = [r for r in report["rows"] if r["provider_requested"] or args.local_only]
    report["status"] = "complete"
    report["summary"] = {
        "cases": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "actual_agent_passed": sum(r["passed"] and r["answer"]["planner"] == "agent" for r in rows),
        "fallbacks": sum(
            r["provider_requested"] and r["answer"]["planner"] != "agent" for r in rows
        ),
        "wrong_selections": sum(
            bool(r["answer"]["states"]) and not r["checks"]["object"] for r in rows
        ),
    }
    save()
    print(json.dumps(report["summary"]))
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
