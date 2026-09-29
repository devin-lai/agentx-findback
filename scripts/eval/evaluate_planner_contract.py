"""Score frozen language/grounding cases against labels, with an identical local baseline."""

import argparse
import hashlib
import json
import platform
import shutil
import statistics
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from agentx.api.app import create_app
from agentx.config import Settings
from agentx.domain.contracts import IndexRequest, Question
from agentx.services.answering import AMBIGUOUS_MATCH, NO_MATCH
from agentx.services.demo import create_fixture
from agentx.storage.models import Video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=Path("tests/fixtures/planner_contract_v1.json")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/planner-contract.json"))
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--without-query-skills", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Preserve completed and partial reports: choose a new output path.")
    raw = args.cases.read_bytes()
    dataset = json.loads(raw)
    code = Path(__file__).resolve().parents[2] / "src" / "agentx"
    source_hashes = {
        str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(code.rglob("*.py"))
    }
    rows = []
    with tempfile.TemporaryDirectory(prefix="agentx-planner-contract-") as directory:
        root = Path(directory)
        app = create_app(Settings(data_dir=root / "data", database_url="", worker_enabled=False))
        s = app.state.services
        planner_calls: list[dict] = []
        planner_chat = s.providers.planner_chat

        def record_planner_call(messages, **kwargs):
            started = time.monotonic()
            receipt = {
                "messages_sha256": hashlib.sha256(
                    json.dumps(messages, sort_keys=True).encode()
                ).hexdigest(),
                "schema_requested": bool(kwargs.get("response_schema")),
            }
            try:
                payload = planner_chat(messages, **kwargs)
                # These are generated fixture questions. Keep raw completions so an
                # invalid or mistranslated predicate can be audited without rerunning.
                receipt["response"] = payload
                return payload
            except Exception as exc:
                receipt["error_type"] = type(exc).__name__
                raise
            finally:
                receipt["seconds"] = round(time.monotonic() - started, 4)
                planner_calls.append(receipt)

        s.providers.planner_chat = record_planner_call
        if not args.local_only and not s.settings.planner_available:
            raise SystemExit("Configure a planner or pass --local-only.")
        s.db.migrate()
        try:
            source = root / "fixture.mp4"
            registrations = create_fixture(source)
            customization = dataset.get("fixture", {})
            registrations = [
                obj.model_copy(update=customization.get("objects", {}).get(obj.name, {}))
                for obj in registrations
            ]
            incoming = root / "incoming.mp4"
            shutil.copyfile(source, incoming)
            video = s.catalog.import_video(incoming, "Planner contract fixture", is_fixture=True)
            if customization.get("regions"):
                with s.db.session.begin() as session:
                    stored = session.get(Video, video.id)
                    stored.regions = [
                        {**region, **customization["regions"].get(region["id"], {})}
                        for region in stored.regions
                    ]
            names = {o.name: s.catalog.register(video.id, o).id for o in registrations}
            run = s.indexer.enqueue(video.id, IndexRequest(backend="reference"))
            s.indexer.process(run.id)
            for number, case in enumerate(dataset["cases"]):
                for provider in [False] if args.local_only else [False, True]:
                    planner_calls.clear()
                    answer = s.workflow.answer(
                        run.id,
                        Question(
                            text=case["text"],
                            at_ms=case["at_ms"],
                            object_id=names.get(case.get("selected_object")),
                            intent=case.get("requested_intent"),
                            use_provider=provider,
                            use_skills=not args.without_query_skills,
                        ),
                    )
                    state = answer["states"][0] if len(answer["states"]) == 1 else None
                    checks = {
                        "object": (state["name"] if state else None) == case["expected_object"],
                        "intent": answer["intent"] == case["expected_intent"],
                        "causal_evidence": all(
                            e["at_ms"] <= case["at_ms"] for e in answer["evidence"]
                        ),
                        "source_scope": all(
                            e["video_id"] == video.id and e["run_id"] == run.id
                            for e in answer["evidence"]
                        ),
                        "object_scope": all(
                            s.memory.observation(e["observation_id"])[0].object_id
                            == names.get(case["expected_object"])
                            for e in answer["evidence"]
                        ),
                    }
                    if case.get("expected_resolution"):
                        resolution = (
                            "unique"
                            if state
                            else "ambiguous"
                            if answer["answer"] == AMBIGUOUS_MATCH
                            else "none"
                            if answer["answer"] == NO_MATCH
                            else "unknown"
                        )
                        checks["resolution"] = resolution == case["expected_resolution"]
                    if case.get("expected_status"):
                        checks["status"] = bool(
                            state and state["status"] == case["expected_status"]
                        )
                    if case.get("expected_zone"):
                        checks["zone"] = bool(state and state["zone"] == case["expected_zone"])
                    if case["expected_object"]:
                        checks["has_evidence"] = bool(answer["evidence"])
                        canonical = s.workflow.local(
                            run.id,
                            Question(
                                text=case["text"],
                                object_id=names[case["expected_object"]],
                                intent=case["expected_intent"],
                            ),
                            *s.memory.context(run.id, case["at_ms"]),
                        )
                        checks["grounded_text"] = answer["answer"] == canonical["answer"]
                    else:
                        checks["no_unsupported_claim"] = (
                            not answer["states"] and not answer["evidence"]
                        )
                    row = {
                        "number": number,
                        "case": case,
                        "provider_requested": provider,
                        "passed": all(checks.values()),
                        "checks": checks,
                        "answer": answer,
                        "planner_calls": list(planner_calls),
                    }
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                "number": number,
                                "provider": provider,
                                "passed": row["passed"],
                                "failed": [k for k, v in checks.items() if not v],
                                "seconds": answer["elapsed_seconds"],
                            }
                        ),
                        flush=True,
                    )
                    # Checkpoint after every answer so interrupted GPU runs retain evidence.
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps({"status": "running", "rows": rows}, indent=2) + "\n"
                    )
        finally:
            s.db.engine.dispose()
    summaries = {}
    for provider in [False] if args.local_only else [False, True]:
        subset = [r for r in rows if r["provider_requested"] == provider]
        summaries["agent" if provider else "local"] = {
            "cases": len(subset),
            "passed": sum(r["passed"] for r in subset),
            "actual_agent_answers": sum(r["answer"]["planner"] == "agent" for r in subset),
            "fallbacks": sum(provider and r["answer"]["planner"] != "agent" for r in subset),
            "median_seconds": statistics.median(r["answer"]["elapsed_seconds"] for r in subset),
            "max_seconds": max(r["answer"]["elapsed_seconds"] for r in subset),
        }
    report = {
        "status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "cases_sha256": hashlib.sha256(raw).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "query_skills": not args.without_query_skills,
        "planner_model": s.settings.planner_endpoint[2],
        "planner_config": {
            "max_steps": s.settings.planner_max_steps,
            "max_tokens": s.settings.planner_max_tokens,
            "thinking": s.settings.planner_thinking,
            "structured_outputs": s.settings.planner_structured_outputs,
            "selection_thinking_budget": getattr(
                s.settings, "planner_selection_thinking_budget", None
            ),
        },
        "source_hashes": source_hashes,
        "summary": summaries,
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
    return 0 if all(r["passed"] for r in rows if r["provider_requested"] or args.local_only) else 1


if __name__ == "__main__":
    raise SystemExit(main())
