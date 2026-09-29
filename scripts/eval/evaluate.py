"""Evaluate explicit ground truth, never self-grade generated prose.

Run without --manifest for the controlled fixture. Real recordings require an
independently annotated manifest; see docs/EVALUATION.md.
"""

import argparse
import json
import platform
import shutil
import statistics
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from agentx.api.app import create_app
from agentx.config import Settings
from agentx.domain.contracts import IndexRequest, Question, RegisterObject
from agentx.services.demo import create_fixture
from agentx.storage.models import IndexRun


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    at_ms: int = Field(ge=0)
    expected_status: str | None = None
    expected_object: str | None = None
    expected_zone: str | None = None
    expected_intent: str = "location"
    expected_last_seen_ms: int | None = None
    tolerance_ms: int = 250
    expect_match: bool = True


class Clip(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    objects: list[RegisterObject]
    cases: list[Case] = Field(min_length=1)


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    provenance: str
    # How the clip was filmed, when the generator scripted a camera. Recorded for the report;
    # it never changes what is evaluated.
    camera: dict | None = None
    clips: list[Clip] = Field(min_length=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    parser.add_argument(
        "--backend", choices=["reference", "rtdetr", "cosmos", "sam2"], default="reference"
    )
    parser.add_argument(
        "--scene",
        choices=["fixed", "bumped"],
        default="fixed",
        help="Which controlled sample to evaluate when no manifest is given.",
    )
    parser.add_argument(
        "--use-provider",
        action="store_true",
        help="Use the configured planner agent; record fallbacks.",
    )
    parser.add_argument(
        "--without-query-skills",
        action="store_true",
        help="Ablate query prompts only; invariants and perception stay unchanged.",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="agentx-evaluation-") as directory:
        root = Path(directory)
        fixture = args.manifest is None
        if fixture:
            source = root / "fixture.mp4"
            objects = create_fixture(source, args.scene)
            if args.scene == "bumped":
                # The camera is knocked at six seconds. Nothing on the desk moves afterwards, so
                # every expectation past the knock repeats the region from before it.
                cases = [
                    Case(
                        question="Where is Red toolkit?",
                        at_ms=t,
                        expected_status="visible",
                        expected_zone=zone,
                    )
                    for t, zone in [
                        (2000, "left"),
                        (5000, "right"),
                        (8000, "right"),
                        (11000, "right"),
                    ]
                ]
                cases += [
                    Case(
                        question="Where is Blue remote?",
                        at_ms=t,
                        expected_status="visible",
                        expected_zone="center",
                    )
                    for t in (2000, 11000)
                ]
                cases += [
                    Case(question="Where is my passport?", at_ms=5000, expect_match=False),
                    Case(
                        question="Show the history of Red toolkit",
                        at_ms=11000,
                        expected_status="visible",
                        expected_zone="right",
                        expected_intent="history",
                    ),
                ]
            else:
                cases = [
                    Case(
                        question="Where is Red toolkit?",
                        at_ms=t,
                        expected_status=status,
                        expected_zone=zone,
                    )
                    for t, status, zone in [
                        (2000, "visible", "left"),
                        (5000, "visible", "right"),
                        (7000, "last_seen", "right"),
                        (11000, "visible", "center"),
                    ]
                ]
                cases[2].expected_last_seen_ms = 5800
                cases += [
                    Case(question="Where is my passport?", at_ms=5000, expect_match=False),
                    Case(
                        question="Show the history of Red toolkit",
                        at_ms=2000,
                        expected_status="visible",
                        expected_zone="left",
                        expected_intent="history",
                    ),
                ]
            dataset = Manifest(
                name=f"Controlled temporal fixture ({args.scene} camera)",
                provenance="Generated test imagery; not a real-world model benchmark.",
                camera={
                    "scene": args.scene,
                    **({"knocked_at_ms": 6000} if args.scene == "bumped" else {}),
                },
                clips=[Clip(path=str(source), objects=objects, cases=cases)],
            )
            base = root
        else:
            dataset = Manifest.model_validate_json(args.manifest.read_text())
            base = args.manifest.resolve().parent
        app = create_app(Settings(data_dir=root / "data", database_url="", worker_enabled=False))
        s = app.state.services
        if args.use_provider and not s.settings.planner_available:
            raise SystemExit(
                "--use-provider requires AGENTX_PLANNER_BASE_URL and AGENTX_PLANNER_MODEL."
            )
        s.db.migrate()
        rows = []
        runs = []
        try:
            for number, clip in enumerate(dataset.clips):
                incoming = root / f"incoming-{number}.mp4"
                shutil.copyfile((base / clip.path).resolve(), incoming)
                video = s.catalog.import_video(incoming, Path(clip.path).name, is_fixture=fixture)
                for obj in clip.objects:
                    s.catalog.register(video.id, obj)
                run = s.indexer.enqueue(video.id, IndexRequest(backend=args.backend))
                s.indexer.process(run.id)
                with s.db.session() as session:
                    run = session.get(IndexRun, run.id)
                if run.status != "complete":
                    raise RuntimeError(f"Index failed: {run.error}")
                runs.append(
                    {
                        "input_sha256": video.sha256,
                        "duration_ms": video.duration_ms,
                        "index_seconds": run.elapsed_seconds,
                        "observations": run.observation_count,
                        "provenance": run.provenance,
                    }
                )
                for case in clip.cases:
                    answer = s.workflow.answer(
                        run.id,
                        Question(
                            text=case.question,
                            at_ms=case.at_ms,
                            use_provider=args.use_provider,
                            use_skills=not args.without_query_skills,
                        ),
                    )
                    state = answer["states"][0] if answer["states"] else None
                    checks = {
                        "match": bool(state) == case.expect_match,
                        "intent": answer["intent"] == case.expected_intent,
                        "causal_evidence": all(
                            e["at_ms"] <= case.at_ms for e in answer["evidence"]
                        ),
                        "source_scope": all(
                            e["video_id"] == video.id and e["run_id"] == run.id
                            for e in answer["evidence"]
                        ),
                        "causal_events": all(e["at_ms"] <= case.at_ms for e in answer["events"]),
                    }
                    if case.expected_status is not None:
                        checks["status"] = bool(state and state["status"] == case.expected_status)
                    if case.expected_object is not None:
                        checks["object"] = bool(state and state["name"] == case.expected_object)
                    if case.expected_zone is not None:
                        checks["zone"] = bool(state and state["zone"] == case.expected_zone)
                    if case.expected_last_seen_ms is not None:
                        checks["last_seen_time"] = bool(
                            state
                            and state["last_observed_ms"] is not None
                            and abs(state["last_observed_ms"] - case.expected_last_seen_ms)
                            <= case.tolerance_ms
                        )
                    if case.expect_match and case.expected_status in {"visible", "last_seen"}:
                        checks["has_evidence"] = bool(answer["evidence"])
                    rows.append(
                        {
                            "case": case.model_dump(),
                            "passed": all(checks.values()),
                            "checks": checks,
                            "answer": answer,
                        }
                    )
        finally:
            s.db.engine.dispose()
        latencies = [r["answer"]["elapsed_seconds"] for r in rows]
        report = {
            "schema_version": "1",
            "created_at": datetime.now(UTC).isoformat(),
            "dataset": dataset.name,
            "dataset_provenance": dataset.provenance,
            "controlled_fixture": fixture,
            "host": {"platform": platform.platform(), "python": platform.python_version()},
            "query_skills": not args.without_query_skills,
            "provider_requested": args.use_provider,
            "summary": {
                "cases": len(rows),
                "passed": sum(r["passed"] for r in rows),
                "accuracy": sum(r["passed"] for r in rows) / len(rows),
                "mean_query_seconds": statistics.mean(latencies),
                "max_query_seconds": max(latencies),
                "provider_fallbacks": sum(
                    args.use_provider and r["answer"]["planner"] != "agent" for r in rows
                ),
                "cases_with_warnings": sum(bool(r["answer"]["warnings"]) for r in rows),
            },
            "runs": runs,
            "results": rows,
            "note": "Evaluation uses temporary media. Evidence IDs in this report describe the run but are not a persistent replay URL. Export a retained workspace run for the demo.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {"report": str(args.output), "fixture": fixture, **report["summary"]}, indent=2
            )
        )
        return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
