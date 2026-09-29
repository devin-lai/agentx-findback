"""Isolate the Agent Skills contribution with three arms over identical frozen cases.

Every case is answered three ways against the same fixture, memory run, tools, model and
step budget, so the only difference between the two planner arms is whether the versioned
SKILL.md bodies are present in the prompt:

  local            deterministic resolver, no planner request (capability floor)
  agent_no_skills  Nemotron tool agent, skill bodies withheld
  agent_skills     Nemotron tool agent, skill bodies loaded

The arms are paired by case, so the planner arms are compared with an exact McNemar test
rather than by two independent pass rates. Scoring reuses the checks in
`evaluate_planner_contract.py`; this harness adds arms, token accounting and the paired
statistic. It measures agreement with frozen labels on generated fixtures, not real-world
accuracy, and a null or negative Skills effect is a valid result that must be reported.
"""

import argparse
import hashlib
import json
import math
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

ARMS = ("local", "agent_no_skills", "agent_skills")


def mcnemar_exact(discordant_b: int, discordant_c: int) -> dict:
    """Two-sided exact McNemar test on the paired disagreements between two arms.

    `discordant_b` counts cases the second arm passes and the first fails; `discordant_c`
    counts the reverse. Concordant cases carry no information about the difference and are
    excluded by construction, which is exactly why the paired test is the honest one here.
    """
    n = discordant_b + discordant_c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p_value": 1.0, "note": "no disagreement"}
    tail = min(discordant_b, discordant_c)
    cumulative = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return {
        "b": discordant_b,
        "c": discordant_c,
        "n_discordant": n,
        "p_value": round(min(1.0, 2 * cumulative), 6),
        "note": "two-sided exact binomial on discordant pairs",
    }


def score(case: dict, answer: dict, services, video_id: str, run_id: str, names: dict) -> dict:
    """The frozen contract checks, identical for every arm."""
    state = answer["states"][0] if len(answer["states"]) == 1 else None
    checks = {
        "object": (state["name"] if state else None) == case["expected_object"],
        "intent": answer["intent"] == case["expected_intent"],
        "causal_evidence": all(e["at_ms"] <= case["at_ms"] for e in answer["evidence"]),
        "source_scope": all(
            e["video_id"] == video_id and e["run_id"] == run_id for e in answer["evidence"]
        ),
        "object_scope": all(
            services.memory.observation(e["observation_id"])[0].object_id
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
        checks["status"] = bool(state and state["status"] == case["expected_status"])
    if case.get("expected_zone"):
        checks["zone"] = bool(state and state["zone"] == case["expected_zone"])
    if case["expected_object"]:
        checks["has_evidence"] = bool(answer["evidence"])
    else:
        checks["no_unsupported_claim"] = not answer["states"] and not answer["evidence"]
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        nargs="+",
        required=True,
        help="One or more frozen case files; each is evaluated on its own fixture.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=list(ARMS),
        choices=list(ARMS),
        help="Subset of arms to run; the default runs all three.",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Evaluate only the first N cases of each suite."
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Preserve completed and partial reports: choose a new output path.")

    code = Path(__file__).resolve().parents[2] / "src" / "agentx"
    source_hashes = {
        str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(code.rglob("*.py"))
    }
    skills_root = code / "skills"
    skill_hashes = {
        p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(skills_root.glob("*/SKILL.md"))
    }

    rows: list[dict] = []
    suites: list[dict] = []
    planner_model = None
    planner_config: dict = {}

    for cases_path in args.cases:
        raw = cases_path.read_bytes()
        dataset = json.loads(raw)
        cases = dataset["cases"][: args.limit] if args.limit else dataset["cases"]
        suites.append(
            {
                "file": str(cases_path),
                "name": dataset.get("name", cases_path.stem),
                "scope": dataset.get("scope", ""),
                "cases_sha256": hashlib.sha256(raw).hexdigest(),
                "case_count": len(cases),
            }
        )
        with tempfile.TemporaryDirectory(prefix="agentx-skill-ablation-") as directory:
            root = Path(directory)
            app = create_app(
                Settings(data_dir=root / "data", database_url="", worker_enabled=False)
            )
            s = app.state.services
            if "agent_no_skills" in args.arms or "agent_skills" in args.arms:
                if not s.settings.planner_available:
                    raise SystemExit("Configure a planner, or pass --arms local.")
            planner_model = s.settings.planner_endpoint[2]
            planner_config = {
                "max_steps": s.settings.planner_max_steps,
                "max_tokens": s.settings.planner_max_tokens,
                "thinking": s.settings.planner_thinking,
                "structured_outputs": s.settings.planner_structured_outputs,
                "selection_thinking_budget": s.settings.planner_selection_thinking_budget,
            }

            usage: list[dict] = []
            s.providers.usage_sink = lambda model, payload, sink=usage: sink.append(payload)

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
                video = s.catalog.import_video(incoming, "Skill ablation fixture", is_fixture=True)
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

                for number, case in enumerate(cases):
                    for arm in args.arms:
                        usage.clear()
                        question = Question(
                            text=case["text"],
                            at_ms=case["at_ms"],
                            object_id=names.get(case.get("selected_object")),
                            intent=case.get("requested_intent"),
                            use_provider=arm != "local",
                            use_skills=arm == "agent_skills",
                        )
                        started = time.monotonic()
                        answer = s.workflow.answer(run.id, question)
                        elapsed = round(time.monotonic() - started, 3)
                        checks = score(case, answer, s, video.id, run.id, names)
                        state = answer["states"][0] if len(answer["states"]) == 1 else None
                        rows.append(
                            {
                                "suite": cases_path.stem,
                                "number": number,
                                "arm": arm,
                                "case": case,
                                "passed": all(checks.values()),
                                "checks": checks,
                                # Named an object the labels do not expect: the failure that
                                # produces a confident wrong answer for a user.
                                "wrong_object": bool(
                                    state and state["name"] != case["expected_object"]
                                ),
                                "planner_used": answer["planner"],
                                "fell_back": arm != "local" and answer["planner"] != "agent",
                                "tool_calls": [t["name"] for t in answer["tools"]],
                                "skills_loaded": [t["name"] for t in answer["skills"]],
                                "warnings": answer["warnings"],
                                "seconds": elapsed,
                                "planner_requests": len(usage),
                                "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
                                "completion_tokens": sum(
                                    u.get("completion_tokens", 0) for u in usage
                                ),
                                "answer": answer,
                            }
                        )
                        print(
                            json.dumps(
                                {
                                    "suite": cases_path.stem,
                                    "number": number,
                                    "arm": arm,
                                    "passed": rows[-1]["passed"],
                                    "failed": [k for k, v in checks.items() if not v],
                                    "seconds": elapsed,
                                }
                            ),
                            flush=True,
                        )
                        # Checkpoint after every answer so an interrupted GPU run keeps evidence.
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(
                            json.dumps({"status": "running", "rows": rows}, indent=2) + "\n"
                        )
            finally:
                s.providers.usage_sink = None
                s.db.engine.dispose()

    def arm_rows(arm: str) -> list[dict]:
        return [r for r in rows if r["arm"] == arm]

    summary = {}
    for arm in args.arms:
        subset = arm_rows(arm)
        if not subset:
            continue
        summary[arm] = {
            "cases": len(subset),
            "passed": sum(r["passed"] for r in subset),
            "pass_rate": round(sum(r["passed"] for r in subset) / len(subset), 4),
            "wrong_object": sum(r["wrong_object"] for r in subset),
            "fallbacks": sum(r["fell_back"] for r in subset),
            "median_seconds": statistics.median(r["seconds"] for r in subset),
            "p95_seconds": round(
                sorted(r["seconds"] for r in subset)[max(0, math.ceil(0.95 * len(subset)) - 1)], 3
            ),
            "total_prompt_tokens": sum(r["prompt_tokens"] for r in subset),
            "total_completion_tokens": sum(r["completion_tokens"] for r in subset),
            "planner_requests": sum(r["planner_requests"] for r in subset),
        }

    per_suite = {}
    for suite in {r["suite"] for r in rows}:
        per_suite[suite] = {
            arm: {
                "cases": len([r for r in arm_rows(arm) if r["suite"] == suite]),
                "passed": sum(r["passed"] for r in arm_rows(arm) if r["suite"] == suite),
                "wrong_object": sum(
                    r["wrong_object"] for r in arm_rows(arm) if r["suite"] == suite
                ),
            }
            for arm in args.arms
            if any(r["suite"] == suite for r in arm_rows(arm))
        }

    paired = {}
    if "agent_no_skills" in args.arms and "agent_skills" in args.arms:
        keyed = {(r["suite"], r["number"]): r for r in arm_rows("agent_no_skills")}
        b = c = 0
        flipped = []
        for row in arm_rows("agent_skills"):
            other = keyed.get((row["suite"], row["number"]))
            if other is None or row["passed"] == other["passed"]:
                continue
            if row["passed"]:
                b += 1
            else:
                c += 1
            flipped.append(
                {
                    "suite": row["suite"],
                    "number": row["number"],
                    "text": row["case"]["text"],
                    "skills_passed": row["passed"],
                    "skills_failed_checks": [k for k, v in row["checks"].items() if not v],
                    "no_skills_failed_checks": [k for k, v in other["checks"].items() if not v],
                }
            )
        paired["agent_skills_vs_agent_no_skills"] = {
            **mcnemar_exact(b, c),
            "b_meaning": "skills pass, no-skills fail",
            "c_meaning": "skills fail, no-skills pass",
            "discordant_cases": flipped,
        }

    report = {
        "status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "design": (
            "Three arms over identical frozen cases, one shared fixture and memory run per "
            "suite. Only the presence of the SKILL.md bodies differs between the planner "
            "arms. The local arm never issues a planner request; skill traces are recorded "
            "for provenance but the deterministic resolver's behavior does not depend on "
            "them. Labels are frozen fixture annotations, not real-world ground truth."
        ),
        "arms": list(args.arms),
        "suites": suites,
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "planner_model": planner_model,
        "planner_config": planner_config,
        "skill_sha256": skill_hashes,
        "source_hashes": source_hashes,
        "summary": summary,
        "per_suite": per_suite,
        "paired": paired,
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    test = paired.get("agent_skills_vs_agent_no_skills", {})
    print(
        json.dumps(
            {
                "summary": summary,
                "paired": {k: v for k, v in test.items() if k != "discordant_cases"},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
