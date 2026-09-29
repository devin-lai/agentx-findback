"""Ask a frozen question set through a running API with and without the planner agent.

Reports, per question: planner actually used, intent, cited evidence times, whether the agent's
evidence equals the deterministic resolver's evidence at the same cutoff, warnings and latency.
It measures agreement and validity, not truth; truth needs an annotated recording.
"""

import argparse
import json
import os
import time
from pathlib import Path

import httpx

DEFAULT_QUESTIONS = [
    {"text": "Where was {a} last seen?", "at_ms": 7000},
    {"text": "Where is {a} right now?", "at_ms": 4000},
    {"text": "What happened to {a} before 9 seconds?", "at_ms": 9000},
    {"text": "Show the history of {b}.", "at_ms": 11000},
    {"text": "Where is my passport?", "at_ms": 5000},
    {"text": "Is {a} still where it was?", "at_ms": 7000},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9000")
    parser.add_argument("--token", default=os.environ.get("AGENTX_API_TOKEN", ""))
    parser.add_argument("--run-id", help="Completed run to question; default: create the fixture")
    parser.add_argument("--questions", type=Path, help="JSON list of {text, at_ms}; {a}/{b} expand")
    parser.add_argument("--output", type=Path, default=Path("artifacts/planner-comparison.json"))
    args = parser.parse_args()
    client = httpx.Client(
        base_url=args.base_url,
        headers={"Authorization": f"Bearer {args.token}"} if args.token else {},
        timeout=600,
    )
    capabilities = client.get("/api/v1/capabilities").json()
    if not capabilities.get("planner"):
        raise SystemExit("No planner is configured on that server; nothing to compare.")
    run_id = args.run_id
    if run_id is None:
        video = client.post("/api/v1/demo").json()
        run = client.post(
            f"/api/v1/videos/{video['id']}/runs", json={"backend": "reference"}
        ).json()
        while run["status"] in {"queued", "running"}:
            time.sleep(0.5)
            run = client.get(f"/api/v1/runs/{run['id']}").json()
        run_id = run["id"]
    run = client.get(f"/api/v1/runs/{run_id}").json()
    video = client.get(f"/api/v1/videos/{run['video_id']}").json()
    names = [o["name"] for o in video["objects"]]
    a, b = names[0], names[-1]
    questions = json.loads(args.questions.read_text()) if args.questions else DEFAULT_QUESTIONS
    rows = []
    for q in questions:
        text = q["text"].format(a=a, b=b)
        pair = {}
        for use_provider in (True, False):
            started = time.monotonic()
            answer = client.post(
                f"/api/v1/runs/{run_id}/questions",
                json={"text": text, "at_ms": q["at_ms"], "use_provider": use_provider},
            ).json()
            pair["agent" if use_provider else "local"] = {
                "planner": answer.get("planner"),
                "intent": answer.get("intent"),
                "answer": answer.get("answer"),
                "evidence": sorted((e["at_ms"], e["zone"]) for e in answer.get("evidence", [])),
                "states": {s["name"]: s["status"] for s in answer.get("states", [])},
                "tools": [t["name"] for t in answer.get("tools", []) if t["name"] != "load_skill"],
                "warnings": answer.get("warnings", []),
                "seconds": round(time.monotonic() - started, 3),
            }
        agent, local = pair["agent"], pair["local"]
        rows.append(
            {
                "question": text,
                "at_ms": q["at_ms"],
                **pair,
                "agent_used": agent["planner"] == "agent",
                "intent_agrees": agent["intent"] == local["intent"],
                "evidence_agrees": agent["evidence"] == local["evidence"],
                "agent_evidence_subset": set(agent["evidence"]) <= set(local["evidence"])
                or not local["evidence"],
            }
        )
        print(
            f"{'agent' if rows[-1]['agent_used'] else 'FALLBACK':8} {agent['seconds']:6.1f}s "
            f"intent={'=' if rows[-1]['intent_agrees'] else '≠'} evidence={'=' if rows[-1]['evidence_agrees'] else '≠'} | {text}"
        )
    report = {
        "scope": "Agreement between the planner agent and the deterministic resolver at identical cutoffs; not ground truth.",
        "planner_model": capabilities.get("planner_model"),
        "run_id": run_id,
        "questions": len(rows),
        "agent_used": sum(r["agent_used"] for r in rows),
        "intent_agreement": sum(r["intent_agrees"] for r in rows),
        "evidence_agreement": sum(r["evidence_agrees"] for r in rows),
        "mean_agent_seconds": round(sum(r["agent"]["seconds"] for r in rows) / len(rows), 3),
        "mean_local_seconds": round(sum(r["local"]["seconds"] for r in rows) / len(rows), 3),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
