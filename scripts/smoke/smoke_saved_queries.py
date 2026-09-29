"""Re-ask saved public-fixture questions through a live app after a planner service change."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from agentx.config import Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:9000")
    args = parser.parse_args()
    raw = args.receipt.read_bytes()
    previous = json.loads(raw)
    if previous.get("status") != "complete":
        raise SystemExit("Use a completed live receipt.")
    settings = Settings()
    headers = {"Authorization": f"Bearer {settings.api_token}"} if settings.api_token else {}
    rows = []
    with httpx.Client(base_url=args.base_url, headers=headers, timeout=120) as client:
        for item in previous["answers"]:
            original = item["answer"]
            response = client.post(
                f"/api/v1/runs/{original['run_id']}/questions",
                json={
                    "text": original["question"],
                    "at_ms": original["as_of_ms"],
                    "object_id": original["states"][0]["object_id"],
                    "use_provider": True,
                },
            )
            response.raise_for_status()
            answer = response.json()
            checks = {
                "actual_agent": answer["planner"] == "agent",
                "same_model": answer["planner_model"] == original["planner_model"],
                "same_answer": answer["answer"] == original["answer"],
                "same_states": answer["states"] == original["states"],
                "same_evidence": answer["evidence"] == original["evidence"],
                "no_fallback_warning": not any(
                    "fallback" in w or "local query" in w for w in answer["warnings"]
                ),
            }
            rows.append({"name": item["name"], "checks": checks, "answer": answer})
            print(json.dumps({"name": item["name"], "checks": checks}), flush=True)
    report = {
        "status": "complete" if all(all(row["checks"].values()) for row in rows) else "failed",
        "created_at": datetime.now(UTC).isoformat(),
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "scope": "Replay existing saved development questions; no new tracking evaluation or user trial.",
        "answers": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
