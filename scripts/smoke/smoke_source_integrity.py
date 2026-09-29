"""Verify live source checks and unchanged real-footage memory without invoking models."""

import argparse
import hashlib
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from agentx.api.app import create_app
from agentx.config import Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    previous = json.loads(args.receipt.read_text())
    if previous.get("status") != "complete":
        raise SystemExit("Use a complete live receipt.")
    settings = Settings(worker_enabled=False)
    app = create_app(settings)
    services = app.state.services
    headers = {"Authorization": f"Bearer {settings.api_token}"} if settings.api_token else {}
    checks = []
    with httpx.Client(base_url="http://127.0.0.1:9000", headers=headers, timeout=120) as client:
        for row in previous["answers"]:
            original = row["answer"]
            response = client.post(
                f"/api/v1/runs/{original['run_id']}/questions",
                json={
                    "text": original["question"],
                    "at_ms": original["as_of_ms"],
                    "object_id": original["states"][0]["object_id"],
                    "use_provider": False,
                },
            )
            response.raise_for_status()
            answer = response.json()
            same = {key: answer[key] == original[key] for key in ("answer", "states", "evidence")}
            frames = []
            for ref in answer["evidence"]:
                frame = client.get(ref["frame_url"])
                frame.raise_for_status()
                frames.append(
                    {
                        "at_ms": ref["at_ms"],
                        "timestamp_matches": int(frame.headers["x-frame-time-ms"]) == ref["at_ms"],
                        "cache_revalidation": frame.headers.get("cache-control") == "no-store",
                        "sha256": hashlib.sha256(frame.content).hexdigest(),
                    }
                )
            checks.append({"name": row["name"], "same": same, "frames": frames})
    _, video, _, _ = services.memory.context(previous["answers"][0]["answer"]["run_id"], None)
    durations = []
    for _ in range(101):
        start = time.monotonic()
        source = services.catalog.require_source(video)
        durations.append(time.monotonic() - start)
    services.db.engine.dispose()
    passed = all(
        all(row["same"].values())
        and all(
            frame["timestamp_matches"] and frame["cache_revalidation"] for frame in row["frames"]
        )
        for row in checks
    )
    result = {
        "status": "complete" if passed else "failed",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "Existing real-footage memory and source integrity; no new tracking or planner accuracy claim.",
        "checks": checks,
        "source_hash_check": {
            "bytes": source.stat().st_size,
            "sha256": video.sha256,
            "first_seconds": durations[0],
            "cached_median_seconds": statistics.median(durations[1:]),
            "cached_calls": 100,
            "note": "In-process cached filesystem reads on this existing recording, not an end-to-end API latency benchmark.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
