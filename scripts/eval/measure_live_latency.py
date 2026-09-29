"""Warm live-query latency on the node, across both controlled samples.

Every request goes through the running application over loopback: the actual planner agent, the
real memory, the real evidence validation. Failures stay in the report. This is a warm,
single-client measurement with no concurrent experiment; it is not a capacity or cold-start
result.
"""

import argparse
import json
import math
import statistics
import time
from datetime import UTC, datetime

import httpx

QUESTIONS = [
    ("Where is Blue remote?", "location"),
    ("Where is Red toolkit?", "location"),
    ("Where was Blue remote last seen?", "last_seen"),
    ("Show the history of Red toolkit", "history"),
    ("Where is the object that is no longer visible?", "indirect"),
    ("Where is my passport?", "no_match"),
]
CUTOFFS = [4000, 7000, 11000]


def nearest_rank_p95(values):
    if not values:
        raise ValueError("No successful query timings.")
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def build(client, scene):
    video = client.post("/v1/demo", params={"scene": scene}).json()
    run = client.post(f"/v1/videos/{video['id']}/runs", json={}).json()
    started = time.monotonic()
    while True:
        state = client.get(f"/v1/runs/{run['id']}").json()
        if state["status"] in {"complete", "failed", "cancelled"}:
            break
        if time.monotonic() - started > 300:
            raise SystemExit(f"{scene}: indexing did not finish")
        time.sleep(1)
    return video, state, round(time.monotonic() - started, 3)


def main(token: str, out: str, base_url: str = "http://127.0.0.1:9000/api") -> int:
    client = httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {token}"} if token else {},
        timeout=180,
    )
    report = {
        "measured_at": datetime.now(UTC).isoformat(),
        "scope": "Warm single-client live-query latency through the running node application. "
        "No concurrent experiment. Not a capacity, cold-start or multi-user result.",
        "capabilities": client.get("/v1/capabilities").json(),
        "scenes": {},
        "queries": [],
    }
    runs = {}
    for scene in ("fixed", "bumped"):
        video, state, wall = build(client, scene)
        runs[scene] = state["id"]
        report["scenes"][scene] = {
            "video_id": video["id"],
            "run_id": state["id"],
            "duration_ms": video["duration_ms"],
            "observations": state["observation_count"],
            "index_seconds": state["elapsed_seconds"],
            "queue_to_complete_seconds": wall,
            "camera": state["provenance"].get("camera"),
        }
    # Three warm-up questions that are not timed.
    for _ in range(3):
        client.post(
            f"/v1/runs/{runs['fixed']}/questions",
            json={"text": "Where is Red toolkit?", "at_ms": 7000, "use_provider": True},
        )
    for scene, run_id in runs.items():
        for cutoff in CUTOFFS:
            for text, kind in QUESTIONS:
                started = time.monotonic()
                failure = None
                try:
                    response = client.post(
                        f"/v1/runs/{run_id}/questions",
                        json={"text": text, "at_ms": cutoff, "use_provider": True},
                    )
                    answer = response.json() if response.status_code == 200 else None
                    if answer is None:
                        failure = f"HTTP {response.status_code}"
                except Exception as exc:  # a timeout is a result, not a reason to stop
                    answer, failure = None, f"{type(exc).__name__}: {exc}"
                report["queries"].append(
                    {
                        "scene": scene,
                        "cutoff_ms": cutoff,
                        "kind": kind,
                        "question": text,
                        "seconds": round(time.monotonic() - started, 3),
                        "failure": failure,
                        "planner": (answer or {}).get("planner"),
                        "warnings": (answer or {}).get("warnings"),
                        "intent": (answer or {}).get("intent"),
                        "objects": [s["name"] for s in (answer or {}).get("states", [])],
                        "evidence": len((answer or {}).get("evidence", [])),
                    }
                )
    timings = [q["seconds"] for q in report["queries"] if not q["failure"]]
    timings.sort()
    report["summary"] = {
        "queries": len(report["queries"]),
        "failures": sum(1 for q in report["queries"] if q["failure"]),
        "planner_fallbacks": sum(1 for q in report["queries"] if q["planner"] != "agent"),
        "answers_with_warnings": sum(1 for q in report["queries"] if q["warnings"]),
        "median_seconds": round(statistics.median(timings), 3),
        "p95_seconds": round(nearest_rank_p95(timings), 3),
        "minimum_seconds": round(timings[0], 3),
        "maximum_seconds": round(timings[-1], 3),
    }
    with open(out, "w") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"scenes": report["scenes"], "summary": report["summary"]}, indent=2))
    by_kind = {}
    for q in report["queries"]:
        by_kind.setdefault(q["kind"], []).append(q["seconds"])
    for kind, values in by_kind.items():
        print(
            f"  {kind:10s} n={len(values):2d} median {statistics.median(values):6.3f}s  max {max(values):6.3f}s"
        )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("token", help="Access token, or an empty string for a loopback test app.")
    parser.add_argument("output", help="Private JSON report path.")
    parser.add_argument("--base-url", default="http://127.0.0.1:9000/api")
    args = parser.parse_args()
    raise SystemExit(main(args.token, args.output, args.base_url))
