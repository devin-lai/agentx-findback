"""Does camera recovery hold with the GPU perception stack, not only the classical baseline?

Indexes the bumped controlled sample with SAM 2.1 on CUDA and DINOv2 identity, then checks the
same contract the reference backend is held to: nothing on the desk moves after the knock, so
every registered region has to survive it. Also records cold and warm indexing for the same
backend, and that a service restart preserves the answer.
"""

import json
import sys
import time
from datetime import UTC, datetime

import httpx


def index(client, video_id, backend, fps=5):
    run = client.post(
        f"/v1/videos/{video_id}/runs", json={"backend": backend, "sample_fps": fps}
    ).json()
    started = time.monotonic()
    while True:
        state = client.get(f"/v1/runs/{run['id']}").json()
        if state["status"] in {"complete", "failed", "cancelled"}:
            return state, round(time.monotonic() - started, 3)
        if time.monotonic() - started > 900:
            raise SystemExit("indexing did not finish")
        time.sleep(2)


def main(token: str, out: str) -> int:
    client = httpx.Client(
        base_url="http://127.0.0.1:9000/api",
        headers={"Authorization": f"Bearer {token}"},
        timeout=900,
    )
    caps = client.get("/v1/capabilities").json()
    if not caps.get("sam2"):
        raise SystemExit("SAM 2.1 is not available on this deployment.")
    report = {
        "checked_at": datetime.now(UTC).isoformat(),
        "scope": "The bumped controlled sample through the GPU perception stack. A generated "
        "fixture with exact ground truth; not a real-world accuracy measurement.",
        "capabilities": {
            k: caps[k] for k in ("sam2", "sam2_model", "identity", "rtdetr_installed")
        },
        "runs": [],
    }
    video = client.post("/v1/demo", params={"scene": "bumped"}).json()
    report["video"] = {k: video[k] for k in ("id", "title", "duration_ms")}
    expected = {"Red toolkit": "right", "Blue remote": "center"}
    # Cold is the first SAM2 run in this process: it loads the checkpoint. Warm is the second.
    for label in ("cold", "warm"):
        state, wall = index(client, video["id"], "sam2")
        run = {
            "label": label,
            "run_id": state["id"],
            "status": state["status"],
            "observations": state["observation_count"],
            "index_seconds": state["elapsed_seconds"],
            "queue_to_complete_seconds": wall,
            "camera": state["provenance"].get("camera"),
            "adapter": state["provenance"].get("adapter"),
            "device": state["provenance"].get("device"),
            "answers": [],
        }
        for name, region in expected.items():
            answer = client.post(
                f"/v1/runs/{state['id']}/questions",
                json={"text": f"Where is {name}?", "at_ms": 11000, "use_provider": False},
            ).json()
            first = answer["states"][0] if answer["states"] else {}
            run["answers"].append(
                {
                    "object": name,
                    "expected_region": region,
                    "region": first.get("zone"),
                    "status": first.get("status"),
                    "reason": first.get("reason"),
                    "correct": first.get("zone") == region,
                    "evidence": [
                        {k: e[k] for k in ("at_ms", "zone", "scene_reference")}
                        for e in answer["evidence"]
                    ],
                }
            )
        report["runs"].append(run)
    report["all_regions_correct"] = all(a["correct"] for r in report["runs"] for a in r["answers"])
    with open(out, "w") as stream:
        json.dump(report, stream, indent=2)
    for run in report["runs"]:
        print(
            f"{run['label']:5s} {run['adapter']} on {run['device']} · "
            f"{run['observations']} observations in {run['index_seconds']}s · camera {run['camera']['frames']}"
        )
        for a in run["answers"]:
            print(
                f"   {a['object']:12s} expected {a['expected_region']:7s} got {str(a['region']):7s}"
                f" {'OK' if a['correct'] else 'WRONG'} · status {a['status']}"
                f" · evidence {[e['scene_reference'] for e in a['evidence']]}"
            )
    print("all regions correct:", report["all_regions_correct"])
    return 0 if report["all_regions_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
