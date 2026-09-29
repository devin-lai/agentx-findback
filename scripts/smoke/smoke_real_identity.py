"""Verify the deployed guarded tracker, Nemotron, and offline evidence on cup-20.

This is a previously inspected LaSOT research clip, not a held-out accuracy test.
Media and exported frames stay in local research artifacts outside source releases.
"""

import argparse
import io
import json
import time
from pathlib import Path
from zipfile import ZipFile

import httpx

from agentx.config import Settings
from agentx.services.bundle_verify import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9000")
    parser.add_argument(
        "--dataset", type=Path, default=Path("artifacts/datasets/lasot-prepared/cup-20")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = Settings()
    headers = {"Authorization": f"Bearer {settings.api_token}"} if settings.api_token else {}
    labels = json.loads((args.dataset / "labels.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", scope=__doc__, answers=[])

    def save():
        (args.output / "live-identity.json").write_text(json.dumps(report, indent=2) + "\n")

    with httpx.Client(base_url=args.base_url, headers=headers, timeout=180) as client:

        def call(method, path, **kwargs):
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()

        with (args.dataset / "cup-20.mp4").open("rb") as source:
            video = call(
                "POST",
                "/api/v1/videos",
                files={"file": ("LaSOT-cup-20-research.mp4", source, "video/mp4")},
            )
        obj = call(
            "POST",
            f"/api/v1/videos/{video['id']}/objects",
            json=dict(
                name="Registered paper cup", label="cup", at_ms=0, box=labels["labels"][0]["box"]
            ),
        )
        run = call(
            "POST", f"/api/v1/videos/{video['id']}/runs", json=dict(backend="sam2", sample_fps=5)
        )
        report.update(video_id=video["id"], object_id=obj["id"], run_id=run["id"])
        save()
        deadline = time.monotonic() + 600
        while run["status"] in {"queued", "running"} and time.monotonic() < deadline:
            time.sleep(1)
            run = call("GET", f"/api/v1/runs/{run['id']}")
        if run["status"] != "complete":
            raise ValueError(f"Index did not complete: {run['status']}")
        report["run"] = run
        if not run["provenance"].get("distractor_guard", {}).get("enabled"):
            raise ValueError("The live service has not enabled the companion guard.")
        for name, cutoff, expected in [
            ("before-collision", 10000, "visible"),
            ("after-collision", 30000, "unknown"),
        ]:
            answer = call(
                "POST",
                f"/api/v1/runs/{run['id']}/questions",
                json=dict(
                    text="Where is Registered paper cup?",
                    object_id=obj["id"],
                    at_ms=cutoff,
                    use_provider=True,
                ),
            )
            state = answer["states"][0] if len(answer["states"]) == 1 else {}
            checks = dict(
                state=state.get("status") == expected,
                only_registered_object=state.get("object_id") == obj["id"],
                actual_nemotron=answer["planner"] == "agent"
                and "nemotron" in (answer.get("planner_model") or "").lower(),
                cuda=run["provenance"]["device"] == "cuda",
                causal=bool(answer["evidence"])
                and all(e["at_ms"] <= cutoff for e in answer["evidence"]),
            )
            if expected == "unknown":
                checks["current_location_unconfirmed"] = state.get("current_zone") is None
                checks["last_supported_before_collision"] = (
                    isinstance(state.get("last_observed_ms"), int)
                    and state["last_observed_ms"] < 23800
                )
            response = client.get(f"/api/v1/questions/{answer['id']}/bundle")
            response.raise_for_status()
            (args.output / f"{name}.zip").write_bytes(response.content)
            folder = args.output / name
            with ZipFile(io.BytesIO(response.content)) as bundle:
                bundle.extractall(folder)
            verification = verify(folder)
            checks["bundle_integrity"] = verification["status"] == "verified"
            report["answers"].append(
                dict(name=name, answer=answer, checks=checks, verification=verification)
            )
            save()
            print(json.dumps(dict(name=name, checks=checks)), flush=True)
        report["status"] = (
            "complete" if all(all(a["checks"].values()) for a in report["answers"]) else "failed"
        )
        save()
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
