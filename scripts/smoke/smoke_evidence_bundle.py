"""Exercise the running Spark app: CUDA SAM2, actual Nemotron answers, portable evidence.

Run from the deployed repository, using its local Settings for authentication.
Creates a labeled generated fixture and keeps the resulting memory for UI replay.
No real-world accuracy claim; no secrets are written to the receipt.
"""

import argparse
import hashlib
import io
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

import httpx

from agentx.config import Settings
from agentx.services.bundle_verify import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    headers = {"Authorization": f"Bearer {settings.api_token}"} if settings.api_token else {}
    receipt: dict = {
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "Live Spark pipeline on a generated software fixture; not a real-footage benchmark.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "answers": [],
    }

    def save():
        (args.output / "spark-smoke.json").write_text(json.dumps(receipt, indent=2) + "\n")

    with httpx.Client(base_url=args.base_url, headers=headers, timeout=180) as client:

        def call(method, path, **kwargs):
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()

        try:
            video = call("POST", "/api/v1/demo")
            receipt["video_id"] = video["id"]
            run = call("POST", f"/api/v1/videos/{video['id']}/runs", json={"backend": "sam2"})
            receipt["run_id"] = run["id"]
            save()
            deadline = time.monotonic() + 300
            while run["status"] in {"queued", "running"} and time.monotonic() < deadline:
                time.sleep(0.5)
                run = call("GET", f"/api/v1/runs/{run['id']}")
            if run["status"] != "complete":
                raise ValueError(f"Indexing did not complete: {run['status']}")
            if run["provenance"].get("device") != "cuda":
                raise ValueError("The SAM2 run did not execute on CUDA.")
            receipt["index"] = {
                "backend": run["backend"],
                "device": run["provenance"]["device"],
                "elapsed_seconds": run["elapsed_seconds"],
                "observations": run["observation_count"],
                "weights_sha256": run["provenance"]["weights_sha256"],
            }
            cases = [
                ("early-location", "Where is Red toolkit?", 2000, "visible", "left"),
                (
                    "uncertain-location",
                    "Where is the object that is no longer visible?",
                    7000,
                    "last_seen",
                    "right",
                ),
                (
                    "object-history",
                    "Show the history of the item that disappeared.",
                    7000,
                    "last_seen",
                    "right",
                ),
            ]
            for name, question, cutoff, status, zone in cases:
                answer = call(
                    "POST",
                    f"/api/v1/runs/{run['id']}/questions",
                    json={
                        "text": question,
                        "at_ms": cutoff,
                        "use_provider": True,
                    },
                )
                state = answer["states"][0] if answer["states"] else {}
                checks = {
                    "actual_agent": answer["planner"] == "agent",
                    "nemotron": "nemotron" in (answer.get("planner_model") or "").lower(),
                    "object": state.get("name") == "Red toolkit",
                    "state": state.get("status") == status,
                    "zone": state.get("zone") == zone,
                    "causal_evidence": bool(answer["evidence"])
                    and all(e["at_ms"] <= cutoff for e in answer["evidence"]),
                }
                started = time.monotonic()
                response = client.get(f"/api/v1/questions/{answer['id']}/bundle")
                response.raise_for_status()
                archive = args.output / f"{name}.zip"
                archive.write_bytes(response.content)
                folder = args.output / name
                # Input is the authenticated application's just-generated archive.
                with ZipFile(io.BytesIO(response.content)) as bundle:
                    bundle.extractall(folder)
                verified = verify(folder)
                checks["bundle_verified"] = verified["status"] == "verified"
                result = {
                    "name": name,
                    "answer": answer,
                    "checks": checks,
                    "passed": all(checks.values()),
                    "verification": verified,
                    "export_seconds": round(time.monotonic() - started, 3),
                    "zip_bytes": len(response.content),
                    "zip_sha256": hashlib.sha256(response.content).hexdigest(),
                }
                receipt["answers"].append(result)
                save()
                print(
                    json.dumps(
                        {k: result[k] for k in ("name", "checks", "export_seconds", "zip_bytes")}
                    ),
                    flush=True,
                )
            receipt["status"] = (
                "complete" if all(a["passed"] for a in receipt["answers"]) else "failed"
            )
            save()
        except Exception as exc:
            receipt["status"] = "failed"
            receipt["error_type"] = type(exc).__name__
            save()
            raise
    raise SystemExit(0 if receipt["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
