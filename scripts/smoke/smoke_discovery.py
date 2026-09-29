"""Run registration discovery against the configured backends on a generated fixture.

This exercises the real perception path and records what each backend proposed. It is a
software smoke check on a synthetic clip, not a detection accuracy benchmark: the fixture's
two patches are drawn rectangles, so a low RT-DETR count is expected and is not a defect.

The decisive property it asserts is the contract, not the proposals: after discovery runs, the
video's registered objects are exactly what they were before.
"""

import argparse
import json
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from agentx.api.app import create_app
from agentx.config import Settings
from agentx.domain.contracts import Box, RegisterObject
from agentx.services.demo import create_fixture
from agentx.storage.models import RegisteredObject
from agentx.vision.cosmos_detector import iou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=["rtdetr", "cosmos"])
    parser.add_argument("--at-ms", type=int, default=1000)
    parser.add_argument("--source", type=Path, help="Use this video instead of the fixture.")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Register the first clip's labeled objects from a benchmark manifest before "
        "discovery, so --against-registrations has boxes to compare with.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/discovery-smoke.json"))
    parser.add_argument(
        "--against-registrations",
        action="store_true",
        help="Report the best proposal IoU against each registered box, to show how close a "
        "proposal starts to a box a person drew. Two boxes, one frame: not a benchmark.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="agentx-discovery-") as directory:
        root = Path(directory)
        app = create_app(Settings(data_dir=root / "data", database_url="", worker_enabled=False))
        s = app.state.services
        s.db.migrate()
        try:
            available = s.discovery.available()
            incoming = root / "incoming.mp4"
            if args.source:
                shutil.copyfile(args.source, incoming)
                registrations = []
                if args.manifest:
                    clip = json.loads(args.manifest.read_text())["clips"][0]
                    registrations = [RegisterObject(**obj) for obj in clip["objects"]]
            else:
                source = root / "fixture.mp4"
                registrations = create_fixture(source)
                shutil.copyfile(source, incoming)
            video = s.catalog.import_video(incoming, "Discovery smoke", is_fixture=not args.source)
            for obj in registrations:
                s.catalog.register(video.id, obj)
            with s.db.session() as session:
                before = sorted(
                    o.name
                    for o in session.scalars(
                        select(RegisteredObject).where(RegisteredObject.video_id == video.id)
                    )
                )

            results = {}
            for backend in args.backends:
                if not available.get(backend):
                    results[backend] = {"skipped": "backend not available in this environment"}
                    print(json.dumps({backend: "skipped"}), flush=True)
                    continue
                with s.db.session() as session:
                    taken = list(
                        session.scalars(
                            select(RegisteredObject).where(RegisteredObject.video_id == video.id)
                        )
                    )
                started = time.monotonic()
                try:
                    suggestion = s.discovery.suggest(video, args.at_ms, backend, taken)
                except Exception as exc:
                    results[backend] = {"error": f"{type(exc).__name__}: {exc}"}
                    print(json.dumps({backend: results[backend]}), flush=True)
                    continue
                results[backend] = {
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "count": len(suggestion["proposals"]),
                    "provenance": suggestion["provenance"],
                    "proposals": suggestion["proposals"],
                }
                if args.against_registrations and taken:
                    results[backend]["against_registrations"] = [
                        {
                            "registered": obj.name,
                            "best_iou": round(
                                max(
                                    (
                                        iou(Box(**obj.box), Box(**p["box"]))
                                        for p in suggestion["proposals"]
                                    ),
                                    default=0.0,
                                ),
                                4,
                            ),
                        }
                        for obj in taken
                    ]
                print(
                    json.dumps(
                        {
                            backend: [p["suggested_name"] for p in suggestion["proposals"]],
                            "seconds": results[backend]["elapsed_seconds"],
                            "best_iou": results[backend].get("against_registrations"),
                        }
                    ),
                    flush=True,
                )

            with s.db.session() as session:
                after = sorted(
                    o.name
                    for o in session.scalars(
                        select(RegisteredObject).where(RegisteredObject.video_id == video.id)
                    )
                )
            report = {
                "created_at": datetime.now(UTC).isoformat(),
                "at_ms": args.at_ms,
                "available": available,
                "registered_before": before,
                "registered_after": after,
                "registrations_unchanged": before == after,
                "results": results,
                "scope": (
                    "Generated fixture unless --source was given. A proposal count is not a "
                    "detection accuracy measurement."
                ),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
            if before != after:
                raise SystemExit(
                    "Discovery changed the registered objects; that must never happen."
                )
            return 0
        finally:
            s.db.engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
