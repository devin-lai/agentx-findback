"""Check frozen case labels against memory read directly from storage.

This exists so a held-out suite can be verified before any arm of an evaluation runs
against it. It never calls the query workflow, the deterministic resolver or a planner,
so confirming a label here cannot tune the suite towards an arm that is later measured.
It prints the fixture's own trajectory facts (per-object state at each cutoff, first
observation, recorded events) next to every case's expected status and zone.

A mismatch means the label is wrong, not that a model failed.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from agentx.api.app import create_app
from agentx.config import Settings
from agentx.domain.contracts import IndexRequest
from agentx.services.demo import create_fixture
from agentx.storage.models import Video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dataset = json.loads(args.cases.read_text())
    cases = dataset["cases"]

    with tempfile.TemporaryDirectory(prefix="agentx-holdout-labels-") as directory:
        root = Path(directory)
        app = create_app(Settings(data_dir=root / "data", database_url="", worker_enabled=False))
        s = app.state.services
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
            video = s.catalog.import_video(incoming, "Label verification fixture", is_fixture=True)
            if customization.get("regions"):
                with s.db.session.begin() as session:
                    stored = session.get(Video, video.id)
                    stored.regions = [
                        {**region, **customization["regions"].get(region["id"], {})}
                        for region in stored.regions
                    ]
            for obj in registrations:
                s.catalog.register(video.id, obj)
            run = s.indexer.enqueue(video.id, IndexRequest(backend="reference"))
            s.indexer.process(run.id)

            cutoffs = sorted({c["at_ms"] for c in cases})
            trajectory: dict[str, dict] = {}
            for cutoff in cutoffs:
                run_ctx, video_ctx, objects, resolved = s.memory.context(run.id, cutoff)
                for obj in objects:
                    state = s.memory.object_state(run_ctx, video_ctx, obj, resolved)
                    trajectory.setdefault(obj.name, {})[cutoff] = {
                        "status": state.status,
                        "zone": state.zone,
                        "current_zone": state.current_zone,
                        "last_observed_ms": state.last_observed_ms,
                    }

            run_ctx, video_ctx, objects, end = s.memory.context(run.id, None)
            facts = {}
            for obj in objects:
                events = s.memory.history(run.id, end, obj.id)
                first = min(
                    (e for e in events if e["kind"] == "appeared"),
                    key=lambda e: e["at_ms"],
                    default=None,
                )
                facts[obj.name] = {
                    "registered_at_ms": obj.registered_at_ms,
                    "first_appeared": {"at_ms": first["at_ms"], "zone": first["zone"]}
                    if first
                    else None,
                    "events": [
                        {
                            "kind": e["kind"],
                            "at_ms": e["at_ms"],
                            "zone": e["zone"],
                            "previous_zone": e["previous_zone"],
                        }
                        for e in events
                    ],
                    "states_by_cutoff": trajectory.get(obj.name, {}),
                }

            mismatches = []
            for number, case in enumerate(cases):
                name = case["expected_object"]
                if not name:
                    continue
                observed = trajectory.get(name, {}).get(case["at_ms"])
                if observed is None:
                    mismatches.append({"number": number, "reason": "no state", "case": case})
                    continue
                for key in ("status", "zone"):
                    expected = case.get(f"expected_{key}")
                    if expected is not None and observed[key] != expected:
                        mismatches.append(
                            {
                                "number": number,
                                "text": case["text"],
                                "field": key,
                                "expected": expected,
                                "observed": observed[key],
                            }
                        )

            report = {
                "cases": str(args.cases),
                "fixture_facts": facts,
                "labelled_cases_checked": sum(1 for c in cases if c["expected_object"]),
                "mismatches": mismatches,
            }
            print(json.dumps(report, indent=2))
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
            return 1 if mismatches else 0
        finally:
            s.db.engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
