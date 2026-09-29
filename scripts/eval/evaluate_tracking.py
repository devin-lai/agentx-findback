"""Measure end-to-end perception against independent LaSOT boxes and visibility flags.

This is a public tracking stress test, not a held-out desk-user study. Camera
changes, abstentions and wrong-object boxes remain in the denominator. Reports
checkpoint after each clip, including failed indexing jobs.
"""

import argparse
import hashlib
import json
import platform
import shutil
import statistics
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from agentx.api.app import create_app
from agentx.config import Settings
from agentx.domain.contracts import IndexRequest, RegisterObject
from agentx.storage.models import IndexRun, Observation


def intersection_over_union(a: dict | None, b: dict | None) -> float:
    if a is None or b is None:
        return 0.0
    intersection = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])) * max(
        0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"])
    )

    def area(box):
        return max(0, box["x2"] - box["x1"]) * max(0, box["y2"] - box["y1"])

    union = area(a) + area(b) - intersection
    return intersection / union if union else 0.0


def summarize(rows: list[dict]) -> dict:
    visible = sum(r["truth_visible"] for r in rows)
    absent = len(rows) - visible
    predicted = sum(r["predicted_visible"] for r in rows)
    correct = sum(r["truth_visible"] and r["predicted_visible"] and r["iou"] >= 0.5 for r in rows)
    return {
        "samples": len(rows),
        "truth_visible": visible,
        "truth_absent": absent,
        "predicted_visible": predicted,
        "correct_at_iou_0_5": correct,
        "precision_at_iou_0_5": correct / predicted if predicted else None,
        "recall_at_iou_0_5": correct / visible if visible else None,
        "false_visible_when_absent": sum(
            r["predicted_visible"] and not r["truth_visible"] for r in rows
        ),
        "poor_overlap_when_visible": sum(
            r["truth_visible"] and r["predicted_visible"] and r["iou"] < 0.1 for r in rows
        ),
        "mean_iou_when_truth_visible": statistics.mean(r["iou"] for r in rows if r["truth_visible"])
        if visible
        else None,
        "zone_correct_and_visible": sum(
            r["truth_visible"] and r["predicted_visible"] and r["truth_zone"] == r["predicted_zone"]
            for r in rows
        ),
        "abstention_reasons": dict(
            Counter(r["reason"] or "unspecified" for r in rows if not r["predicted_visible"])
        ),
    }


_RELEASED: dict = {}


def use_scene_guard(name: str) -> dict:
    """Select the camera-evidence implementation this arm runs with."""
    from agentx.services import indexing
    from agentx.vision.scene import SceneTracker

    # Capture the shipped pair once, so repeated arm selection cannot restore a patched one.
    released = _RELEASED.setdefault("pair", (indexing.SceneTracker, indexing._anchor))

    if name == "legacy":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from legacy_scene_guard import LegacySceneGuard, legacy_anchor

        indexing.SceneTracker = LegacySceneGuard  # type: ignore[misc]
        indexing._anchor = legacy_anchor  # type: ignore[assignment]
        return dict(LegacySceneGuard.policy)
    indexing.SceneTracker, indexing._anchor = released  # type: ignore[misc,assignment]
    assert indexing.SceneTracker is SceneTracker
    return dict(SceneTracker.policy)


def evaluate(directory: Path, backend: str, sample_fps: float, identity: bool | None) -> dict:
    labels_raw = (directory / "labels.json").read_bytes()
    labels = json.loads(labels_raw)
    manifest = json.loads((directory / "manifest.json").read_text())
    clip = manifest["clips"][0]
    source = directory / clip["path"]
    with source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != labels["video_sha256"]:
            raise ValueError("Video checksum differs from the frozen annotations.")
    with tempfile.TemporaryDirectory(prefix="agentx-tracking-") as temporary:
        root = Path(temporary)
        overrides = {"identity_enabled": identity} if identity is not None else {}
        app = create_app(
            Settings(data_dir=root / "data", database_url="", worker_enabled=False, **overrides)
        )
        s = app.state.services
        s.db.migrate()
        try:
            incoming = root / source.name
            shutil.copyfile(source, incoming)
            video = s.catalog.import_video(incoming, labels["sequence"], is_fixture=False)
            s.catalog.register(video.id, RegisterObject.model_validate(clip["objects"][0]))
            run = s.indexer.enqueue(video.id, IndexRequest(backend=backend, sample_fps=sample_fps))
            s.indexer.process(run.id)
            with s.db.session() as session:
                run = session.get(IndexRun, run.id)
                observations = session.scalars(
                    select(Observation)
                    .where(Observation.run_id == run.id)
                    .order_by(Observation.at_ms)
                ).all()
            report = {
                "sequence": labels["sequence"],
                "status": run.status,
                "error": run.error,
                "video_sha256": video.sha256,
                "labels_sha256": hashlib.sha256(labels_raw).hexdigest(),
                "label_provenance": {k: v for k, v in labels.items() if k != "labels"},
                "duration_ms": video.duration_ms,
                "index_seconds": run.elapsed_seconds,
                "index_to_video_ratio": run.elapsed_seconds / (video.duration_ms / 1000),
                "run_config": run.config,
                "provenance": run.provenance,
            }
            rows = []
            for obs in observations:
                index = round(obs.at_ms * labels["assigned_fps"] / 1000)
                if index >= len(labels["labels"]):
                    raise ValueError("An observation falls outside the frozen labels.")
                truth = labels["labels"][index]
                if abs(truth["at_ms"] - obs.at_ms) > 1:
                    raise ValueError("Source frame timestamps do not align with annotation times.")
                rows.append(
                    {
                        "at_ms": obs.at_ms,
                        "frame": truth["frame"],
                        "truth_visible": truth["visible"],
                        "truth_box": truth["box"],
                        "truth_zone": truth["zone"],
                        "predicted_visible": obs.visible,
                        "predicted_box": obs.box,
                        "predicted_zone": obs.zone,
                        "iou": intersection_over_union(
                            obs.box if obs.visible else None, truth["box"]
                        ),
                        "reason": obs.reason,
                        "detector_score": obs.score,
                        "identity_score": obs.identity_score,
                    }
                )
            if run.status == "complete" and not rows:
                raise ValueError("Completed index produced no observations.")
            report.update(summary=summarize(rows), rows=rows)
            return report
        finally:
            s.db.engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/datasets/lasot-prepared"))
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument(
        "--backend", choices=["reference", "rtdetr", "cosmos", "sam2"], default="rtdetr"
    )
    parser.add_argument("--sample-fps", type=float, default=5)
    parser.add_argument("--identity", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--split", choices=["development", "evaluation"], required=True)
    parser.add_argument(
        "--scene-guard",
        choices=["compensated", "legacy"],
        default="compensated",
        help="Camera-evidence arm: the released tracker, or the previous latching control.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[2] / "src" / "agentx"
    scene_policy = use_scene_guard(args.scene_guard)
    report = {
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "Public LaSOT tracking research stress test; not a desk-user study or evidence of new real-world generalization.",
        "platform": platform.platform(),
        "split": args.split,
        "scene_guard_arm": args.scene_guard,
        "scene_guard_policy": scene_policy,
        "source_hashes": {
            str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(code.rglob("*.py"))
        },
        "clips": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for sequence in args.sequences:
        try:
            result = evaluate(args.dataset / sequence, args.backend, args.sample_fps, args.identity)
        except Exception as exc:
            result = {
                "sequence": sequence,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        report["clips"].append(result)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k in {"sequence", "status", "error", "summary", "index_seconds"}
                }
            ),
            flush=True,
        )
    report["status"] = "complete"
    report["failed_clips"] = sum(c["status"] != "complete" for c in report["clips"])
    report["summary"] = summarize([r for c in report["clips"] for r in c.get("rows", [])])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return int(bool(report["failed_clips"]))


if __name__ == "__main__":
    raise SystemExit(main())
