"""Preserve verified legacy review JPEGs before migrating the complete data directory.

Run on the original working decoder/runtime. No model calls or memory-row changes.
Frames that cannot reproduce their original hash remain failures, never replacements.
"""

import argparse
import json
from pathlib import Path

from sqlalchemy import select

from agentx.api.app import create_app
from agentx.config import Settings
from agentx.storage.models import ReviewRecord


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    services = create_app(Settings(worker_enabled=False)).state.services
    with services.db.session() as session:
        reviews = [r.response for r in session.scalars(select(ReviewRecord)).all()]
    results = []
    for review in reviews:
        for frame in review["frames"]:
            row = {"review_id": review["id"], "frame_id": frame["id"], "sha256": frame["sha256"]}
            try:
                services.reviews.frame(review["id"], frame["id"])
                row["status"] = "verified"
            except (ValueError, LookupError, OSError) as exc:
                row.update(status="failed", error=str(exc))
            results.append(row)
    report = {
        "status": "verified" if all(r["status"] == "verified" for r in results) else "failed",
        "reviews": len(reviews),
        "frames": results,
        "scope": "Exact reviewed bytes preserved; no visual-accuracy or authorship claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "reviews": len(reviews), "frames": len(results)}))
    if report["status"] != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
