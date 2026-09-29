"""Verify a FindBack SQLite/media snapshot without starting model services.

The database is opened read-only. Use an online SQLite backup (or stop the app
before copying database plus WAL files) and copy the data/videos directory before
running this verifier. It checks stored bytes and references, not model accuracy.
"""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path, PurePosixPath


def media_file(root: Path, stored: str) -> Path:
    if not isinstance(stored, str) or not stored or "\\" in stored:
        raise ValueError(f"Invalid stored media path: {stored!r}")
    path = PurePosixPath(stored)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in stored.split("/")):
        raise ValueError(f"Invalid stored media path: {stored!r}")
    resolved = (root / stored).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Stored media path leaves the data directory: {stored!r}")
    if not resolved.is_file():
        raise ValueError(f"Missing stored media file: {stored!r}")
    return resolved


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(database: Path, data_dir: Path) -> dict:
    root = data_dir.resolve(strict=True)
    db = database.resolve(strict=True)
    with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("SQLite quick_check failed")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("SQLite foreign_key_check failed")

        videos = {row["id"]: row for row in connection.execute(
            "SELECT id, sha256, source_path, media_path, duration_ms FROM videos"
        )}
        bytes_hashed = 0
        for video in videos.values():
            source = media_file(root, video["source_path"])
            if file_sha256(source) != video["sha256"]:
                raise ValueError(f"Original video hash mismatch: {video['id']}")
            bytes_hashed += source.stat().st_size
            media_file(root, video["media_path"])
            media_file(root, f"videos/{video['id']}/poster.jpg")

        objects = {}
        for obj in connection.execute("SELECT id, video_id, reference_path FROM objects"):
            if obj["video_id"] not in videos:
                raise ValueError(f"Object has no video: {obj['id']}")
            media_file(root, obj["reference_path"])
            objects[obj["id"]] = obj["video_id"]

        runs = {}
        completed = 0
        for run in connection.execute("SELECT id, video_id, status FROM index_runs"):
            if run["video_id"] not in videos:
                raise ValueError(f"Run has no video: {run['id']}")
            runs[run["id"]] = run["video_id"]
            completed += run["status"] == "complete"

        observations = {}
        for observation in connection.execute(
            "SELECT id, run_id, object_id, at_ms, visible FROM observations"
        ):
            if observations.get(observation["id"]):
                raise ValueError(f"Duplicate observation: {observation['id']}")
            video_id = runs.get(observation["run_id"])
            if video_id is None or objects.get(observation["object_id"]) != video_id:
                raise ValueError(f"Observation has inconsistent scope: {observation['id']}")
            observations[observation["id"]] = observation

        questions = 0
        citations = 0
        for question in connection.execute(
            "SELECT id, run_id, cutoff_ms, response FROM queries"
        ):
            questions += 1
            video_id = runs.get(question["run_id"])
            if video_id is None:
                raise ValueError(f"Question has no run: {question['id']}")
            cutoff = question["cutoff_ms"]
            if not 0 <= cutoff <= videos[video_id]["duration_ms"]:
                raise ValueError(f"Question cutoff is outside video: {question['id']}")
            answer = json.loads(question["response"])
            if answer.get("id") != question["id"] or answer.get("run_id") != question["run_id"]:
                raise ValueError(f"Saved answer identity mismatch: {question['id']}")
            state_ids = {state["object_id"] for state in answer.get("states", [])}
            for evidence in answer.get("evidence", []):
                observation = observations.get(evidence.get("observation_id"))
                if (
                    observation is None
                    or not observation["visible"]
                    or observation["run_id"] != question["run_id"]
                    or observation["object_id"] not in state_ids
                    or observation["at_ms"] > cutoff
                    or evidence.get("at_ms") != observation["at_ms"]
                    or evidence.get("run_id") != question["run_id"]
                    or evidence.get("video_id") != video_id
                ):
                    raise ValueError(f"Saved answer citation mismatch: {question['id']}")
                citations += 1

        return {
            "status": "verified",
            "videos": len(videos),
            "objects": len(objects),
            "runs": len(runs),
            "completed_runs": completed,
            "observations": len(observations),
            "saved_questions": questions,
            "citations": citations,
            "source_bytes_hashed": bytes_hashed,
            "scope": "SQLite and stored media integrity; not visual accuracy or model availability",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.database, args.data_dir), indent=2))


if __name__ == "__main__":
    main()
