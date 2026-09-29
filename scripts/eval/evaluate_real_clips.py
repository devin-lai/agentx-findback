"""Answer frozen questions about real desk clips through a running FindBack server and grade them.

For each clip in the question file: upload the video, register its objects at the frozen boxes and
times, build memory with one perception backend, then ask "Where was <object> last seen?" at every
frozen cutoff with the deterministic resolver (`use_provider=false`), so the score measures memory
rather than planner wording. Grading, from the question file:

* the status must match (`visible`, `last_seen`, `unknown` or `not_observed`);
* `visible`: the answer's zone must be one of `expected_zones`;
* `last_seen`: the evidence time must fall inside `last_seen_window_ms` widened by `tolerance_ms`,
  and its zone must be one of `expected_zones`.
* `unknown` and `not_observed`: no location is graded; an asserted visible or last-seen
  state is a miss.

Every clip and label is checked before the first HTTP request. Each file is checked again
immediately before upload. Nothing is retried or tuned.
"""

import argparse
import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import httpx

from agentx.domain.contracts import Box, default_regions
from agentx.vision.video import probe

MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
}


def preflight(spec: dict, clips_dir: Path, *, min_evaluation_clips: int = 0) -> dict:
    """Validate the *entire* frozen corpus before any server or model request.

    This cannot verify that a human's answer labels are true. It catches malformed
    metadata, changed files and impossible times before a partial run pollutes the
    first-attempt record.
    """
    root = clips_dir.resolve(strict=True)
    clips = spec.get("clips") if isinstance(spec, dict) else None
    if not isinstance(clips, list) or not clips:
        raise ValueError("The question file needs at least one clip.")
    tolerance = spec.get("tolerance_ms")
    if type(tolerance) is not int or not 0 <= tolerance <= 5000:
        raise ValueError("tolerance_ms must be an integer between 0 and 5000.")
    if min_evaluation_clips < 0:
        raise ValueError("min_evaluation_clips cannot be negative.")
    regions = {region["id"] for region in default_regions()}
    filenames = set()
    rows = []
    for number, clip in enumerate(clips):
        if not isinstance(clip, dict):
            raise ValueError(f"Clip {number} must be an object.")
        filename = clip.get("file")
        if not isinstance(filename, str) or not filename or "\\" in filename:
            raise ValueError(f"Clip {number} has an invalid filename.")
        parts = filename.split("/")
        if any(part in {"", ".", ".."} for part in parts) or PurePosixPath(filename).is_absolute():
            raise ValueError(f"Clip {number} has an unsafe filename.")
        if filename in filenames:
            raise ValueError(f"Duplicate clip filename: {filename}")
        filenames.add(filename)
        suffix = Path(filename).suffix.lower()
        if suffix not in MEDIA_TYPES:
            raise ValueError(f"Unsupported video extension: {filename}")
        path = (root / filename).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Clip path leaves the supplied directory: {filename}")
        digest = clip.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError(f"Clip {filename} needs a lowercase SHA-256 digest.")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"Clip {filename} does not match its frozen SHA-256.")
        split = clip.get("split")
        if not isinstance(split, str) or split not in {"development", "evaluation"}:
            raise ValueError(f"Clip {filename} needs a development or evaluation split.")
        info = probe(path)
        objects = clip.get("objects")
        if not isinstance(objects, list) or not objects:
            raise ValueError(f"Clip {filename} needs registered objects.")
        names = set()
        for item in objects:
            if not isinstance(item, dict):
                raise ValueError(f"Clip {filename} has a malformed object.")
            name = item.get("name")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError(f"Clip {filename} has an empty or duplicate object name.")
            names.add(name)
            at_ms = item.get("at_ms")
            if type(at_ms) is not int or not 0 <= at_ms <= info.duration_ms:
                raise ValueError(f"Registration time for {name} is outside {filename}.")
            box = item.get("box")
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(f"Registration box for {name} must have four coordinates.")
            try:
                Box(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Registration box for {name} is invalid: {exc}") from exc
        questions = clip.get("questions")
        if not isinstance(questions, list) or not questions:
            raise ValueError(f"Clip {filename} needs at least one question.")
        for index, question in enumerate(questions):
            if (
                not isinstance(question, dict)
                or not isinstance(question.get("object"), str)
                or question["object"] not in names
            ):
                raise ValueError(f"Question {index} in {filename} names no registered object.")
            cutoff = question.get("at_ms")
            if type(cutoff) is not int or not 0 <= cutoff <= info.duration_ms:
                raise ValueError(f"Question {index} cutoff is outside {filename}.")
            status = question.get("status")
            if not isinstance(status, str) or status not in {
                "visible",
                "last_seen",
                "unknown",
                "not_observed",
            }:
                raise ValueError(f"Question {index} in {filename} has an unsupported status.")
            zones = question.get("expected_zones")
            if status in {"visible", "last_seen"}:
                if (
                    not isinstance(zones, list)
                    or not zones
                    or any(not isinstance(z, str) or z not in regions for z in zones)
                ):
                    raise ValueError(f"Question {index} in {filename} needs valid expected zones.")
            elif zones not in (None, []):
                raise ValueError(f"Refusal question {index} in {filename} must not claim a zone.")
            window = question.get("last_seen_window_ms")
            if status == "last_seen":
                if (
                    not isinstance(window, list)
                    or len(window) != 2
                    or any(type(value) is not int for value in window)
                    or not 0 <= window[0] <= window[1] <= cutoff
                ):
                    raise ValueError(
                        f"Question {index} in {filename} has an invalid sighting window."
                    )
            elif window is not None:
                raise ValueError(f"Question {index} in {filename} has an unneeded sighting window.")
        rows.append(
            {
                "file": filename,
                "sha256": digest,
                "bytes": path.stat().st_size,
                "duration_ms": info.duration_ms,
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
                "split": split,
                "objects": len(objects),
                "questions": len(questions),
            }
        )
    evaluation = sum(row["split"] == "evaluation" for row in rows)
    if evaluation < min_evaluation_clips:
        raise ValueError(
            f"Need at least {min_evaluation_clips} evaluation clips; found {evaluation}."
        )
    return {
        "status": "valid",
        "clip_count": len(rows),
        "evaluation_clip_count": evaluation,
        "question_count": sum(row["questions"] for row in rows),
        "clips": rows,
    }


def object_state(answer: dict, object_id: str) -> dict:
    """The asked object's memory state; its `evidence` is the cited observation."""
    return next((s for s in answer.get("states") or [] if s.get("object_id") == object_id), {})


def grade(question: dict, state: dict, tolerance: int) -> dict:
    evidence = state.get("evidence") or {}
    expected_status = question["status"]
    if expected_status not in {"visible", "last_seen", "unknown", "not_observed"}:
        raise ValueError(f"Unsupported expected status: {expected_status}")
    status_ok = state.get("status") == expected_status
    zone_ok = (
        evidence.get("zone") in question["expected_zones"]
        if expected_status in {"visible", "last_seen"}
        else True
    )
    time_ok = True
    if expected_status == "last_seen":
        start, end = question["last_seen_window_ms"]
        at = evidence.get("at_ms")
        time_ok = at is not None and start - tolerance <= at <= end + tolerance
    return {
        "correct": status_ok and zone_ok and time_ok,
        "status_ok": status_ok,
        "zone_ok": zone_ok,
        "time_ok": time_ok,
    }


def classify_result(question: dict, state: dict, checks: dict, *, error: str | None) -> dict:
    """Separate unanswered requests from unsupported location assertions.

    Coarse status/zone labels cannot establish a physical identity swap. Such cases
    require inspection of the cited original frame and are deliberately not called
    false target identifications here.
    """
    expected = question["status"]
    given = state.get("status")
    asserted = given in {"visible", "last_seen"}
    return {
        "unanswered": bool(
            error or not given or (expected in {"visible", "last_seen"} and not asserted)
        ),
        "false_visible": given == "visible" and expected != "visible",
        "wrong_zone": bool(
            asserted and expected in {"visible", "last_seen"} and not checks["zone_ok"]
        ),
        "wrong_last_seen_time": bool(
            given == "last_seen" and expected == "last_seen" and not checks["time_ok"]
        ),
        "unsupported_location_claim": bool(
            asserted
            and (
                expected in {"unknown", "not_observed"}
                or not checks["zone_ok"]
                or not checks["time_ok"]
                or (given == "visible" and expected == "last_seen")
            )
        ),
        # Status and region labels cannot detect a swap to a look-alike in the same region.
        "physical_identity_adjudication": "not_assessed",
        "frame_review_priority": bool(asserted and not checks["correct"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--clips", required=True, type=Path)
    parser.add_argument("--findback-url", default="http://127.0.0.1:9000")
    parser.add_argument("--token", default="", help="Server access token, if any")
    parser.add_argument("--backend", default="reference")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--preflight-only", action="store_true", help="Validate all files and labels without HTTP."
    )
    parser.add_argument(
        "--min-evaluation-clips",
        type=int,
        default=0,
        help="Require this many reserved evaluation clips in the frozen corpus.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Preserve first reports: choose a new output path.")
    raw = args.questions.read_bytes()
    spec = json.loads(raw)
    checked = preflight(spec, args.clips, min_evaluation_clips=args.min_evaluation_clips)
    checked["questions_sha256"] = hashlib.sha256(raw).hexdigest()
    checked["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        args.output.write_text(json.dumps(checked, indent=2) + "\n")
        print(json.dumps({k: checked[k] for k in ("status", "clip_count", "question_count")}))
        return
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    client = httpx.Client(
        base_url=args.findback_url.rstrip("/") + "/api", headers=headers, timeout=120
    )
    capabilities = client.get("/v1/capabilities").raise_for_status().json()
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "questions_sha256": hashlib.sha256(raw).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "backend": args.backend,
        "identity_scoring_limit": "Coarse status and region labels do not adjudicate physical target identity; inspect cited source frames separately.",
        "capabilities": {k: capabilities.get(k) for k in ("version", "sam2_model", "identity")},
        "preflight": checked,
        "clips": [],
    }
    for clip in spec["clips"]:
        clip_started = time.monotonic()
        path = args.clips / clip["file"]
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != clip["sha256"]:
                raise ValueError(f"Clip {clip['file']} changed after preflight.")
            stream.seek(0)
            video = (
                client.post(
                    "/v1/videos",
                    files={"file": (clip["file"], stream, MEDIA_TYPES[path.suffix.lower()])},
                )
                .raise_for_status()
                .json()
            )
        upload_seconds = round(time.monotonic() - clip_started, 3)
        objects = {}
        registration_started = time.monotonic()
        for item in clip["objects"]:
            x1, y1, x2, y2 = item["box"]
            created = (
                client.post(
                    f"/v1/videos/{video['id']}/objects",
                    json={
                        "name": item["name"],
                        "label": "custom",
                        "at_ms": item["at_ms"],
                        "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    },
                )
                .raise_for_status()
                .json()
            )
            objects[item["name"]] = created["id"]
        registration_seconds = round(time.monotonic() - registration_started, 3)
        started = time.monotonic()
        run = (
            client.post(f"/v1/videos/{video['id']}/runs", json={"backend": args.backend})
            .raise_for_status()
            .json()
        )
        while run["status"] not in {"complete", "failed", "cancelled"}:
            time.sleep(1)
            run = client.get(f"/v1/runs/{run['id']}").json()
        index_seconds = round(time.monotonic() - started, 1)
        rows = []
        for question in clip["questions"]:
            answer: dict = {}
            question_started = time.monotonic()
            if run["status"] == "complete":
                response = client.post(
                    f"/v1/runs/{run['id']}/questions",
                    json={
                        "text": f"Where was {question['object']} last seen?",
                        "at_ms": question["at_ms"],
                        "object_id": objects[question["object"]],
                        "use_provider": False,
                    },
                )
                answer = response.json()
                if response.is_error:
                    answer = {"error": f"HTTP {response.status_code}: {answer.get('detail')}"}
            else:
                answer = {"error": f"Index run {run['status']}: {run.get('error') or 'no detail'}"}
            question_seconds = round(time.monotonic() - question_started, 3)
            state = object_state(answer, objects[question["object"]])
            checks = grade(question, state, spec["tolerance_ms"])
            rows.append(
                {
                    **question,
                    "answer": answer.get("answer"),
                    "error": answer.get("error"),
                    "status_given": state.get("status"),
                    "evidence": state.get("evidence"),
                    "question_seconds": question_seconds,
                    **checks,
                    **classify_result(question, state, checks, error=answer.get("error")),
                }
            )
        report["clips"].append(
            {
                "file": clip["file"],
                "split": clip["split"],
                "video_id": video["id"],
                "run": {
                    k: run.get(k) for k in ("id", "status", "backend", "observation_count", "error")
                },
                "index_seconds": index_seconds,
                "upload_seconds": upload_seconds,
                "registration_seconds": registration_seconds,
                "setup_and_index_seconds": round(
                    upload_seconds + registration_seconds + index_seconds, 3
                ),
                "total_task_seconds": round(time.monotonic() - clip_started, 3),
                "rows": rows,
            }
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "clip": clip["file"],
                    "run": run["status"],
                    "index_seconds": index_seconds,
                    "correct": f"{sum(r['correct'] for r in rows)}/{len(rows)}",
                }
            ),
            flush=True,
        )
    for split in ("development", "evaluation"):
        rows = [r for c in report["clips"] if c["split"] == split for r in c["rows"]]
        report[f"{split}_correct"] = f"{sum(r['correct'] for r in rows)}/{len(rows)}"
        report[f"{split}_counts"] = {
            key: sum(bool(row[key]) for row in rows)
            for key in (
                "unanswered",
                "false_visible",
                "wrong_zone",
                "wrong_last_seen_time",
                "unsupported_location_claim",
            )
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("development_correct", "evaluation_correct")}))


if __name__ == "__main__":
    main()
