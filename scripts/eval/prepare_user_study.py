"""Counterbalance and score a small manual-scrubbing versus FindBack study.

The schedule is built only from human-frozen real-clip labels. It contains no gold
answers. The blank outcome form is filled by the study operator after each attempt;
all attempt times include registration and indexing when FindBack is used. Keep
schedule, outcomes and reports under ignored ``artifacts/`` until reviewed.
"""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path

METHODS = ("manual", "findback")
STATES = {"visible", "last_seen", "unknown", "not_observed", "no_answer"}
ZONES = {"left", "center", "right"}
# First method, and which of a pair's two clips is done manually. For six
# participants, the two extra patterns preserve 3/3 balance on both margins.
PATTERNS = (
    ("manual", 0),
    ("manual", 1),
    ("findback", 0),
    ("findback", 1),
    ("manual", 0),
    ("findback", 1),
)


def build_schedule(spec: dict, pairs: dict, participants: int) -> list[dict]:
    """Cross method order and clip assignment without exposing answer labels."""
    if not 4 <= participants <= 6:
        raise ValueError("The study needs four to six anonymized participants.")
    available = {clip["file"]: clip for clip in spec["clips"]}
    definitions = pairs.get("pairs") if isinstance(pairs, dict) else None
    if not isinstance(definitions, list) or len(definitions) < 2:
        raise ValueError("Pair at least four evaluation clips in two task pairs.")
    used_files: set[str] = set()
    used_ids: set[str] = set()
    selected = []
    statuses = set()
    for pair in definitions:
        pair_id = pair.get("id") if isinstance(pair, dict) else None
        tasks = pair.get("tasks") if isinstance(pair, dict) else None
        reason = pair.get("match_rationale") if isinstance(pair, dict) else None
        if not isinstance(pair_id, str) or not pair_id or pair_id in used_ids:
            raise ValueError("Every task pair needs a unique nonempty id.")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Pair {pair_id} needs a human match rationale.")
        if not isinstance(tasks, list) or len(tasks) != 2:
            raise ValueError(f"Pair {pair_id} needs exactly two clip questions.")
        used_ids.add(pair_id)
        chosen = []
        for task in tasks:
            filename = task.get("file") if isinstance(task, dict) else None
            number = task.get("question_index") if isinstance(task, dict) else None
            if not isinstance(filename, str) or filename not in available:
                raise ValueError(f"Pair {pair_id} names no frozen clip: {filename}")
            if filename in used_files:
                raise ValueError(f"Study clip is repeated: {filename}")
            clip = available[filename]
            if clip["split"] != "evaluation":
                raise ValueError(f"Study clip must be reserved for evaluation: {filename}")
            questions = clip["questions"]
            if type(number) is not int or not 0 <= number < len(questions):
                raise ValueError(f"Pair {pair_id} has an invalid question index for {filename}.")
            question = questions[number]
            chosen.append(
                {
                    "file": filename,
                    "sha256": clip["sha256"],
                    "question_index": number,
                    "text": f"Where was {question['object']} last seen?",
                    "cutoff_ms": question["at_ms"],
                    "status_for_validation": question["status"],
                }
            )
            used_files.add(filename)
        if chosen[0]["status_for_validation"] != chosen[1]["status_for_validation"]:
            raise ValueError(f"Pair {pair_id} mixes different expected states.")
        statuses.add(chosen[0]["status_for_validation"])
        selected.append((pair_id, chosen))
    if "visible" not in statuses or not statuses.intersection(
        {"last_seen", "unknown", "not_observed"}
    ):
        raise ValueError("Include one visible pair and one absence or uncertainty pair.")

    assignments = []
    for participant in range(participants):
        first, manual_clip = PATTERNS[participant]
        ordered_pairs = selected if participant % 2 == 0 else list(reversed(selected))
        task_order = 0
        for pair_id, tasks in ordered_pairs:
            for method in (first, next(other for other in METHODS if other != first)):
                task = tasks[manual_clip if method == "manual" else 1 - manual_clip]
                task_order += 1
                assignments.append(
                    {
                        "participant_id": f"P{participant + 1:02d}",
                        "order": task_order,
                        "pair_id": pair_id,
                        "method": method,
                        "clip_file": task["file"],
                        "clip_sha256": task["sha256"],
                        "question_index": task["question_index"],
                        "question": task["text"],
                        "cutoff_ms": task["cutoff_ms"],
                    }
                )
    return assignments


def blank_outcomes(assignments: list[dict]) -> list[dict]:
    return [
        {
            "participant_id": task["participant_id"],
            "order": task["order"],
            "pair_id": task["pair_id"],
            "method": task["method"],
            "clip_file": task["clip_file"],
            "question_index": task["question_index"],
            "total_seconds": None,
            "setup_seconds": None,
            "status": None,
            "zone": None,
            "evidence_at_ms": None,
            "notes": "",
        }
        for task in assignments
    ]


def make_task_media(assignments: list[dict], clips_dir: Path, output_dir: Path) -> None:
    """Give both arms identical video with no source frame after the frozen cutoff."""
    from agentx.vision.video import ffmpeg_executable, probe

    executable = ffmpeg_executable()
    if executable is None:
        raise ValueError("FFmpeg is required to make cutoff-limited study media.")
    destination = output_dir / "task-media"
    destination.mkdir()
    prepared: dict[tuple[str, int], dict] = {}
    for task in assignments:
        key = (task["clip_file"], task["cutoff_ms"])
        if key not in prepared:
            source = clips_dir / task["clip_file"]
            target = destination / f"clip-{len(prepared) + 1:02d}-to-{task['cutoff_ms']}ms.mp4"
            cutoff_seconds = task["cutoff_ms"] / 1000
            video_filter = (
                f"setpts=PTS-STARTPTS,select='lte(t,{cutoff_seconds:.3f})',setpts=PTS-STARTPTS"
            )
            subprocess.run(
                [
                    executable,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source),
                    "-vf",
                    video_filter,
                    "-an",
                    "-fps_mode",
                    "vfr",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-preset",
                    "fast",
                    "-pix_fmt",
                    "yuv420p",
                    str(target),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            info = probe(target)
            with target.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            prepared[key] = {
                "path": str(target.relative_to(output_dir)),
                "sha256": digest,
                "duration_ms": info.duration_ms,
                "source_cutoff_ms": task["cutoff_ms"],
                "audio_removed": True,
            }
        task["task_media"] = prepared[key]


def score(schedule: dict, outcomes: list[dict], spec: dict) -> dict:
    """Keep every assigned attempt; an incomplete form cannot yield a result."""
    from evaluate_real_clips import grade

    assignments = schedule["assignments"]
    expected = {(row["participant_id"], row["order"]): row for row in assignments}
    if len(expected) != len(assignments) or len(outcomes) != len(assignments):
        raise ValueError("Outcome count does not match the frozen schedule.")
    clips = {clip["file"]: clip for clip in spec["clips"]}
    seen = set()
    rows = []
    for outcome in outcomes:
        key = (outcome.get("participant_id"), outcome.get("order"))
        assignment = expected.get(key)
        if assignment is None or key in seen:
            raise ValueError("Outcome has an extra or duplicate study assignment.")
        seen.add(key)
        for field in ("pair_id", "method", "clip_file", "question_index"):
            if outcome.get(field) != assignment[field]:
                raise ValueError(f"Outcome changed the assigned {field} for {key}.")
        total = outcome.get("total_seconds")
        setup = outcome.get("setup_seconds")
        if (
            type(total) not in (int, float)
            or type(setup) not in (int, float)
            or not math.isfinite(total)
            or not math.isfinite(setup)
            or not 0 <= setup <= total
        ):
            raise ValueError(f"Outcome {key} needs valid total and setup seconds.")
        status = outcome.get("status")
        zone = outcome.get("zone")
        at_ms = outcome.get("evidence_at_ms")
        if (
            not isinstance(status, str)
            or status not in STATES
            or (zone is not None and (not isinstance(zone, str) or zone not in ZONES))
        ):
            raise ValueError(f"Outcome {key} has an invalid answer state or zone.")
        if at_ms is not None and (type(at_ms) is not int or at_ms < 0):
            raise ValueError(f"Outcome {key} has an invalid evidence time.")
        if status == "no_answer" and (zone is not None or at_ms is not None):
            raise ValueError(f"No-answer outcome {key} must not assert a location.")
        clip = clips[assignment["clip_file"]]
        question = clip["questions"][assignment["question_index"]]
        if (
            assignment["clip_sha256"] != clip["sha256"]
            or assignment["cutoff_ms"] != question["at_ms"]
            or assignment["question"] != f"Where was {question['object']} last seen?"
        ):
            raise ValueError(f"Schedule question or source changed for {key}.")
        evidence = {"zone": zone, "at_ms": at_ms}
        checks = grade(question, {"status": status, "evidence": evidence}, spec["tolerance_ms"])
        causal_violation = at_ms is not None and at_ms > assignment["cutoff_ms"]
        refusal_with_location = status in {"unknown", "not_observed"} and (
            zone is not None or at_ms is not None
        )
        gold = question["status"]
        positive = status in {"visible", "last_seen"}
        wrong_zone = (
            positive and gold in {"visible", "last_seen"} and zone not in question["expected_zones"]
        )
        unsupported_time = status == "last_seen" and gold == "last_seen" and not checks["time_ok"]
        unsupported_location = bool(
            wrong_zone
            or unsupported_time
            or (status == "visible" and gold != "visible")
            or (status == "last_seen" and gold in {"unknown", "not_observed"})
            or causal_violation
            or refusal_with_location
        )
        rows.append(
            {
                **assignment,
                "total_seconds": total,
                "setup_seconds": setup,
                "status_given": status,
                "zone_given": zone,
                "evidence_at_ms_given": at_ms,
                "gold_status": gold,
                "correct": checks["correct"] and not causal_violation and not refusal_with_location,
                "causal_time_violation": causal_violation,
                "unsupported_location_claim": unsupported_location,
                "notes": outcome.get("notes", ""),
            }
        )
    by_method = {}
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        by_method[method] = {
            "attempts": len(subset),
            "correct": sum(row["correct"] for row in subset),
            "unsupported_location_claims": sum(row["unsupported_location_claim"] for row in subset),
            "median_total_seconds": statistics.median(row["total_seconds"] for row in subset),
            "median_setup_seconds": statistics.median(row["setup_seconds"] for row in subset),
        }
    participant_totals = []
    for participant in sorted({row["participant_id"] for row in rows}):
        totals = {
            method: round(
                sum(
                    row["total_seconds"]
                    for row in rows
                    if row["participant_id"] == participant and row["method"] == method
                ),
                3,
            )
            for method in METHODS
        }
        participant_totals.append({"participant_id": participant, **totals})
    return {
        "status": "complete",
        "scope": "Small counterbalanced participant study; descriptive counts only.",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "questions_sha256": schedule["questions_sha256"],
        "pairs_sha256": schedule["pairs_sha256"],
        "participants": len(participant_totals),
        "by_method": by_method,
        "participant_totals": participant_totals,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("schedule")
    setup.add_argument("--questions", required=True, type=Path)
    setup.add_argument("--clips", required=True, type=Path)
    setup.add_argument("--pairs", required=True, type=Path)
    setup.add_argument("--participants", required=True, type=int)
    setup.add_argument("--output", required=True, type=Path)
    setup.add_argument("--min-evaluation-clips", type=int, default=7)
    scoring = commands.add_parser("score")
    scoring.add_argument("--schedule", required=True, type=Path)
    scoring.add_argument("--questions", required=True, type=Path)
    scoring.add_argument("--pairs", required=True, type=Path)
    scoring.add_argument("--outcomes", required=True, type=Path)
    scoring.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.command == "schedule":
        from evaluate_real_clips import preflight

        if args.output.exists():
            raise SystemExit("Choose a new schedule directory; preserve earlier assignments.")
        question_bytes = args.questions.read_bytes()
        pair_bytes = args.pairs.read_bytes()
        spec = json.loads(question_bytes)
        preflight(spec, args.clips, min_evaluation_clips=args.min_evaluation_clips)
        assignments = build_schedule(spec, json.loads(pair_bytes), args.participants)
        args.output.mkdir(parents=True)
        make_task_media(assignments, args.clips, args.output)
        schedule = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "scope": "Blinded assignment; gold labels remain only in the frozen question file.",
            "questions_sha256": hashlib.sha256(question_bytes).hexdigest(),
            "pairs_sha256": hashlib.sha256(pair_bytes).hexdigest(),
            "assignments": assignments,
        }
        (args.output / "schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
        (args.output / "outcomes-blank.json").write_text(
            json.dumps(blank_outcomes(assignments), indent=2) + "\n"
        )
        print(json.dumps({"participants": args.participants, "assignments": len(assignments)}))
    else:
        if args.output.exists():
            raise SystemExit("Choose a new report path; preserve earlier study attempts.")
        question_bytes = args.questions.read_bytes()
        schedule = json.loads(args.schedule.read_text())
        if hashlib.sha256(question_bytes).hexdigest() != schedule["questions_sha256"]:
            raise ValueError("Frozen question file changed after scheduling.")
        pair_bytes = args.pairs.read_bytes()
        if hashlib.sha256(pair_bytes).hexdigest() != schedule["pairs_sha256"]:
            raise ValueError("Frozen task pairs changed after scheduling.")
        spec = json.loads(question_bytes)
        expected = build_schedule(
            spec,
            json.loads(pair_bytes),
            len({row["participant_id"] for row in schedule["assignments"]}),
        )
        for actual, original in zip(schedule["assignments"], expected, strict=True):
            if {key: value for key, value in actual.items() if key != "task_media"} != original:
                raise ValueError("Study schedule changed after generation.")
        checked_media = set()
        media_root = args.schedule.parent.resolve(strict=True)
        for assignment in schedule["assignments"]:
            media = assignment.get("task_media")
            if not isinstance(media, dict) or not isinstance(media.get("path"), str):
                raise ValueError("Assignment is missing its cutoff-limited media.")
            path = (media_root / media["path"]).resolve(strict=True)
            if not path.is_relative_to(media_root):
                raise ValueError("Manual study media escaped the schedule directory.")
            if path in checked_media:
                continue
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != media["sha256"]:
                    raise ValueError(f"Study media changed: {media['path']}")
            checked_media.add(path)
        report = score(schedule, json.loads(args.outcomes.read_text()), spec)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({"participants": report["participants"], "by_method": report["by_method"]})
        )


if __name__ == "__main__":
    main()
