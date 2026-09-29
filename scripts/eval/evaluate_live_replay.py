"""Replay frozen real clips through FindBack's live capture and ask their questions while recording.

Each clip is decoded at the capture rate and pushed frame by frame at its own real-time pace to
`POST /api/v1/live/{id}/frames`, exactly as a camera would feed it. The server stamps every frame
on arrival. When the stream reaches a frozen registration time, the objects are registered on
that live frame and a memory run starts following the capture. When the stream reaches a frozen
question cutoff, the evaluator waits only until memory has processed that frame and asks
"Where was <object> last seen?" with the deterministic resolver — during the recording, not
after it.

The answers are graded with the unchanged rules of `evaluate_real_clips.py`, after mapping live
timestamps back to source timestamps through the frames the evaluator itself sent. After the
clip ends the capture is sealed; every question is asked again at the same live cutoff on the
completed memory, and (with --rebuild) a second memory is built offline from the sealed file and
compared observation by observation.

Timing receipts: for every sent frame, the wall-clock delay from the server acknowledging the
frame to that frame being in queryable memory, and for every question the delay from its cutoff
frame's arrival to the answer. These are single-client measurements on the named hardware.
"""

import argparse
import hashlib
import json
import statistics
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import httpx
from evaluate_real_clips import grade, object_state, preflight

from agentx.vision.video import frames


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * fraction // 1))
    return round(ordered[int(rank) - 1], 3)


def summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "median": round(statistics.median(values), 3) if values else None,
        "p95": percentile(values, 0.95),
        "max": round(max(values), 3) if values else None,
    }


class Pusher(threading.Thread):
    """Send a clip's sampled frames at their source pace and remember each server timestamp."""

    def __init__(self, client, video_id, path, fps, max_edge, quality):
        super().__init__(daemon=True)
        self.client, self.video_id, self.path = client, video_id, path
        self.fps, self.max_edge, self.quality = fps, max_edge, quality
        self.sent: list[dict] = []  # source_ms, live_ms, acked_at (monotonic)
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.error: str | None = None

    def run(self):
        try:
            started = None
            for source_ms, image in frames(self.path, self.fps):
                if started is None:
                    started = time.monotonic() - source_ms / 1000
                delay = started + source_ms / 1000 - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                scale = min(1.0, self.max_edge / max(image.shape[:2]))
                if scale < 1:
                    image = cv2.resize(
                        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                    )
                ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if not ok:
                    raise ValueError(f"Could not encode the frame at {source_ms} ms.")
                response = self.client.post(
                    f"/v1/live/{self.video_id}/frames",
                    content=data.tobytes(),
                    headers={"Content-Type": "image/jpeg"},
                )
                response.raise_for_status()
                result = response.json()
                if result["accepted"]:
                    with self.lock:
                        self.sent.append(
                            {
                                "source_ms": source_ms,
                                "live_ms": result["at_ms"],
                                "acked_at": time.monotonic(),
                            }
                        )
        except Exception as exc:  # reported, never retried
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.done.set()

    def frame_at_or_before(self, source_ms: int) -> dict | None:
        """The newest sent frame whose source time is at or before `source_ms`, once a later
        frame proves no closer one will follow."""
        with self.lock:
            if not self.sent or (self.sent[-1]["source_ms"] < source_ms and not self.done.is_set()):
                return None
            eligible = [row for row in self.sent if row["source_ms"] <= source_ms]
            return eligible[-1] if eligible else None


class Watcher(threading.Thread):
    """Sample how far memory has processed, to time each frame's arrival in memory."""

    def __init__(self, client, run_id):
        super().__init__(daemon=True)
        self.client, self.run_id = client, run_id
        self.samples: list[tuple[float, int]] = []
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                run = self.client.get(f"/v1/runs/{self.run_id}").json()
                self.samples.append((time.monotonic(), run["processed_ms"]))
                if run["status"] in {"complete", "failed", "cancelled"}:
                    return
            except httpx.HTTPError:
                pass
            self.stop.wait(0.05)

    def processed_at(self, live_ms: int) -> float | None:
        return next((when for when, done in self.samples if done >= live_ms), None)


def wait_processed(client, run_id, live_ms, timeout=120.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        run = client.get(f"/v1/runs/{run_id}").json()
        if run["processed_ms"] >= live_ms or run["status"] in {"complete", "failed", "cancelled"}:
            return run
        if time.monotonic() > deadline:
            raise TimeoutError(f"Memory did not reach {live_ms} ms within {timeout} s.")
        time.sleep(0.02)


def ask(client, run_id, question, object_id, live_cutoff) -> tuple[dict, float]:
    started = time.monotonic()
    response = client.post(
        f"/v1/runs/{run_id}/questions",
        json={
            "text": f"Where was {question['object']} last seen?",
            "at_ms": live_cutoff,
            "object_id": object_id,
            "use_provider": False,
        },
    )
    answer = response.json()
    if response.is_error:
        answer = {"error": f"HTTP {response.status_code}: {answer.get('detail')}"}
    return answer, time.monotonic() - started


def to_source(state: dict, live_to_source: dict[int, int]) -> dict:
    """The same state with its cited evidence time expressed in source-clip time."""
    evidence = state.get("evidence")
    if not evidence:
        return state
    return {
        **state,
        "evidence": {**evidence, "at_ms": live_to_source.get(evidence["at_ms"])},
    }


def graded_row(question, answer, object_id, live_to_source, tolerance, extra) -> dict:
    live_state = object_state(answer, object_id)
    state = to_source(live_state, live_to_source)
    return {
        **question,
        **extra,
        "answer": answer.get("answer"),
        "error": answer.get("error"),
        "warnings": answer.get("warnings"),
        "status_given": state.get("status"),
        "evidence_live_ms": (live_state.get("evidence") or {}).get("at_ms"),
        "evidence": state.get("evidence"),
        **grade(question, state, tolerance),
    }


def observations(client, run_id) -> list[tuple]:
    rows = client.get(f"/v1/runs/{run_id}/export").json()["observations"]
    keys = ("object_id", "at_ms", "box", "zone", "reason", "visible")
    return sorted(tuple(json.dumps(row[k], sort_keys=True) for k in keys) for row in rows)


def replay_clip(client, clip, path, args, tolerance) -> dict:
    video = client.post(
        "/v1/live",
        json={"title": f"Live replay · {clip['file']}", "fps": args.fps, "source": "push"},
    )
    video = video.raise_for_status().json()
    pusher = Pusher(client, video["id"], path, args.fps, args.max_edge, args.jpeg_quality)
    pusher.start()
    registration_ms = max(item["at_ms"] for item in clip["objects"])
    while (frame := pusher.frame_at_or_before(registration_ms)) is None:
        if pusher.done.is_set() and pusher.error:
            raise RuntimeError(pusher.error)
        time.sleep(0.01)
    # The registration frame must be readable; the capture lags its newest frame by one.
    while client.get(f"/v1/videos/{video['id']}").json()["duration_ms"] <= frame["live_ms"]:
        time.sleep(0.01)
    objects = {}
    registrations = []
    for item in clip["objects"]:
        own = pusher.frame_at_or_before(item["at_ms"])
        x1, y1, x2, y2 = item["box"]
        created = client.post(
            f"/v1/videos/{video['id']}/objects",
            json={
                "name": item["name"],
                "label": "custom",
                "at_ms": own["live_ms"],
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            },
        )
        created = created.raise_for_status().json()
        objects[item["name"]] = created["id"]
        registrations.append(
            {
                "name": item["name"],
                "source_ms": own["source_ms"],
                "live_ms": created["registered_at_ms"],
            }
        )
    run = client.post(f"/v1/videos/{video['id']}/runs", json={"backend": args.backend})
    run = run.raise_for_status().json()
    watcher = Watcher(client, run["id"])
    watcher.start()

    ordered = sorted(clip["questions"], key=lambda q: q["at_ms"])
    live_rows = []
    sealed: dict | None = None

    def seal() -> dict:
        nonlocal sealed
        if sealed is None:
            pusher.join()
            if pusher.error:
                raise RuntimeError(pusher.error)
            sealed = client.post(f"/v1/live/{video['id']}/stop").raise_for_status().json()
        return sealed

    for question in ordered:
        while (frame := pusher.frame_at_or_before(question["at_ms"])) is None:
            if pusher.done.is_set() and pusher.error:
                raise RuntimeError(pusher.error)
            time.sleep(0.005)
        # The newest frame of a capture becomes readable when the next one arrives. A cutoff
        # on the clip's final frame therefore cannot be answered until the capture is sealed.
        deadline = time.monotonic() + 120
        while True:
            state = client.get(f"/v1/runs/{run['id']}").json()
            if state["processed_ms"] >= frame["live_ms"] or state["status"] not in {
                "queued",
                "running",
            }:
                break
            if sealed is None and pusher.done.is_set() and frame is pusher.sent[-1]:
                seal()
            if time.monotonic() > deadline:
                raise TimeoutError(f"Memory did not reach {frame['live_ms']} ms within 120 s.")
            time.sleep(0.02)
        in_memory = time.monotonic()
        answer, answer_seconds = ask(
            client, run["id"], question, objects[question["object"]], frame["live_ms"]
        )
        answered = time.monotonic()
        live_to_source = {row["live_ms"]: row["source_ms"] for row in pusher.sent}
        live_rows.append(
            graded_row(
                question,
                answer,
                objects[question["object"]],
                live_to_source,
                tolerance,
                {
                    # "stream_ended": the clip had ended but the capture was not yet sealed.
                    "asked_while": "after_seal"
                    if sealed is not None
                    else "recording"
                    if not pusher.done.is_set()
                    else "stream_ended",
                    "run_status_when_asked": state["status"],
                    "cutoff_source_ms": frame["source_ms"],
                    "cutoff_live_ms": frame["live_ms"],
                    "frame_to_memory_seconds": round(in_memory - frame["acked_at"], 3),
                    "answer_seconds": round(answer_seconds, 3),
                    "frame_to_answer_seconds": round(answered - frame["acked_at"], 3),
                },
            )
        )
    sealed = seal()
    final = wait_processed(client, run["id"], 10**9, timeout=600)
    watcher.stop.set()
    watcher.join(timeout=5)
    live_to_source = {row["live_ms"]: row["source_ms"] for row in pusher.sent}

    sealed_rows = []
    for question, row in zip(ordered, live_rows, strict=True):
        answer, _ = ask(client, run["id"], question, objects[row["object"]], row["cutoff_live_ms"])
        graded = graded_row(question, answer, objects[row["object"]], live_to_source, tolerance, {})
        graded["same_as_live"] = (
            graded["status_given"] == row["status_given"] and graded["evidence"] == row["evidence"]
        )
        sealed_rows.append(graded)

    memory_latency = [
        when - frame["acked_at"]
        for frame in pusher.sent
        if frame["live_ms"] >= min(r["live_ms"] for r in registrations)
        and (when := watcher.processed_at(frame["live_ms"])) is not None
    ]
    result = {
        "file": clip["file"],
        "split": clip["split"],
        "video_id": video["id"],
        "sealed_sha256": sealed["sha256"],
        "sealed_duration_ms": sealed["duration_ms"],
        "capture": sealed.get("capture"),
        "frames_sent": len(pusher.sent),
        "registrations": registrations,
        "run": {
            k: final.get(k)
            for k in ("id", "status", "backend", "observation_count", "elapsed_seconds", "error")
        },
        "frame_to_memory_seconds": summary(memory_latency),
        "live_rows": live_rows,
        "sealed_rows": sealed_rows,
        "live_to_source": [[row["live_ms"], row["source_ms"]] for row in pusher.sent],
    }
    if args.rebuild:
        rebuilt = client.post(f"/v1/videos/{video['id']}/runs", json={"backend": args.backend})
        rebuilt = rebuilt.raise_for_status().json()
        while rebuilt["status"] not in {"complete", "failed", "cancelled"}:
            time.sleep(0.5)
            rebuilt = client.get(f"/v1/runs/{rebuilt['id']}").json()
        live_obs, rebuilt_obs = observations(client, run["id"]), observations(client, rebuilt["id"])
        rebuilt_rows = []
        for question, row in zip(ordered, live_rows, strict=True):
            answer, _ = ask(
                client, rebuilt["id"], question, objects[row["object"]], row["cutoff_live_ms"]
            )
            rebuilt_rows.append(
                graded_row(question, answer, objects[row["object"]], live_to_source, tolerance, {})
            )
        result["rebuild"] = {
            "run": {k: rebuilt.get(k) for k in ("id", "status", "observation_count", "error")},
            "identical_observations": live_obs == rebuilt_obs,
            "live_observations": len(live_obs),
            "rebuilt_observations": len(rebuilt_obs),
            "only_live": len(set(live_obs) - set(rebuilt_obs)),
            "only_rebuilt": len(set(rebuilt_obs) - set(live_obs)),
            "rows": rebuilt_rows,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--clips", required=True, type=Path)
    parser.add_argument("--findback-url", default="http://127.0.0.1:9000")
    parser.add_argument("--token", default="")
    parser.add_argument("--backend", default="reference")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-edge", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Preserve first reports: choose a new output path.")
    raw = args.questions.read_bytes()
    spec = json.loads(raw)
    checked = preflight(spec, args.clips)
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
        "capture": {
            "fps": args.fps,
            "max_edge": args.max_edge,
            "jpeg_quality": args.jpeg_quality,
            "transport": "HTTP POST per frame, one client, loopback",
        },
        "capabilities": {k: capabilities.get(k) for k in ("version", "sam2_model", "identity")},
        "preflight": checked,
        "clips": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for clip in spec["clips"]:
        path = args.clips / clip["file"]
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != clip["sha256"]:
                raise ValueError(f"Clip {clip['file']} changed after preflight.")
        try:
            result = replay_clip(client, clip, path, args, spec["tolerance_ms"])
        except Exception as exc:
            result = {"file": clip["file"], "split": clip["split"], "error": repr(exc)}
        report["clips"].append(result)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "clip": clip["file"],
                    "error": result.get("error"),
                    "live_correct": f"{sum(r['correct'] for r in result.get('live_rows', []))}/"
                    f"{len(clip['questions'])}",
                    "frame_to_memory": result.get("frame_to_memory_seconds"),
                    "identical_rebuild": (result.get("rebuild") or {}).get(
                        "identical_observations"
                    ),
                }
            ),
            flush=True,
        )
    for split in ("development", "evaluation"):
        clips = [c for c in report["clips"] if c["split"] == split]
        expected = sum(len(c["questions"]) for c in spec["clips"] if c["split"] == split)
        for key in ("live_rows", "sealed_rows"):
            rows = [r for c in clips for r in c.get(key, [])]
            report[f"{split}_{key.replace('_rows', '')}_correct"] = (
                f"{sum(r['correct'] for r in rows)}/{expected}"
            )
        rebuilt = [r for c in clips for r in (c.get("rebuild") or {}).get("rows", [])]
        if args.rebuild:
            report[f"{split}_rebuild_correct"] = f"{sum(r['correct'] for r in rebuilt)}/{expected}"
    latency = [
        r[key]
        for c in report["clips"]
        for r in c.get("live_rows", [])
        for key in ("frame_to_answer_seconds",)
    ]
    report["frame_to_answer_seconds"] = summary(latency)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k.endswith("_correct")}))


if __name__ == "__main__":
    main()
