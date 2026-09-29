"""Portable, cutoff-scoped reports built from saved answers and original source frames."""

import hashlib
import json
import threading
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from string import Template
from zipfile import ZIP_DEFLATED, ZipFile

from agentx.api.schemas import AnswerResponse
from agentx.domain.contracts import timestamp
from agentx.services import bundle_verify
from agentx.services.answering import AMBIGUOUS_MATCH, NO_MATCH, grounded_answer
from agentx.storage.models import IndexRun, QueryRecord, Video
from agentx.vision.video import jpeg, read_frame

MAX_FRAMES = 120
MAX_BYTES = 64 * 1024 * 1024
PUBLIC_PROVENANCE = {
    "pipeline_version",
    "adapter",
    "model",
    "revision",
    "weights_sha256",
    "device",
    "dtype",
    "torch",
    "transformers",
    "trackers",
    "opencv_version",
    "method",
    "learned_model",
    "identity",
    "identity_model",
    "identity_revision",
    "identity_transformers",
    "identity_threshold",
    "identity_margin",
    "identity_method",
    "identity_scope",
    "memory_frames",
    "presence_threshold",
    "presence_gate",
    "confident_presence",
    "appearance_threshold",
    "recent_accepted_views",
    "policy",
    "appearance",
    "embedding",
    "score_semantics",
    "prompt",
    "model_downloads_at_runtime",
    "backend",
    "max_edge",
    "reference_crop",
    "prompt_sha256",
    "skill",
    "name",
    "sha256",
    "purpose",
    "input_sha256",
}


class BundleLimitError(ValueError):
    pass


class BundleBusyError(ValueError):
    pass


def public_provenance(value):
    """Export only known model/measurement metadata, excluding host paths and endpoints."""
    if isinstance(value, dict):
        return {k: public_provenance(v) for k, v in value.items() if k in PUBLIC_PROVENANCE}
    if isinstance(value, list):
        return [public_provenance(v) for v in value]
    return value


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


class EvidenceBundles:
    def __init__(self, db, catalog, memory):
        self.db, self.catalog, self.memory = db, catalog, memory
        self.lock = threading.Lock()

    def build(self, question_id: str, destination: Path) -> dict:
        if not self.lock.acquire(blocking=False):
            raise BundleBusyError("Another evidence report is being prepared. Try again shortly.")
        try:
            return self._build(question_id, destination)
        finally:
            self.lock.release()

    def _build(self, question_id: str, destination: Path) -> dict:
        with self.db.session() as session:
            record = session.get(QueryRecord, question_id)
            if record is None:
                raise LookupError("Saved answer not found.")
            run = session.get(IndexRun, record.run_id)
            if run.status != "complete":
                raise ValueError("Finish this memory version before exporting its evidence.")
            video = session.get(Video, run.video_id)
            answer = AnswerResponse.model_validate(record.response).model_dump(mode="json")
        if (
            answer["id"] != record.id
            or answer["run_id"] != run.id
            or answer["as_of_ms"] != record.cutoff_ms
            or answer["question"] != record.question
        ):
            raise ValueError("The saved answer no longer matches its source record.")
        _, _, objects, cutoff = self.memory.context(run.id, record.cutoff_ms)
        names = {o.id: o for o in objects}
        # A report is not a way to re-publish unverified legacy planner prose. The
        # immutable memory must still support exactly the saved claims and citations.
        if answer["states"]:
            obj = names.get(answer["states"][0]["object_id"])
            if obj is None:
                raise ValueError("The saved answer refers to an object outside this memory.")
            expected = grounded_answer(self.memory, run, video, obj, cutoff, answer["intent"])
            if any(
                answer[key] != expected[key] for key in ("answer", "states", "events", "evidence")
            ):
                raise ValueError("This saved answer cannot be revalidated. Ask the question again.")
        elif answer["answer"] not in {NO_MATCH, AMBIGUOUS_MATCH} or any(
            answer[key] for key in ("evidence", "events")
        ):
            raise ValueError("This saved answer cannot be revalidated. Ask the question again.")

        source = self.catalog.path(video.source_path)
        if not source.is_file():
            raise LookupError("Original evidence is missing.")
        with source.open("rb") as stream:
            source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if source_hash != video.sha256:
            raise ValueError("The original video changed. Evidence export was refused.")
        unique_times = sorted({e["at_ms"] for e in answer["evidence"]})
        context = None
        if answer["evidence"] and answer["states"][0]["status"] != "visible":
            context_ms, context_image = read_frame(source, cutoff)
            if not 0 <= context_ms <= cutoff:
                raise ValueError("The context frame is outside the answer cutoff.")
            context = (context_ms, jpeg(context_image))
        if len(set(unique_times) | ({context[0]} if context else set())) > MAX_FRAMES:
            raise BundleLimitError(
                "This answer cites more than 120 frames. Ask with an earlier cutoff."
            )
        report: dict = {
            "schema_version": "1.0",
            "created_at": datetime.now(UTC).isoformat(),
            "source": {
                "id": video.id,
                "title": video.title,
                "sha256": source_hash,
                "duration_ms": video.duration_ms,
                "width": video.width,
                "height": video.height,
                "is_fixture": video.is_fixture,
            },
            "run": {
                "id": run.id,
                "backend": run.backend,
                "created_at": run.created_at.isoformat(),
                "config": run.config,
                "regions": run.regions,
                "provenance": public_provenance(run.provenance),
            },
            "answer": answer,
            "frames": [],
            "context_frame": None,
            "scope": "One saved answer, its cited source frames, optional cutoff context for an uncertain location, and its memory version. No later frames, full video, or separate Cosmos reviews are included.",
            "limitations": "Memory claims depend on perception accuracy. Boxes and model scores are not proof of object identity. Planner trace text is a recorded model draft, not visual evidence. File hashes establish integrity against this manifest, not authorship or visual correctness.",
        }
        entries: dict = {}
        size = 0
        with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:

            def add(name: str, data: bytes):
                nonlocal size
                size += len(data)
                if size > MAX_BYTES:
                    raise BundleLimitError("The report exceeds 64 MiB. Ask with an earlier cutoff.")
                archive.writestr(name, data)
                entries[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

            for at_ms in unique_times:
                actual_ms, image = read_frame(source, at_ms)
                if actual_ms != at_ms or actual_ms > cutoff:
                    raise ValueError("A source frame no longer matches its observation timestamp.")
                name = f"frames/{at_ms:010d}.jpg"
                data = jpeg(image)
                add(name, data)
                report["frames"].append(
                    {"at_ms": at_ms, "file": name, "sha256": entries[name]["sha256"]}
                )

            if context:
                at_ms, data = context
                name = f"frames/{at_ms:010d}.jpg"
                if name not in entries:
                    add(name, data)
                    report["frames"].append(
                        {"at_ms": at_ms, "file": name, "sha256": entries[name]["sha256"]}
                    )
                report["context_frame"] = {
                    "at_ms": at_ms,
                    "requested_ms": cutoff,
                    "file": name,
                    "role": "context_only",
                    "sha256": entries[name]["sha256"],
                }

            # Canonical facts have been checked against storage. Replace live server
            # links with portable files; no playback link can reveal future footage.
            for ref in answer["evidence"] + [
                s["evidence"] for s in answer["states"] if s["evidence"]
            ]:
                ref["frame_url"] = f"frames/{ref['at_ms']:010d}.jpg"
                ref.pop("media_url", None)
            add("report.json", json_bytes(report))
            add("report.html", render_report(report).encode())
            add("verify.py", Path(bundle_verify.__file__).read_bytes())
            add(
                "README.txt",
                (
                    "AgentX FindBack — offline evidence report\n\n"
                    "Extract the ZIP, then open report.html in a browser. No server, network, or model is required.\n"
                    "Frames are JPEG encodings decoded from the fingerprinted original video; boxes are separate overlays.\n"
                    "The images contain the full source frame, including surrounding scene content.\n"
                    "report.json holds the machine-readable answer, evidence, Skills and tool trace.\n\n"
                    "Verify all files: python3 verify.py\n"
                    "Also check your original: python3 verify.py --source /path/to/original.mp4\n"
                    "Python 3.10 or newer; no packages required.\n"
                    "Integrity is relative to manifest.json. This is not a digital signature or an accuracy score.\n"
                ).encode(),
            )
            archive.writestr(
                "manifest.json",
                json_bytes(
                    {
                        "format": "agentx-evidence-bundle-v1",
                        "source_sha256": source_hash,
                        "files": entries,
                    }
                ),
            )
        return {"frames": len(report["frames"]), "bytes": destination.stat().st_size}


def render_report(report: dict) -> str:
    answer, source, run = report["answer"], report["source"], report["run"]
    regions = {r["id"]: r["name"] for r in run["regions"]}
    # A region read through a recovered camera pose is a different kind of claim from one read
    # directly, and an offline reader has no other way to tell them apart.
    camera_notes = {
        "compensated": "Camera moved · region recovered from the registration view",
        "unavailable": "Camera moved · no region could be named for this frame",
    }
    evidence = []
    for number, ref in enumerate(sorted(answer["evidence"], key=lambda e: e["at_ms"]), start=1):
        box = ref["box"]
        style = f"left:{box['x1'] * 100}%;top:{box['y1'] * 100}%;width:{(box['x2'] - box['x1']) * 100}%;height:{(box['y2'] - box['y1']) * 100}%"
        note = camera_notes.get(ref.get("scene_reference", "registered"), "")
        badge = f'<span class="camera">{escape(note)}</span>' if note else ""
        evidence.append(
            f'<figure><div class="frame"><img src="{escape(ref["frame_url"], quote=True)}" '
            f'alt="Source evidence at {timestamp(ref["at_ms"])}"><span class="box" style="{style}"></span>'
            f"{badge}</div>"
            f'<figcaption><span class="time">{timestamp(ref["at_ms"])}</span>'
            f"<span>{escape(regions.get(ref['zone'], 'Unassigned area'))}</span>"
            f'<a href="{escape(ref["frame_url"], quote=True)}">Open original frame ↗</a></figcaption>'
            f'<p class="small">Evidence {number} · Observation {escape(ref["observation_id"])}</p></figure>'
        )
    context = report["context_frame"]
    if context:
        evidence.append(
            f'<figure class="context-frame"><div class="frame"><img src="{escape(context["file"], quote=True)}" '
            f'alt="Context near the requested cutoff at {timestamp(context["at_ms"])}"></div>'
            f'<figcaption><span class="time">{timestamp(context["at_ms"])}</span><span>At query cutoff · context only</span>'
            f'<a href="{escape(context["file"], quote=True)}">Open original frame ↗</a></figcaption>'
            '<p class="small">For your inspection. This frame is not a confirmed sighting of the selected object.</p></figure>'
        )
    events = "".join(
        f"<tr><td>{timestamp(e['at_ms'])}</td><td>{escape(e['name'])}</td><td>{escape(e['kind'])}</td>"
        f"<td>{escape(regions.get(e['zone'], '—'))}</td></tr>"
        for e in answer["events"]
    )
    history = (
        '<section><h2>Recorded changes</h2><div class="table-wrap"><table><thead><tr><th>Time</th>'
        "<th>Object</th><th>Event</th><th>Region</th></tr></thead><tbody>"
        + events
        + "</tbody></table></div></section>"
        if events
        else ""
    )
    statuses = {
        "visible": "Visible at cutoff",
        "last_seen": "Last seen · position unconfirmed",
        "unknown": "Uncertain",
        "not_observed": "Not yet observed",
    }
    state = answer["states"][0] if answer["states"] else None
    values = {
        "title": escape(source["title"]),
        "question": escape(answer["question"]),
        "answer": escape(answer["answer"]),
        "cutoff": timestamp(answer["as_of_ms"]),
        "status": escape(
            statuses.get(state["status"], state["status"]) if state else "No matched object"
        ),
        "frame_count": str(len(report["frames"])),
        "backend": escape(run["backend"]),
        "fixture": '<p class="notice">Generated software fixture · demonstrates behavior, not real-world accuracy.</p>'
        if source["is_fixture"]
        else "",
        "evidence": "".join(evidence)
        or '<p class="notice">No visual evidence is attached to this answer.</p>',
        "history": history,
        "trace": escape(
            json.dumps(
                {
                    k: answer[k]
                    for k in (
                        "planner",
                        "planner_model",
                        "elapsed_seconds",
                        "skills",
                        "tools",
                        "warnings",
                    )
                },
                ensure_ascii=False,
                indent=2,
            )
        ),
        "provenance": escape(json.dumps(run, ensure_ascii=False, indent=2)),
        "source_hash": escape(source["sha256"]),
        "created_at": escape(report["created_at"]),
        "scope": escape(report["scope"]),
        "limitations": escape(report["limitations"]),
    }
    return Template(Path(__file__).with_name("evidence_report.html").read_text()).substitute(values)
