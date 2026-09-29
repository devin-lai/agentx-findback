"""Audit a FindBack evidence ZIP without extracting or executing files from it.

Python 3.10+ standard library only. File hashes establish integrity relative to
the archive's manifest; supply a trusted archive SHA-256 for origin checking.
"""

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile

MAX_FILES = 140
MAX_MANIFEST_BYTES = 256 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024 + MAX_MANIFEST_BYTES
MAX_REPORT_BYTES = 2 * 1024 * 1024
REQUIRED = {"report.html", "report.json", "verify.py", "README.txt"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class AuditError(ValueError):
    """An archive violates the FindBack evidence contract."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name)
        and not path.is_absolute()
        and "\\" not in name
        and all(part not in {"", ".", ".."} for part in name.split("/"))
    )


def read_limited(archive: ZipFile, name: str, limit: int) -> bytes:
    with archive.open(name) as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise AuditError(f"File exceeds audit limit: {name}")
    return data


def required_dict(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise AuditError(f"Invalid {label} object")
    return value


def sha256_value(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise AuditError(f"Invalid SHA-256 in {label}")
    return value


def audit(path: Path, source: Path | None = None, expected_sha256: str | None = None) -> dict:
    if expected_sha256 is not None:
        sha256_value(expected_sha256, "expected archive hash")
    archive_hash = digest_file(path)
    if expected_sha256 is not None and archive_hash != expected_sha256:
        raise AuditError("Archive SHA-256 does not match the trusted value")

    with ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_FILES:
            raise AuditError("Too many files in archive")
        if sum(info.file_size for info in infos) > MAX_TOTAL_BYTES:
            raise AuditError("Archive expands beyond the audit limit")
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise AuditError("Duplicate ZIP member names")
        for info in infos:
            mode = stat.S_IFMT(info.external_attr >> 16)
            if (
                not safe_name(info.filename)
                or info.is_dir()
                or (mode not in {0, stat.S_IFREG})
                or info.flag_bits & 1
                or info.compress_type not in {ZIP_STORED, ZIP_DEFLATED}
            ):
                raise AuditError(f"Unsafe ZIP member: {info.filename!r}")
        if "manifest.json" not in names:
            raise AuditError("Missing manifest.json")
        manifest = required_dict(
            json.loads(read_limited(archive, "manifest.json", MAX_MANIFEST_BYTES)), "manifest"
        )
        if manifest.get("format") != "agentx-evidence-bundle-v1":
            raise AuditError("Unsupported evidence bundle format")
        entries = required_dict(manifest.get("files"), "manifest files")
        if not REQUIRED <= entries.keys() or set(names) != set(entries) | {"manifest.json"}:
            raise AuditError("Manifest membership disagrees with ZIP members")
        for name in entries:
            if not isinstance(name, str) or not safe_name(name):
                raise AuditError("Unsafe manifest path")

        for name, raw in entries.items():
            metadata = required_dict(raw, f"manifest entry {name}")
            expected = sha256_value(metadata.get("sha256"), name)
            size = metadata.get("bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise AuditError(f"Invalid byte count: {name}")
            if archive.getinfo(name).file_size != size:
                raise AuditError(f"Byte count mismatch: {name}")
            digest = hashlib.sha256()
            count = 0
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    count += len(block)
                    if count > MAX_TOTAL_BYTES:
                        raise AuditError(f"File exceeds audit limit: {name}")
            if count != size or digest.hexdigest() != expected:
                raise AuditError(f"Integrity check failed: {name}")

        report = required_dict(
            json.loads(read_limited(archive, "report.json", MAX_REPORT_BYTES)), "report"
        )
        source_info = required_dict(report.get("source"), "source")
        run = required_dict(report.get("run"), "run")
        answer = required_dict(report.get("answer"), "answer")
        source_hash = sha256_value(source_info.get("sha256"), "source")
        if source_hash != manifest.get("source_sha256"):
            raise AuditError("Source fingerprints disagree")
        if source is not None and digest_file(source) != source_hash:
            raise AuditError("Original video does not match the report")
        cutoff = answer.get("as_of_ms")
        duration = source_info.get("duration_ms")
        if (
            isinstance(cutoff, bool)
            or not isinstance(cutoff, int)
            or isinstance(duration, bool)
            or not isinstance(duration, int)
            or not 0 <= cutoff <= duration
        ):
            raise AuditError("Answer cutoff is outside the source duration")
        if answer.get("run_id") != run.get("id"):
            raise AuditError("Answer and run identifiers disagree")

        frames = report.get("frames")
        if not isinstance(frames, list):
            raise AuditError("Invalid frames list")
        frame_times: dict[str, int] = {}
        for raw in frames:
            frame = required_dict(raw, "frame")
            name = frame.get("file")
            at_ms = frame.get("at_ms")
            if (
                not isinstance(name, str)
                or not name.startswith("frames/")
                or name in frame_times
                or isinstance(at_ms, bool)
                or not isinstance(at_ms, int)
                or not 0 <= at_ms <= cutoff
                or entries.get(name, {}).get("sha256") != frame.get("sha256")
            ):
                raise AuditError("Invalid or post-cutoff evidence frame")
            frame_times[name] = at_ms

        context = report.get("context_frame")
        if context is not None:
            context = required_dict(context, "context frame")
            if (
                context.get("role") != "context_only"
                or context.get("requested_ms") != cutoff
                or frame_times.get(context.get("file")) != context.get("at_ms")
                or entries.get(context.get("file"), {}).get("sha256") != context.get("sha256")
            ):
                raise AuditError("Cutoff context frame is inconsistent")

        states = answer.get("states")
        events = answer.get("events")
        evidence = answer.get("evidence")
        if not all(isinstance(value, list) for value in (states, events, evidence)):
            raise AuditError("Invalid saved answer lists")
        state_ids = set()
        supported_observations = set()
        for raw in states:
            state = required_dict(raw, "object state")
            if state.get("as_of_ms") != cutoff or not isinstance(state.get("object_id"), str):
                raise AuditError("Object state has inconsistent scope")
            state_ids.add(state["object_id"])
            if state.get("evidence") is not None:
                state_ref = required_dict(state["evidence"], "state evidence")
                supported_observations.add(state_ref.get("observation_id"))
        for raw in events:
            event = required_dict(raw, "event")
            at_ms = event.get("at_ms")
            if (
                event.get("object_id") not in state_ids
                or not isinstance(at_ms, int)
                or not 0 <= at_ms <= cutoff
            ):
                raise AuditError("Event is outside the saved answer scope")
            supported_observations.add(event.get("observation_id"))
        cited_observations = set()
        for raw in evidence:
            citation = required_dict(raw, "citation")
            name = citation.get("frame_url")
            if (
                not isinstance(name, str)
                or frame_times.get(name) != citation.get("at_ms")
                or citation.get("video_id") != source_info.get("id")
                or citation.get("run_id") != run.get("id")
                or citation.get("observation_id") not in supported_observations
            ):
                raise AuditError("Citation does not match a scoped evidence frame")
            cited_observations.add(citation["observation_id"])
        if evidence and not state_ids:
            raise AuditError("Citations have no object state")
        for state in states:
            state_ref = state.get("evidence")
            if state_ref is not None and state_ref.get("observation_id") not in cited_observations:
                raise AuditError("State sighting is missing its citation")

    return {
        "status": "verified",
        "archive_sha256": archive_hash,
        "archive_checksum_checked": expected_sha256 is not None,
        "source_checked": source is not None,
        "files": len(entries),
        "frames": len(frames),
        "citations": len(evidence),
        "cutoff_ms": cutoff,
        "recorded_answer": answer.get("answer"),
        "warnings": answer.get("warnings", []),
        "scope": "Archive/manifest integrity and saved citation timing; not publisher identity or visual accuracy",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="FindBack evidence ZIP")
    parser.add_argument("--source", type=Path, help="Original recording to hash-check")
    parser.add_argument("--expected-sha256", help="Archive digest from a trusted channel")
    args = parser.parse_args()
    try:
        result = audit(args.archive, args.source, args.expected_sha256)
    except (OSError, ValueError, TypeError, KeyError, BadZipFile) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        raise SystemExit(1) from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
