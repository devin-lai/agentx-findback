"""Verify an extracted FindBack evidence report using only the Python standard library.

Usage: python3 verify.py [report_directory] [--source original_video.mp4]
Hashes detect changed files relative to the manifest, not publisher identity or
visual correctness. Obtain the archive checksum through a trusted channel.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def file_sha256(path: Path) -> str:
    """Bounded-memory hashing on Python 3.10, without hashlib.file_digest (3.11+)."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _desktop_noise(relative: Path) -> bool:
    """Files an operating system or Python adds when a report is opened, never report content.

    Only regular files qualify: a symlink with one of these names is still reported."""
    path_parts = relative.parts
    name = path_parts[-1]
    return (
        name in {".DS_Store", "Thumbs.db", "desktop.ini"}
        or name.startswith("._")
        or "__MACOSX" in path_parts
        or "__pycache__" in path_parts
    )


def verify(directory: Path, source: Path | None = None) -> dict:
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format") != "agentx-evidence-bundle-v1":
        raise ValueError("Unsupported evidence bundle format.")
    entries = manifest.get("files", {})
    if not {"report.html", "report.json", "verify.py", "README.txt"} <= entries.keys():
        raise ValueError("The manifest is missing required report files.")
    actual = {
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.is_symlink() or (p.is_file() and not _desktop_noise(p.relative_to(directory)))
    }
    if actual != set(entries) | {"manifest.json"}:
        raise ValueError("The report contains missing or unexpected files.")
    for name, expected in entries.items():
        relative = PurePosixPath(name)
        path = directory / name
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or path.is_symlink()
            or any(parent.is_symlink() for parent in path.parents if parent != directory)
            or not path.resolve().is_relative_to(directory)
        ):
            raise ValueError("Unsafe report file path.")
        digest = file_sha256(path)
        if digest != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Integrity check failed: {name}")
    report = json.loads((directory / "report.json").read_text())
    if report["source"]["sha256"] != manifest["source_sha256"]:
        raise ValueError("The source fingerprints disagree.")
    for frame in report["frames"]:
        if frame["at_ms"] > report["answer"]["as_of_ms"]:
            raise ValueError("An evidence frame is after the answer cutoff.")
        if entries.get(frame["file"], {}).get("sha256") != frame["sha256"]:
            raise ValueError("An evidence frame fingerprint disagrees with the manifest.")
    if source is not None:
        digest = file_sha256(source)
        if digest != manifest["source_sha256"]:
            raise ValueError("The original video does not match this report.")
    return {
        "status": "verified",
        "files": len(entries),
        "frames": len(report["frames"]),
        "cutoff_ms": report["answer"]["as_of_ms"],
        "source_checked": source is not None,
        "scope": "File integrity against the supplied manifest; not a signature or accuracy score.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path(__file__).parent)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.directory, args.source)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        raise SystemExit(1) from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
