"""Verify a release without extracting it; optionally detect drift from the working tree.

This checks archive membership and bytes, not model accuracy, media rights, secret
absence, public availability or competition eligibility. Python 3.10+, stdlib only.
"""

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath

PREFIX = "agentx-findback/"
MAX_BYTES = 512 * 1024 * 1024
MAX_FILES = 10000


def digest(stream):
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(chunk)
    return result.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest key: {key}")
        result[key] = value
    return result


def safe_name(name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Invalid release path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError(f"Invalid release path: {name}")
    return name


def verify_archive(bundle: Path, root: Path | None = None):
    hashes = {}
    manifest = None
    total = 0
    with tarfile.open(bundle, mode="r|gz") as archive:
        seen = set()
        for member in archive:
            if not member.name.startswith(PREFIX):
                raise ValueError("Unexpected archive root")
            name = safe_name(member.name[len(PREFIX) :])
            if name in seen:
                raise ValueError(f"Duplicate archive member: {name}")
            seen.add(name)
            if not member.isfile():
                raise ValueError(f"Non-regular archive member: {name}")
            total += member.size
            if len(seen) > MAX_FILES or total > MAX_BYTES:
                raise ValueError("Release exceeds verifier size limits")
            with archive.extractfile(member) as stream:
                if name == "RELEASE_MANIFEST.json":
                    if member.size > 4 * 1024 * 1024:
                        raise ValueError("Manifest exceeds verifier size limit")
                    manifest = json.loads(stream.read(), object_pairs_hook=unique_object)
                else:
                    hashes[name] = digest(stream)
    if not isinstance(manifest, dict) or type(manifest.get("format")) is not int:
        raise ValueError("Missing or invalid release manifest")
    if manifest["format"] != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported release manifest")
    expected = manifest["files"]
    if not expected:
        raise ValueError("Release inventory is empty")
    for name, value in expected.items():
        safe_name(name)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"Invalid SHA-256 for {name}")
    if set(expected) != set(hashes):
        raise ValueError("Archive membership differs from manifest")
    changed = sorted(name for name in expected if hashes[name] != expected[name])
    if changed:
        raise ValueError("Archive digest mismatch: " + ", ".join(changed))
    report = {
        "archive_valid": True,
        "file_count": len(hashes),
        "uncompressed_bytes": total,
        "base_commit": manifest.get("base_commit"),
        "working_tree_checked": root is not None,
    }
    if root is not None:
        # Same inventory contract as the packager; bytes are read independently.
        from package_release import release_files

        current = {}
        for path in release_files(root):
            with path.open("rb") as stream:
                current[path.relative_to(root).as_posix()] = digest(stream)
        report["working_tree_drift"] = {
            "added": sorted(current.keys() - expected.keys()),
            "removed": sorted(expected.keys() - current.keys()),
            "changed": sorted(
                name for name in current.keys() & expected.keys() if current[name] != expected[name]
            ),
        }
        report["working_tree_matches"] = not any(report["working_tree_drift"].values())
    report["verified"] = report["archive_valid"] and report.get("working_tree_matches", True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--root", type=Path, help="Compare against this checkout, including built UI"
    )
    args = parser.parse_args()
    try:
        report = verify_archive(args.bundle, args.root)
    except (OSError, ValueError, tarfile.TarError, EOFError) as error:
        report = {"verified": False, "error": str(error)}
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["verified"] else 1)


if __name__ == "__main__":
    main()
