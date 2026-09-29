"""Build an allowlisted source + built-web bundle without credentials, data or model weights."""

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

from check_repository import content_findings
from repository_policy import is_private_path

ROOT_FILES = (
    "README.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "uv.lock",
    "Dockerfile",
    ".dockerignore",
    ".gitignore",
    ".env.example",
    ".python-version",
    "Makefile",
    "web/package.json",
    "web/package-lock.json",
    "web/index.html",
    "web/tsconfig.json",
    "web/vite.config.ts",
    "web/license-notices.js",
    "web/svelte.config.js",
    "web/playwright.config.ts",
    "web/.prettierrc.json",
    "web/.prettierignore",
)
TREES = (
    "src",
    "tests",
    "scripts",
    "skills",
    "integrations",
    "docs",
    "web/src",
    "web/public",
    "web/tests",
    "web/dist",
    ".github",
)


def release_files(root: Path) -> list[Path]:
    paths = {root / name for name in ROOT_FILES if (root / name).is_file()}
    for name in TREES:
        if (root / name).is_symlink():
            raise ValueError("Release inputs must not contain symlinks.")
        for path in (root / name).rglob("*"):
            if path.is_symlink():
                raise ValueError("Release inputs must not contain symlinks.")
            if path.is_file() and not any(
                p.startswith(".") or p == "__pycache__" for p in path.relative_to(root / name).parts
            ):
                paths.add(path)
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Release inputs must not contain symlinks.")
    selected = sorted(
        (p for p in paths if not is_private_path(p.relative_to(root))),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    findings = content_findings(root, {p.relative_to(root).as_posix() for p in selected})
    if findings:
        locations = "; ".join(f"{name!r}:{line}: {rule}" for name, line, rule in findings)
        raise ValueError("Potentially sensitive release content (values redacted): " + locations)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/releases/agentx-findback.tar.gz")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if not (root / "web/dist/index.html").is_file():
        raise SystemExit("Build the frontend first: npm --prefix web run build")
    paths = release_files(root)
    manifest = {
        "format": 1,
        "scope": "Application source and frontend. No runtime, media, secrets or model weights.",
        "files": {},
    }
    # File digests describe the reviewed working copy without exposing local Git history.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.output.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w") as archive,
    ):

        def add(name, data, mode=0o644):
            info = tarfile.TarInfo("agentx-findback/" + name)
            info.size, info.mode, info.mtime = len(data), mode, 0
            archive.addfile(info, io.BytesIO(data))

        for path in paths:
            name = path.relative_to(root).as_posix()
            data = path.read_bytes()
            manifest["files"][name] = hashlib.sha256(data).hexdigest()
            add(name, data, 0o755 if name.endswith(".sh") else 0o644)
        add("RELEASE_MANIFEST.json", (json.dumps(manifest, indent=2) + "\n").encode())
    checksum = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{checksum}  {args.output.name}\n"
    )
    print(
        json.dumps(
            {
                "bundle": str(args.output),
                "sha256": checksum,
                "file_count": len(manifest["files"]),
                "bytes": args.output.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
