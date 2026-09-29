"""Check public Git candidates for private paths and obvious sensitive content.

Ignored, untracked files and private paths are never opened. Findings report only a
relative path, line number and rule, never the matching content. This is a narrow
publication check, not a guarantee that prose, images or Git history are private.
"""

import argparse
import re
import subprocess
from pathlib import Path, PurePosixPath

from repository_policy import is_private_path

CONTENT_RULES = {
    "personal home path": re.compile(r"/(?:Users|home)/[A-Za-z][^/\s<>\"']+/"),
    "private key marker": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "GitHub access token": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b"
    ),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
}
ENV_SECRET = re.compile(
    r"^[ \t]*(?:export[ \t]+)?[A-Z][A-Z0-9_]*(?:TOKEN|KEY|PASSWORD|SECRET)"
    r"[ \t]*=[ \t]*([^#\r\n]+)",
    re.MULTILINE,
)


def git_paths(root: Path, *args: str) -> set[str]:
    result = subprocess.run(
        ["git", "-c", "core.excludesFile=/dev/null", "ls-files", "-z", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return {p.decode("utf-8", errors="surrogateescape") for p in result.stdout.split(b"\0") if p}


def tracked_private_paths(root: Path) -> list[str]:
    tracked = git_paths(root, "--cached")
    ignored = git_paths(root, "--cached", "--ignored", "--exclude-standard")
    return sorted(ignored | {p for p in tracked if is_private_path(PurePosixPath(p))})


def content_findings(root: Path, names: set[str]) -> list[tuple[str, int, str]]:
    findings = []
    for name in sorted(names):
        if is_private_path(PurePosixPath(name)):
            continue
        path = root / name
        parts = PurePosixPath(name).parts
        if any(
            root.joinpath(*parts[:i]).is_symlink() for i in range(1, len(parts) + 1)
        ) or not path.resolve().is_relative_to(root.resolve()):
            findings.append((name, 0, "symlink outside the publication boundary"))
            continue
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        content = data.decode("utf-8", errors="replace")
        for rule, pattern in CONTENT_RULES.items():
            findings.extend(
                (name, content.count("\n", 0, match.start()) + 1, rule)
                for match in pattern.finditer(content)
            )
        if name == ".env.example":
            findings.extend(
                (name, content.count("\n", 0, match.start()) + 1, "nonempty example credential")
                for match in ENV_SECRET.finditer(content)
                if match[1].strip() not in {"", "''", '""'}
            )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    try:
        private = tracked_private_paths(args.root)
        candidates = git_paths(args.root, "--cached", "--others", "--exclude-standard")
        private = sorted(
            set(private) | {p for p in candidates if is_private_path(PurePosixPath(p))}
        )
        findings = content_findings(args.root, candidates - set(private))
    except (OSError, subprocess.CalledProcessError):
        print("Privacy check requires Git, an initialized repository and readable public files.")
        return 2
    if private:
        print("Private or ignored paths are in the public Git candidates:")
        for path in private:
            print(f"  {path!r}")
        print("Preserve local copies, remove these paths from the index, and review Git history.")
    if findings:
        print("Potentially sensitive content (matching values are redacted):")
        for path, line, rule in findings:
            print(f"  {path!r}:{line}: {rule}")
    if private or findings:
        return 1
    print(
        "Privacy check passed for tracked and untracked public files. Review images and Git history separately."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
