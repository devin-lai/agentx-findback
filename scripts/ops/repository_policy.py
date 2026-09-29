"""Private-path exclusions for portable archives, including checkouts without Git.

Keep this policy aligned with .gitignore and the build exclusions in pyproject.toml.
It is a filename boundary, not a content or credential scanner.
"""

from fnmatch import fnmatchcase
from pathlib import PurePosixPath

PRIVATE_DIRECTORIES = {
    "private",
    "personal",
    "plans",
    "secrets",
    "credentials",
    "__pycache__",
    "node_modules",
    "venv",
    "htmlcov",
    "test-results",
    "playwright-report",
}
PRIVATE_FILES = (
    "*.private.*",
    "*.local.*",
    "secrets.*",
    "credentials.*",
    "credentials-*",
    "credentials_*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "id_rsa*",
    "id_ed25519*",
    "*.code-workspace",
    "*.py[cod]",
    "*.egg-info",
    "*.sqlite*",
    "*.db",
    "*.db-*",
    "*.log",
    "*.mp4",
    "*.mov",
    "*.avi",
    "*.mkv",
    "*.wav",
    "*.mp3",
    "*.safetensors",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.onnx",
    "*.gguf",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.bak",
    "*.swp",
    "*.xlsx",
    "*.xls",
    "*.numbers",
    "*.docx",
    "*.pptx",
    "thumbs.db",
    "coverage.xml",
)
PUBLIC_DOTFILES = {
    ".gitignore",
    ".dockerignore",
    ".env.example",
    ".python-version",
    "web/.prettierrc.json",
    "web/.prettierignore",
}
PRIVATE_ROOTS = {
    "model",
    "models",
    "artifacts",
    "data",
    "datasets",
    "uploads",
    "recordings",
    "build",
    "dist",
    "tmp",
}


def is_private_path(path: PurePosixPath) -> bool:
    """Return whether a repository-relative file must stay out of source archives."""
    name = path.as_posix()
    if name in PUBLIC_DOTFILES:
        return False
    parts = tuple(part.lower() for part in path.parts)
    if not parts or parts[0] in PRIVATE_ROOTS:
        return True
    if parts[:2] == ("docs", "submission"):
        return True
    if parts[:2] == ("scripts", "research"):
        return True
    if parts[:2] == ("web", "demo"):
        return True
    if name in {
        "scripts/eval/evaluate_skill_lift.py",
        "scripts/eval/local_skill_agent.py",
        "scripts/eval/plot_skill_lift.py",
    }:
        return True
    if parts[:2] == ("docs", "benchmarks") and name != "docs/benchmarks/INDEX.md":
        return True
    if parts[:2] == ("docs", "images") and (
        path.suffix.lower() == ".json"
        or path.stem.startswith(("tracking-", "camera-recovery", "skill-lift"))
        or path.name.endswith("offline-evidence-report.png")
    ):
        return True
    if (
        len(parts) == 2
        and parts[0] == "docs"
        and (
            parts[1] in {"roadmap.md", "competition.md", "changelog.md"}
            or fnmatchcase(parts[1], "development_run*.md")
        )
    ):
        return True
    for index, part in enumerate(parts):
        if part.startswith(".") and not (index == 0 and part == ".github"):
            return True
        if part in PRIVATE_DIRECTORIES or any(fnmatchcase(part, p) for p in PRIVATE_FILES):
            return True
    return False
