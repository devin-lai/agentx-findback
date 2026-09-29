"""Exercise Git's real index and ignore behavior at the publication boundary."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_bytes((ROOT / ".gitignore").read_bytes())
    return tmp_path


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        "web/.env.production",
        ".private/docs/ROADMAP.md",
        "docs/ROADMAP.md",
        "docs/COMPETITION.md",
        "docs/DEVELOPMENT_RUN_future.md",
        "docs/CHANGELOG.md",
        "docs/benchmarks/RAW_RUN.md",
        "docs/benchmarks/results/summary.json",
        "docs/images/skill-lift.json",
        "docs/images/node-offline-evidence-report.png",
        "tmp/experiment.json",
        "web/demo/storyboard.ts",
        "docs/submission/WINNING_PLAN.md",
        "docs/personal/notes.md",
        "docs/new.private.md",
        "scripts/config.local.json",
        "docs/credentials.xlsx",
        "src/agentx/secrets.json",
        "src/agentx/service.key",
        "src/agentx/server.pem",
        ".agentx-e2e/memory.sqlite3",
        "model/weights.safetensors",
        "artifacts/release.tar.gz",
        "recordings/desk.mp4",
        ".codex/session.json",
    ],
)
def test_git_refuses_private_adds(repo, path):
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", path],
        cwd=repo,
        check=False,
    )
    assert result.returncode == 0, path


@pytest.mark.parametrize(
    "path",
    [
        ".env.example",
        "README.md",
        "CONTRIBUTING.md",
        "AGENTS.md",
        "SECURITY.md",
        ".github/workflows/ci.yml",
        "src/agentx/config.py",
        "tests/fixtures/planner.json",
        "docs/benchmarks/INDEX.md",
        "docs/images/screenshot.png",
        "web/package-lock.json",
        "uv.lock",
    ],
)
def test_public_source_can_be_added(repo, path):
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", path],
        cwd=repo,
        check=False,
    )
    assert result.returncode == 1, path


def test_privacy_check_catches_forced_adds_and_keeps_local_copy(repo):
    public = repo / "README.md"
    public.write_text("Public source")
    private = repo / "docs/COMPETITION.md"
    private.parent.mkdir()
    private.write_text("private-canary-must-not-be-printed")
    subprocess.run(["git", "add", ".gitignore", "README.md"], cwd=repo, check=True)
    command = [sys.executable, str(ROOT / "scripts/ops/check_repository.py"), "--root", str(repo)]
    assert subprocess.run(command, capture_output=True).returncode == 0
    subprocess.run(["git", "add", "-f", "docs/COMPETITION.md"], cwd=repo, check=True)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1
    assert "docs/COMPETITION.md" in result.stdout
    assert "private-canary" not in result.stdout + result.stderr
    subprocess.run(["git", "rm", "--cached", "docs/COMPETITION.md"], cwd=repo, check=True)
    assert private.read_text() == "private-canary-must-not-be-printed"
    assert subprocess.run(command, capture_output=True).returncode == 0


def test_public_narrative_has_no_experiment_dates_or_personal_paths():
    pages = [
        ROOT / "README.md",
        *(ROOT / "docs").rglob("*.md"),
        *(ROOT / "skills").rglob("*.md"),
        *(ROOT / "src/agentx/skills").rglob("*.md"),
        *(ROOT / "integrations").rglob("*.md"),
    ]
    date = re.compile(r"(?<!\d)20\d{2}[-_/]?\d{2}[-_/]?\d{2}(?!\d)|\bSeptember\s+\d{1,2}\b")
    for page in pages:
        content = page.read_text()
        assert not date.search(content), page
        assert "/Users/" not in content, page
        assert not re.search(r"/home/[A-Za-z][^/<\s]*/", content), page


@pytest.mark.parametrize("tracked", [False, True])
@pytest.mark.parametrize("kind", ["home", "token", "environment"])
def test_content_check_catches_sensitive_candidates_without_echoing_them(repo, tracked, kind):
    values = {
        "home": "/" + "Users" + "/private-canary/project/",
        "token": "ghp_" + "x" * 36,
        "environment": "AGENTX_API_TOKEN=" + "private-canary-value",
    }
    name = ".env.example" if kind == "environment" else "README.md"
    (repo / name).write_text(values[kind])
    if tracked:
        subprocess.run(["git", "add", name], cwd=repo, check=True)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/ops/check_repository.py"), "--root", str(repo)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert name in result.stdout
    assert values[kind] not in result.stdout + result.stderr
    assert "private-canary" not in result.stdout + result.stderr


@pytest.mark.parametrize("linked_directory", [False, True])
def test_content_check_never_follows_symlinks_or_reads_ignored_files(
    repo, tmp_path_factory, linked_directory
):
    outside = tmp_path_factory.mktemp("outside") / "canary.txt"
    outside.write_text("ghp_" + "x" * 36)
    if linked_directory:
        (repo / "src").mkdir()
        (repo / "src/canary.txt").write_text("public")
        subprocess.run(["git", "add", "src/canary.txt"], cwd=repo, check=True)
        (repo / "src/canary.txt").unlink()
        (repo / "src").rmdir()
        (repo / "src").symlink_to(outside.parent, target_is_directory=True)
    else:
        (repo / "README.md").symlink_to(outside)
    (repo / ".env").write_text("ghp_" + "x" * 36)
    command = [sys.executable, str(ROOT / "scripts/ops/check_repository.py"), "--root", str(repo)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1
    assert "symlink" in result.stdout
    assert "GitHub access token" not in result.stdout
    (repo / ("src" if linked_directory else "README.md")).unlink()
    assert subprocess.run(command, capture_output=True).returncode == 0
