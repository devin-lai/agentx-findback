import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def scripts_path(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts" / "ops"))


def load_packager():
    spec = importlib.util.spec_from_file_location(
        "package_release", Path(__file__).parents[1] / "scripts/ops/package_release.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundle_allowlist_excludes_private_runtime_data(tmp_path):
    for name in [
        ".env",
        "model/weights.safetensors",
        "artifacts/private.json",
        ".agentx/memory.sqlite3",
        "tmp/access.xlsx",
        "src/agentx/__pycache__/config.pyc",
        "src/agentx/.env",
        ".private/docs/ROADMAP.md",
        "docs/ROADMAP.md",
        "docs/COMPETITION.md",
        "docs/DEVELOPMENT_RUN_sample.md",
        "docs/submission/WINNING_PLAN.md",
        "docs/personal/notes.md",
        "docs/ideas.private.md",
        "docs/credentials.xlsx",
        "scripts/secrets.json",
        "src/agentx/credentials/service.json",
        "src/agentx/service.key",
        "src/agentx/recording.mp4",
        "web/dist/config.local.json",
        "web/dist/.env.production",
        "src/agentx/config.py",
        "web/dist/index.html",
        ".env.example",
        "web/.prettierrc.json",
        "CONTRIBUTING.md",
        "NOTICE",
        "docs/ARCHITECTURE.md",
        "docs/benchmarks/results/public.json",
        "docs/benchmarks/INDEX.md",
        "docs/images/node-offline-evidence-report.png",
        "web/demo/storyboard.ts",
        ".github/workflows/ci.yml",
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    found = {str(p.relative_to(tmp_path)) for p in load_packager().release_files(tmp_path)}
    assert found == {
        "src/agentx/config.py",
        "web/dist/index.html",
        ".env.example",
        "web/.prettierrc.json",
        "CONTRIBUTING.md",
        "NOTICE",
        "docs/ARCHITECTURE.md",
        "docs/benchmarks/INDEX.md",
        ".github/workflows/ci.yml",
    }


@pytest.mark.parametrize("target", ["src", "README.md", "src/agentx/config.py"])
def test_bundle_rejects_symlinks(tmp_path, target):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "private"
    if target == "src":
        outside.mkdir()
    else:
        outside.write_text("private")
    link = root / target
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="symlinks"):
        load_packager().release_files(root)


def test_bundle_checks_public_content_without_git_and_redacts_values(tmp_path):
    value = "ghp_" + "x" * 36
    (tmp_path / "README.md").write_text(value)
    with pytest.raises(ValueError, match="values redacted") as error:
        load_packager().release_files(tmp_path)
    assert value not in str(error.value)
    assert "README.md" in str(error.value)
    (tmp_path / "README.md").write_text("Public source")
    (tmp_path / ".env").write_text(value)
    assert [p.name for p in load_packager().release_files(tmp_path)] == ["README.md"]
