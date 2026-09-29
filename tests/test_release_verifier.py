"""Hostile controls for the portable release inventory verifier."""

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest


@pytest.fixture
def verifier(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts" / "ops"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("verify_release", scripts / "verify_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_archive


def write_archive(path, members, manifest=None):
    if manifest is None:
        manifest = json.dumps(
            {"format": 1, "files": {"README.md": hashlib.sha256(b"source").hexdigest()}}
        ).encode()
    with tarfile.open(path, "w:gz") as archive:
        for name, data, kind in [*members, ("RELEASE_MANIFEST.json", manifest, tarfile.REGTYPE)]:
            member = tarfile.TarInfo("agentx-findback/" + name)
            member.type = kind
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def test_release_checks_bytes_and_detects_working_tree_drift(tmp_path, verifier):
    bundle = tmp_path / "release.tar.gz"
    write_archive(bundle, [("README.md", b"source", tarfile.REGTYPE)])
    repo = tmp_path / "relocated"
    repo.mkdir()
    (repo / "README.md").write_bytes(b"source")
    assert verifier(bundle, repo)["verified"]
    (repo / "README.md").write_bytes(b"later source")
    (repo / "src").mkdir()
    (repo / "src/new.py").write_text("new")
    result = verifier(bundle, repo)
    assert not result["verified"]
    assert result["archive_valid"]
    assert result["working_tree_drift"] == {
        "added": ["src/new.py"],
        "removed": [],
        "changed": ["README.md"],
    }
    (repo / "README.md").unlink()
    assert verifier(bundle, repo)["working_tree_drift"]["removed"] == ["README.md"]


@pytest.mark.parametrize(
    ("members", "reason"),
    [
        ([("README.md", b"tamper", tarfile.REGTYPE)], "digest mismatch"),
        ([], "membership differs"),
        ([("README.md", b"source", tarfile.REGTYPE)] * 2, "Duplicate archive"),
        ([("../escape", b"source", tarfile.REGTYPE)], "Invalid release path"),
        ([("README.md", b"", tarfile.SYMTYPE)], "Non-regular archive"),
        ([("README.md", b"", tarfile.LNKTYPE)], "Non-regular archive"),
        (
            [("README.md", b"source", tarfile.REGTYPE), ("extra", b"x", tarfile.REGTYPE)],
            "membership differs",
        ),
    ],
)
def test_release_rejects_corrupt_or_unsafe_archive(tmp_path, verifier, members, reason):
    bundle = tmp_path / "bad.tar.gz"
    write_archive(bundle, members)
    with pytest.raises(ValueError, match=reason):
        verifier(bundle)


@pytest.mark.parametrize(
    ("manifest", "reason"),
    [
        (b'{"format":1,"files":{},"files":{}}', "Duplicate manifest key"),
        (b'{"format":1,"files":{}}', "inventory is empty"),
        (b'{"format":1,"files":{"README.md":"pretend"}}', "Invalid SHA-256"),
        (b'{"format":2,"files":{}}', "Unsupported release"),
        (b'{"format":true,"files":{}}', "invalid release manifest"),
    ],
)
def test_release_rejects_invalid_manifest(tmp_path, verifier, manifest, reason):
    bundle = tmp_path / "bad.tar.gz"
    write_archive(bundle, [("README.md", b"source", tarfile.REGTYPE)], manifest)
    with pytest.raises(ValueError, match=reason):
        verifier(bundle)
