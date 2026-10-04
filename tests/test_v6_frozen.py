"""Tests for the V6 freeze guard (scripts/check_v6_frozen.py).

V7 Step 1: the V6 confirmatory artefacts are frozen. The guard records a
sha256 manifest of every git-tracked file under the frozen roots and fails
if any file is changed, removed, or added.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_v6_frozen.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_v6_frozen", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def guard():
    return _load()


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_sha256_file_matches_known_digest(guard, tmp_path):
    _write(tmp_path, "a.txt", "abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert guard.sha256_file(tmp_path / "a.txt") == expected


def test_unchanged_files_produce_no_violations(guard, tmp_path):
    _write(tmp_path, "experiments/v6/x.md", "frozen")
    files = ["experiments/v6/x.md"]
    manifest = guard.build_manifest(tmp_path, files, frozen_at="abc123")
    assert guard.compare(manifest, tmp_path, files) == []


def test_modified_file_is_reported(guard, tmp_path):
    _write(tmp_path, "experiments/v6/x.md", "frozen")
    files = ["experiments/v6/x.md"]
    manifest = guard.build_manifest(tmp_path, files, frozen_at="abc123")
    _write(tmp_path, "experiments/v6/x.md", "tampered")
    violations = guard.compare(manifest, tmp_path, files)
    assert violations == [("modified", "experiments/v6/x.md")]


def test_removed_file_is_reported(guard, tmp_path):
    _write(tmp_path, "experiments/v6/x.md", "frozen")
    files = ["experiments/v6/x.md"]
    manifest = guard.build_manifest(tmp_path, files, frozen_at="abc123")
    (tmp_path / "experiments/v6/x.md").unlink()
    violations = guard.compare(manifest, tmp_path, [])
    assert violations == [("removed", "experiments/v6/x.md")]


def test_added_file_is_reported(guard, tmp_path):
    _write(tmp_path, "experiments/v6/x.md", "frozen")
    manifest = guard.build_manifest(
        tmp_path, ["experiments/v6/x.md"], frozen_at="abc123"
    )
    _write(tmp_path, "experiments/v6/new.md", "late addition")
    current = ["experiments/v6/new.md", "experiments/v6/x.md"]
    violations = guard.compare(manifest, tmp_path, current)
    assert violations == [("added", "experiments/v6/new.md")]


def test_manifest_records_schema_and_commit(guard, tmp_path):
    _write(tmp_path, "glassbox/v6/m.py", "x = 1\n")
    manifest = guard.build_manifest(
        tmp_path, ["glassbox/v6/m.py"], frozen_at="deadbeef"
    )
    assert manifest["schema"] == guard.SCHEMA
    assert manifest["frozen_at_commit"] == "deadbeef"
    assert manifest["roots"] == list(guard.FROZEN_ROOTS)
    assert set(manifest["files"]) == {"glassbox/v6/m.py"}


def test_write_refuses_to_overwrite_existing_manifest(guard, tmp_path):
    target = tmp_path / "manifest.json"
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        guard.write_manifest({"files": {}}, target)


def _git_available() -> bool:
    return shutil.which("git") is not None and (REPO_ROOT / ".git").exists()


@pytest.mark.skipif(not _git_available(), reason="needs the git checkout")
def test_repository_v6_is_unchanged_since_freeze(guard):
    manifest_path = REPO_ROOT / guard.MANIFEST_RELPATH
    if not manifest_path.exists():
        pytest.skip("freeze manifest not created yet")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = guard.list_frozen_files(REPO_ROOT)
    assert guard.compare(manifest, REPO_ROOT, current) == []


@pytest.mark.skipif(not _git_available(), reason="needs the git checkout")
def test_cli_check_exits_zero_on_clean_repo():
    if not (REPO_ROOT / "experiments" / "V6_FROZEN_MANIFEST.json").exists():
        pytest.skip("freeze manifest not created yet")
    result = subprocess.run(
        ["python3", str(SCRIPT)], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
