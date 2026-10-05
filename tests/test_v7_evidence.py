"""Tests for V7 evidence packages (V7 plan §15–§17)."""

from __future__ import annotations

import json

import pytest

from glassbox.v7.evidence import (
    STATUSES,
    EvidenceError,
    PackageSpec,
    build_package,
    main,
    verify_package,
)


@pytest.fixture
def artifacts(tmp_path):
    src = tmp_path / "src"
    (src / "traces").mkdir(parents=True)
    (src / "PROTOCOL.md").write_text("# protocol\n", encoding="utf-8")
    (src / "results.json").write_text('{"overall": "PASS"}\n', encoding="utf-8")
    (src / "traces" / "a.trace.jsonl").write_text("{}\n", encoding="utf-8")
    return {
        "protocol": [src / "PROTOCOL.md"],
        "measurements": [src / "results.json"],
        "traces": [src / "traces" / "a.trace.jsonl"],
    }


def _build(tmp_path, artifacts, **kw):
    fields = dict(
        claim="Trace + diff localised an injected retrieval fault.",
        status="EXPERIMENTALLY_SUPPORTED",
        scope="Controlled single-cause fault; GPT-2; 8 questions.",
        artifacts=artifacts,
        provenance={"protocol_commit": "0e70405", "results_commit": "aedfebf"},
    )
    fields.update(kw)
    return build_package(tmp_path / "evidence", "exp1", PackageSpec(**fields))


def test_build_creates_expected_layout(tmp_path, artifacts):
    pkg = _build(tmp_path, artifacts)
    for name in ("claim.json", "provenance.json", "manifest.json", "README.md"):
        assert (pkg / name).is_file(), name
    assert (pkg / "artifacts" / "protocol" / "PROTOCOL.md").is_file()
    assert (pkg / "artifacts" / "traces" / "a.trace.jsonl").is_file()


def test_claim_records_status_scope_and_limits(tmp_path, artifacts):
    pkg = _build(tmp_path, artifacts)
    claim = json.loads((pkg / "claim.json").read_text(encoding="utf-8"))
    assert claim["status"] == "EXPERIMENTALLY_SUPPORTED"
    assert claim["scope"].startswith("Controlled")
    assert "not" in claim["integrity_note"].lower()


def test_fresh_package_verifies(tmp_path, artifacts):
    pkg = _build(tmp_path, artifacts)
    assert verify_package(pkg) == []


def test_modified_artifact_is_detected(tmp_path, artifacts):
    pkg = _build(tmp_path, artifacts)
    (pkg / "artifacts" / "measurements" / "results.json").write_text(
        '{"overall": "FAIL"}\n', encoding="utf-8"
    )
    assert verify_package(pkg) == [("modified", "artifacts/measurements/results.json")]


def test_added_and_removed_files_are_detected(tmp_path, artifacts):
    pkg = _build(tmp_path, artifacts)
    (pkg / "artifacts" / "traces" / "a.trace.jsonl").unlink()
    (pkg / "artifacts" / "extra.txt").write_text("x", encoding="utf-8")
    assert sorted(verify_package(pkg)) == [
        ("added", "artifacts/extra.txt"),
        ("removed", "artifacts/traces/a.trace.jsonl"),
    ]


def test_tampered_claim_is_detected(tmp_path, artifacts):
    pkg = _build(tmp_path, artifacts)
    claim = json.loads((pkg / "claim.json").read_text(encoding="utf-8"))
    claim["status"] = "VERIFIED"
    (pkg / "claim.json").write_text(json.dumps(claim), encoding="utf-8")
    assert ("modified", "claim.json") in verify_package(pkg)


def test_unknown_status_rejected(tmp_path, artifacts):
    with pytest.raises(EvidenceError, match="status"):
        _build(tmp_path, artifacts, status="PROVEN")
    assert "REPRODUCED" in STATUSES


def test_reproduced_requires_named_independent_reproducer(tmp_path, artifacts):
    with pytest.raises(EvidenceError, match="reproduc"):
        _build(tmp_path, artifacts, status="REPRODUCED")
    pkg = _build(
        tmp_path,
        artifacts,
        status="REPRODUCED",
        provenance={"reproduced_by": "Jane Doe (independent)", "protocol_commit": "x"},
    )
    assert verify_package(pkg) == []


def test_missing_artifact_rejected(tmp_path, artifacts):
    artifacts["traces"].append(tmp_path / "nope.jsonl")
    with pytest.raises(EvidenceError, match="nope.jsonl"):
        _build(tmp_path, artifacts)


def test_duplicate_artifact_names_within_role_rejected(tmp_path, artifacts):
    other = tmp_path / "other"
    other.mkdir()
    (other / "PROTOCOL.md").write_text("different\n", encoding="utf-8")
    artifacts["protocol"].append(other / "PROTOCOL.md")
    with pytest.raises(EvidenceError, match="duplicate"):
        _build(tmp_path, artifacts)


def test_refuses_to_overwrite(tmp_path, artifacts):
    _build(tmp_path, artifacts)
    with pytest.raises(FileExistsError):
        _build(tmp_path, artifacts)


def test_readme_states_claim_status_and_integrity_limit(tmp_path, artifacts):
    pkg = _build(tmp_path, artifacts)
    text = (pkg / "README.md").read_text(encoding="utf-8")
    assert "EXPERIMENTALLY_SUPPORTED" in text
    assert "Integrity is not truth" in text
    assert "verify" in text


def test_cli_verify_exit_codes(tmp_path, artifacts, capsys):
    pkg = _build(tmp_path, artifacts)
    assert main(["verify", str(pkg)]) == 0
    (pkg / "README.md").write_text("changed", encoding="utf-8")
    assert main(["verify", str(pkg)]) == 1
    assert "modified" in capsys.readouterr().out


def test_experiment1_package_builds_and_verifies(tmp_path):
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "experiments" / "v7" / "rag_fault"
    spec = importlib.util.spec_from_file_location("exp1_pkg", root / "package.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    pkg = build_package(tmp_path, mod.PACKAGE_ID, mod.spec())
    assert verify_package(pkg) == []
    n_traces = len(list((pkg / "artifacts").glob("traces_*/*.trace.jsonl")))
    assert n_traces == 32
