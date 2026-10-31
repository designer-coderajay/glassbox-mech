"""Checkpoint/resume for long V7 runs (V8 phase B): per-unit JSON results that survive
a disconnected session and refuse to mix runs with different settings."""

from __future__ import annotations

import json

import pytest

from glassbox.v7.checkpoint import Checkpoint

FP = {"protocol": "x v1.0", "seed": "s", "n_per_cell": 30}


def test_save_load_round_trip(tmp_path):
    cp = Checkpoint(tmp_path / "cp", FP)
    assert not cp.done("q1")
    cp.save("q1", {"attributed": ["index"]})
    assert cp.done("q1") and cp.load("q1") == {"attributed": ["index"]}
    assert cp.units() == ["q1"]


def test_fingerprint_written_and_checked(tmp_path):
    Checkpoint(tmp_path, FP)
    assert json.loads((tmp_path / "_fingerprint.json").read_text()) == FP
    Checkpoint(tmp_path, dict(FP))  # same settings: resumes
    with pytest.raises(ValueError, match="different settings"):
        Checkpoint(tmp_path, dict(FP, seed="other"))


def test_never_overwrites(tmp_path):
    cp = Checkpoint(tmp_path, FP)
    cp.save("q1", {"a": 1})
    with pytest.raises(FileExistsError):
        cp.save("q1", {"a": 2})
    assert cp.load("q1") == {"a": 1}


def test_run_computes_once_across_sessions(tmp_path):
    calls = []

    def work():
        calls.append(1)
        return {"v": len(calls)}

    assert Checkpoint(tmp_path, FP).run("q1", work) == {"v": 1}
    assert Checkpoint(tmp_path, FP).run("q1", work) == {"v": 1}  # resumed, not rerun
    assert calls == [1]


def test_unserialisable_data_leaves_no_file(tmp_path):
    cp = Checkpoint(tmp_path, FP)
    with pytest.raises(TypeError):
        cp.save("q1", {"bad": object()})
    assert not cp.done("q1")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["_fingerprint.json"]


@pytest.mark.parametrize("unit", ["", "../x", "a/b", "_fingerprint", "q 1"])
def test_unit_names_are_validated(tmp_path, unit):
    with pytest.raises(ValueError):
        Checkpoint(tmp_path, FP).save(unit, {})
