"""Direction-aware view of an intervention report (E1b PROTOCOL §3).

A factor is *harmful* iff its sufficiency test is significant and switching it alone
lowers the target rate, or its necessity test is significant and reverting it alone
raises the target rate. *Helpful* iff sufficiency is significant, switching raises the
target rate, and the factor is not harmful. Existing outputs are unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict

import pytest

from glassbox.v7.intervene import Factor, InterventionSettings, run_interventions
from glassbox.v7.trace import Recorder, read_trace

A = Factor("a", base=0, candidate=1)
B = Factor("b", base="x", candidate="y")
INERT = Factor("log_level", base="info", candidate="debug")
FAST = InterventionSettings(n_per_cell=12, n_perm=199, alpha=0.05, seed="d")


def _outcome(path: Path) -> str:
    spans, _ = read_trace(path)
    chat = [s for s in spans if "glassbox.answer.label" in s["attributes"]]
    return str(chat[0]["attributes"]["glassbox.answer.label"])


def runner(correct: Callable[[Dict[str, Any]], bool]):
    """Deterministic runner: the answer is ``correct`` iff ``correct(config)``."""

    def run(config: Dict[str, Any], seed: int, out: Path, run_id: str):
        label = "correct" if correct(config) else "wrong"
        rec = Recorder(out_dir=out, run_id=run_id)
        with rec:
            with rec.span("chat", "toy", attributes={"glassbox.answer.label": label}):
                pass
        return rec.path

    return run


def _report(tmp_path, factors, correct):
    return run_interventions(factors, runner(correct), _outcome, tmp_path, FAST)


def test_rate_reads_cell_counts(tmp_path):
    r = _report(tmp_path, [A], lambda c: c["a"] == 0)
    assert r.rate("baseline", "correct") == 1.0
    assert r.rate("candidate", "correct") == 0.0
    assert r.rate("baseline", "never-seen") == 0.0


def test_breaking_change_is_harmful_not_helpful(tmp_path):
    r = _report(tmp_path, [A, INERT], lambda c: c["a"] == 0)
    assert r.harmful("correct") == ["a"]
    assert r.helpful("correct") == []


def test_improving_change_is_helpful_not_harmful(tmp_path):
    r = _report(tmp_path, [B, INERT], lambda c: c["b"] == "y")
    assert r.attributed == ["b"]
    assert r.harmful("correct") == []
    assert r.helpful("correct") == ["b"]


def test_mixed_directions_are_separated(tmp_path):
    r = _report(tmp_path, [A, B], lambda c: c["a"] == 0 or c["b"] == "y")
    # only_a: wrong (a breaks, b at base) -> harmful by sufficiency.
    # only_b: correct (b fixes) and baseline: correct (a at base) -> no change.
    assert r.harmful("correct") == ["a"]
    assert "a" not in r.helpful("correct")


def test_helpful_and_harmful_in_one_report(tmp_path):
    def correct(c):
        # Baseline is wrong; b alone fixes it; a breaks everything.
        return c["a"] == 0 and c["b"] == "y"

    r = _report(tmp_path, [A, B], correct)
    assert r.helpful("correct") == ["b"]
    assert r.harmful("correct") == ["a"]


def test_necessity_only_harm_counts(tmp_path):
    """Wrong only when both changed: neither switch alone matters, reverting does."""
    r = _report(tmp_path, [A, B], lambda c: not (c["a"] == 1 and c["b"] == "y"))
    assert r.attributed == []
    assert r.harmful("correct") == ["a", "b"]
    assert r.helpful("correct") == []


def test_inert_factor_is_neither(tmp_path):
    r = _report(tmp_path, [A, INERT], lambda c: c["a"] == 0)
    assert "log_level" not in r.harmful("correct")
    assert "log_level" not in r.helpful("correct")


def test_unseen_target_gives_no_direction(tmp_path):
    r = _report(tmp_path, [A], lambda c: c["a"] == 0)
    assert r.harmful("absent") == []
    assert r.helpful("absent") == []


def test_direction_table(tmp_path):
    r = _report(tmp_path, [A, B], lambda c: c["a"] == 0 and c["b"] == "y")
    d = r.direction("correct")
    assert d["a"]["sufficiency_delta"] == 0.0  # baseline already wrong
    assert d["b"]["sufficiency_delta"] == 1.0
    assert set(d["a"]) == {
        "sufficiency_delta",
        "necessity_delta",
        "harmful",
        "helpful",
    }
    json.dumps(d)


def test_existing_outputs_unchanged(tmp_path):
    r = _report(tmp_path, [A, INERT], lambda c: c["a"] == 0)
    keys = set(r.to_dict())
    assert keys == {
        "factors",
        "settings",
        "cells",
        "total",
        "sufficiency",
        "necessity",
        "attributed",
        "evidence",
        "paths",
    }
    assert "harm" not in " ".join(r.evidence).lower()


@pytest.mark.parametrize("target", ["", None])
def test_target_must_be_given(tmp_path, target):
    r = _report(tmp_path, [A], lambda c: True)
    with pytest.raises(ValueError, match="target"):
        r.harmful(target)  # type: ignore[arg-type]
