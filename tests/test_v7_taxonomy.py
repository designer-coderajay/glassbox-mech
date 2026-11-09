"""Agent fault labeller (V8 E2 PROTOCOL §5.1): one class per failing episode."""

from __future__ import annotations

import pytest

from glassbox.v7.taxonomy import (
    FAULTS,
    count_faults,
    episode_from_trace,
    label_episode,
    label_trace,
)
from glassbox.v7.trace import Recorder

D = ("start_quote", "apply_voucher", "compute_total")


def test_fault_names_and_order():
    assert FAULTS == (
        "no_action",
        "premature_finish",
        "wrong_tool",
        "skipped_tool",
        "wrong_order",
        "other",
    )


def test_correct_episode_has_no_label():
    assert label_episode([], D, hit_limit=True, correct=True) is None


@pytest.mark.parametrize(
    "calls,limit,want",
    [
        ([], False, "no_action"),
        ([], True, "no_action"),  # rule 1 beats rule 2
        (["start_quote", "apply_voucher"], True, "premature_finish"),
        (["lookup_order", "start_quote"], True, "premature_finish"),  # 2 beats 3
        (["lookup_order"], False, "wrong_tool"),
        (["start_quote", "lookup_invoice"], False, "wrong_tool"),
        (["start_quote", "compute_total"], False, "skipped_tool"),
        (["start_quote", "compute_total", "apply_voucher"], False, "wrong_order"),
        (["start_quote", "compute_total", "apply_voucher"], True, "wrong_order"),
        (["start_quote", "apply_voucher", "compute_total"], False, "other"),
        (["start_quote", "apply_voucher", "compute_total"], True, "other"),
        # extra outside tool but nothing missing: not wrong_tool
        (["start_quote", "lookup_order", "apply_voucher", "compute_total"], False, "other"),
        # repeated calls: order uses first calls
        (
            ["start_quote", "compute_total", "apply_voucher", "compute_total"],
            False,
            "wrong_order",
        ),
    ],
)
def test_rules(calls, limit, want):
    assert label_episode(calls, D, hit_limit=limit, correct=False) == want


def test_required_must_be_non_empty():
    with pytest.raises(ValueError):
        label_episode(["x"], (), hit_limit=False, correct=False)


def _trace(tmp_path, run_id, calls, limit, errors=()):
    rec = Recorder(out_dir=tmp_path, run_id=run_id)
    with rec:
        for i, name in enumerate(calls):
            with rec.span("chat", "agent", attributes={"glassbox.agent.step": i}):
                pass
            with rec.span(
                "execute_tool",
                name,
                attributes={
                    "gen_ai.tool.name": name,
                    "glassbox.tool.error": name in errors,
                },
            ):
                pass
        with rec.span(
            "chat",
            "agent",
            attributes={
                "glassbox.agent.step_limit_reached": limit,
                "glassbox.answer.label": "wrong",
            },
        ):
            pass
    return rec.path


def test_episode_from_trace(tmp_path):
    p = _trace(tmp_path, "a", ["start_quote", "compute_total"], True)
    calls, limit = episode_from_trace(p)
    assert calls == ["start_quote", "compute_total"]
    assert limit is True
    p2 = _trace(tmp_path, "b", [], False)
    assert episode_from_trace(p2) == ([], False)


def test_label_trace_and_count(tmp_path):
    paths = [
        _trace(tmp_path, "r0", [], False),
        _trace(tmp_path, "r1", ["start_quote", "compute_total", "apply_voucher"], False),
        _trace(tmp_path, "r2", ["start_quote", "apply_voucher"], True),
        _trace(tmp_path, "r3", ["start_quote", "apply_voucher", "compute_total"], False),
    ]
    outcomes = {str(paths[3]): "correct"}

    def outcome(p):
        return outcomes.get(str(p), "wrong")

    assert label_trace(paths[0], D, outcome, "correct") == "no_action"
    assert label_trace(paths[3], D, outcome, "correct") is None
    counts = count_faults(paths, D, outcome, "correct")
    assert counts == {
        "no_action": 1,
        "premature_finish": 1,
        "wrong_tool": 0,
        "skipped_tool": 0,
        "wrong_order": 1,
        "other": 0,
    }


def test_dominant():
    from glassbox.v7.taxonomy import dominant

    c = {f: 0 for f in FAULTS}
    assert dominant(c) is None
    c.update(wrong_tool=3, skipped_tool=3)
    assert dominant(c) == "wrong_tool"  # tie → earlier in FAULTS
    assert dominant(c, min_failing=7) is None
    c["other"] = 4
    assert dominant(c) == "other"
