"""Tests for the V7 noise-aware group diff (experiments/v7/noise_baseline/PROTOCOL.md §2).

Unit tests use hand-built traces only. They do not run Experiment 3 Part A or Part B.
"""

from __future__ import annotations

import json
from typing import List, Optional

import pytest

from glassbox.v7.groupdiff import (
    group_diff,
    holm,
    main,
    permutation_p,
    tvd,
)
from glassbox.v7.trace import Recorder


def _run(
    tmp_path,
    run_id: str,
    *,
    doc: str = "right",
    label: str = "correct",
    nonce: Optional[str] = None,
    tool: bool = True,
    chats: int = 1,
    env: str = "1.0",
):
    """Synthetic retrieval → chat(s) → optional tool run."""
    rec = Recorder(out_dir=tmp_path, run_id=run_id)
    rec.environment["glassbox.env.packages"] = [f"lib=={env}"]
    with rec:
        with rec.span("retrieval", "kb", kind="INTERNAL") as s:
            rec.add_content(s, {"gen_ai.retrieval.documents": [doc]})
        for _ in range(chats):
            with rec.span(
                "chat", "m", attributes={"glassbox.answer.label": label}
            ) as s:
                if nonce is not None:
                    rec.add_content(s, {"gen_ai.output.messages": nonce})
        if tool:
            with rec.span(
                "execute_tool",
                "send",
                kind="INTERNAL",
                attributes={"gen_ai.tool.name": "send"},
            ):
                pass
    return rec.path


def _group(tmp_path, prefix: str, n: int, **kw) -> List:
    return [_run(tmp_path, f"{prefix}{i}", **kw) for i in range(n)]


def _field(report, step: str, field: str):
    matches = [f for f in report.fields if f.step == step and f.field == field]
    assert len(matches) == 1, (step, field, [(f.step, f.field) for f in report.fields])
    return matches[0]


# --- statistics helpers ---------------------------------------------------------


def test_tvd_basic():
    assert tvd(["x", "x"], ["x", "x"]) == 0.0
    assert tvd(["x", "x"], ["y", "y"]) == 1.0
    assert tvd(["x", "y"], ["x", "x"]) == pytest.approx(0.5)


def test_permutation_p_is_bounded_and_seeded():
    a, b = ["x"] * 10, ["y"] * 10
    p1 = permutation_p(a, b, n_perm=199, seed="s")
    p2 = permutation_p(a, b, n_perm=199, seed="s")
    assert p1 == p2
    assert p1 == pytest.approx(1 / 200)
    same = permutation_p(["x", "y"] * 5, ["x", "y"] * 5, n_perm=199, seed="s")
    assert same == 1.0


def test_holm_step_down():
    assert holm([0.01, 0.04, 0.03], alpha=0.05) == [True, False, False]
    assert holm([0.01, 0.02, 0.6], alpha=0.05) == [True, True, False]
    assert holm([], alpha=0.05) == []


# --- group diff -----------------------------------------------------------------


def test_identical_groups_have_nothing_significant(tmp_path):
    r = group_diff(_group(tmp_path, "a", 5), _group(tmp_path, "b", 5), n_perm=99)
    assert r.significant == []
    assert r.first_beyond_noise is None
    assert all(f.classification == "constant" for f in r.fields)
    assert r.environment == {}


def test_noise_with_same_distribution_is_not_significant(tmp_path):
    a = [
        _run(tmp_path, f"a{i}", label="correct" if i % 2 else "distractor")
        for i in range(10)
    ]
    b = [
        _run(tmp_path, f"b{i}", label="distractor" if i % 2 else "correct")
        for i in range(10)
    ]
    r = group_diff(a, b, n_perm=199)
    f = _field(r, "chat#0", "glassbox.answer.label")
    assert f.classification == "tested"
    assert f.varies_in_baseline
    assert not f.significant
    assert r.first_beyond_noise is None
    assert ("chat#0", "glassbox.answer.label") in r.noisy_in_baseline


def test_real_shift_is_significant_and_localised(tmp_path):
    a = _group(tmp_path, "a", 10, label="correct")
    b = _group(tmp_path, "b", 10, label="distractor")
    r = group_diff(a, b, n_perm=199)
    first = r.first_beyond_noise
    assert first is not None
    assert (first.step, first.field) == ("chat#0", "glassbox.answer.label")
    assert first.tvd == 1.0
    assert first.dist_a == {"correct": 10} and first.dist_b == {"distractor": 10}
    assert not any(f.step == "retrieval#0" for f in r.significant)


def test_unique_free_text_is_untestable(tmp_path):
    a = [_run(tmp_path, f"a{i}", nonce=f"text-a-{i}") for i in range(6)]
    b = [_run(tmp_path, f"b{i}", nonce=f"text-b-{i}") for i in range(6)]
    r = group_diff(a, b, n_perm=99)
    f = _field(r, "chat#0", "glassbox.content.gen_ai.output.messages.sha256")
    assert f.classification == "untestable"
    assert f.p_value is None and not f.significant
    assert (f.step, f.field) in r.untestable
    assert r.significant == []


def test_missing_step_shows_as_present_field(tmp_path):
    a = _group(tmp_path, "a", 8, tool=True)
    b = _group(tmp_path, "b", 8, tool=False)
    r = group_diff(a, b, n_perm=199)
    f = _field(r, "execute_tool:send#0", "present")
    assert f.significant
    assert f.dist_a == {"True": 8} and f.dist_b == {"False": 8}
    status = _field(r, "execute_tool:send#0", "status")
    assert status.dist_b == {"None": 8}


def test_all_significant_steps_reported_in_run_order(tmp_path):
    a = _group(tmp_path, "a", 8, doc="right", label="correct")
    b = _group(tmp_path, "b", 8, doc="wrong", label="distractor")
    r = group_diff(a, b, n_perm=199)
    steps = [f.step for f in r.significant]
    assert steps[0] == "retrieval#0"
    assert "chat#0" in steps
    assert r.first_beyond_noise.step == "retrieval#0"
    positions = [f.position for f in r.significant]
    assert positions == sorted(positions)


def test_repeated_signature_gets_occurrence_keys(tmp_path):
    r = group_diff(
        _group(tmp_path, "a", 3, chats=2), _group(tmp_path, "b", 3, chats=2), n_perm=9
    )
    keys = {f.step for f in r.fields}
    assert {"chat#0", "chat#1"} <= keys


def test_environment_reported_separately(tmp_path):
    a = _group(tmp_path, "a", 4, env="1.0")
    b = _group(tmp_path, "b", 4, env="2.0")
    r = group_diff(a, b, n_perm=99)
    assert "glassbox.env.packages" in r.environment
    assert r.significant == []


def test_result_is_deterministic(tmp_path):
    a = [_run(tmp_path, f"a{i}", label="correct" if i < 7 else "x") for i in range(10)]
    b = [_run(tmp_path, f"b{i}", label="correct" if i < 4 else "x") for i in range(10)]
    r1 = group_diff(a, b, n_perm=199, seed=3).to_dict()
    r2 = group_diff(a, b, n_perm=199, seed=3).to_dict()
    assert r1 == r2
    json.dumps(r1)


def test_groups_must_have_two_runs(tmp_path):
    one = _group(tmp_path, "a", 1)
    two = _group(tmp_path, "b", 2)
    with pytest.raises(ValueError):
        group_diff(one, two)
    with pytest.raises(ValueError):
        group_diff(two, one)


def test_cli_exit_codes(tmp_path, capsys):
    a = _group(tmp_path, "a", 6, label="correct")
    b = _group(tmp_path, "b", 6, label="distractor")
    same = _group(tmp_path, "c", 6, label="correct")
    args_same = ["--a", *map(str, a), "--b", *map(str, same), "--n-perm", "99"]
    assert main(args_same) == 0
    assert "No step differs beyond run-to-run noise" in capsys.readouterr().out
    args_diff = ["--a", *map(str, a), "--b", *map(str, b), "--n-perm", "99"]
    assert main(args_diff) == 1
    assert "chat#0" in capsys.readouterr().out
    assert main([*args_diff, "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["first_beyond_noise"]["step"] == "chat#0"


def test_cli_reports_untestable_and_environment(tmp_path, capsys):
    a = [_run(tmp_path, f"a{i}", nonce=f"ta{i}", env="1.0") for i in range(4)]
    b = [_run(tmp_path, f"b{i}", nonce=f"tb{i}", env="2.0") for i in range(4)]
    assert main(["--a", *map(str, a), "--b", *map(str, b), "--n-perm", "9"]) == 0
    out = capsys.readouterr().out
    assert "Untestable" in out and "gen_ai.output.messages" in out
    assert "Environment differs: glassbox.env.packages" in out


def test_groupdiff_module_imports_with_stdlib_only():
    import ast
    import sys
    from pathlib import Path

    import glassbox.v7.groupdiff as mod

    tree = ast.parse(Path(mod.__file__).read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    stdlib = set(sys.stdlib_module_names) | {"__future__", "glassbox"}
    assert roots <= stdlib, roots - stdlib
