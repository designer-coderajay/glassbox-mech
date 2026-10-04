"""Tests for V7 trace diff / first divergence (docs/v7/TRACE_SPEC.md §9)."""

from __future__ import annotations

import json

import pytest

from glassbox.v7.diff import diff_traces, main
from glassbox.v7.trace import Recorder

Q = [{"role": "user", "parts": [{"type": "text", "content": "Where is order 42?"}]}]
Q2 = [{"role": "user", "parts": [{"type": "text", "content": "Where is order 43?"}]}]


def _run(
    tmp_path, run_id, *, model="gpt-x", top_k=5, query=Q, tool=True, fail_tool=False
):
    """Synthetic RAG + tool run; each keyword introduces one controlled change."""
    try:
        with Recorder(out_dir=tmp_path, run_id=run_id) as rec:
            with rec.span(
                "retrieval",
                "kb",
                attributes={"gen_ai.retrieval.top_k": top_k},
                content={"gen_ai.retrieval.query.text": query},
            ):
                pass
            with rec.span("chat", model, attributes={"gen_ai.request.model": model}):
                pass
            if tool:
                with rec.span(
                    "execute_tool",
                    "get_order",
                    kind="INTERNAL",
                    attributes={"gen_ai.tool.name": "get_order"},
                ):
                    if fail_tool:
                        raise RuntimeError("tool down")
            with rec.span("chat", model, attributes={"gen_ai.request.model": model}):
                pass
    except RuntimeError:
        pass
    return rec.path


def test_identical_runs_have_no_divergence(tmp_path):
    a, b = _run(tmp_path, "a"), _run(tmp_path, "b")
    report = diff_traces(a, b)
    assert report.first_divergence is None
    assert report.divergences == []
    assert report.identical


def test_volatile_fields_are_ignored(tmp_path):
    """Different run ids, span ids, trace ids and timestamps are not divergences."""
    report = diff_traces(_run(tmp_path, "a"), _run(tmp_path, "b"))
    assert report.identical


def test_attribute_change_is_located_at_its_step(tmp_path):
    report = diff_traces(_run(tmp_path, "a"), _run(tmp_path, "b", top_k=3))
    first = report.first_divergence
    assert first.kind == "attribute"
    assert first.name_a == "retrieval kb"
    assert first.details == {"gen_ai.retrieval.top_k": (5, 3)}


def test_content_change_detected_via_hash_without_content(tmp_path):
    a, b = _run(tmp_path, "a"), _run(tmp_path, "b", query=Q2)
    report = diff_traces(a, b)
    first = report.first_divergence
    assert first.kind == "content"
    assert list(first.details) == ["gen_ai.retrieval.query.text"]
    assert "order 4" not in json.dumps(report.to_dict())


def test_model_change_reports_first_affected_step_only_as_first(tmp_path):
    report = diff_traces(_run(tmp_path, "a"), _run(tmp_path, "b", model="gpt-y"))
    first = report.first_divergence
    assert first.kind == "attribute"
    assert first.step_a == 2
    assert first.details["gen_ai.request.model"] == ("gpt-x", "gpt-y")
    assert len(report.divergences) == 2  # both chat steps differ


def test_missing_step_is_structural(tmp_path):
    report = diff_traces(_run(tmp_path, "a"), _run(tmp_path, "b", tool=False))
    first = report.first_divergence
    assert first.kind == "removed"
    assert first.name_a == "execute_tool get_order"
    assert first.step_b is None
    assert len(report.divergences) == 1  # later steps re-align


def test_added_step_is_structural(tmp_path):
    report = diff_traces(_run(tmp_path, "a", tool=False), _run(tmp_path, "b"))
    first = report.first_divergence
    assert first.kind == "added"
    assert first.name_b == "execute_tool get_order"


def test_status_change_is_reported(tmp_path):
    report = diff_traces(_run(tmp_path, "a"), _run(tmp_path, "b", fail_tool=True))
    kinds = [d.kind for d in report.divergences]
    assert report.first_divergence.kind == "status"
    assert report.first_divergence.details["status"] == ("OK", "ERROR")
    assert "status" in kinds


def test_earliest_divergence_wins(tmp_path):
    report = diff_traces(
        _run(tmp_path, "a"), _run(tmp_path, "b", top_k=3, model="gpt-y")
    )
    assert report.first_divergence.name_a == "retrieval kb"


def test_environment_differences_are_separate_from_divergences(tmp_path, monkeypatch):
    a = _run(tmp_path, "a")
    monkeypatch.setattr(
        "glassbox.v7.trace.environment_attributes",
        lambda: {"glassbox.env.python": "0.0.0", "glassbox.env.platform": "test"},
    )
    b = _run(tmp_path, "b")
    report = diff_traces(a, b)
    assert report.first_divergence is None
    assert report.environment["glassbox.env.python"][1] == "0.0.0"
    assert not report.identical  # environment differs, behaviour does not


def test_report_is_json_serialisable(tmp_path):
    report = diff_traces(_run(tmp_path, "a"), _run(tmp_path, "b", top_k=3))
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["first_divergence"]["kind"] == "attribute"
    assert payload["identical"] is False


def test_cli_exit_codes(tmp_path, capsys):
    a, b = _run(tmp_path, "a"), _run(tmp_path, "b")
    c = _run(tmp_path, "c", top_k=3)
    assert main([str(a), str(b)]) == 0
    assert main([str(a), str(c)]) == 1
    out = capsys.readouterr().out
    assert "first divergence" in out.lower()


def test_tampered_trace_is_refused(tmp_path):
    a, b = _run(tmp_path, "a"), _run(tmp_path, "b")
    b.write_text(b.read_text(encoding="utf-8").replace("kb", "kz"), encoding="utf-8")
    from glassbox.v7.trace import TraceIntegrityError

    with pytest.raises(TraceIntegrityError):
        diff_traces(a, b)


def test_custom_steps_align_by_kind(tmp_path):
    def run(rid, kinds):
        with Recorder(out_dir=tmp_path, run_id=rid) as rec:
            for k in kinds:
                with rec.step(k):
                    pass
        return rec.path

    report = diff_traces(
        run("a", ["route", "parse"]), run("b", ["route", "rank", "parse"])
    )
    assert [(d.kind, d.name_b) for d in report.divergences] == [
        ("added", "glassbox.step rank")
    ]


def test_cli_text_lists_all_divergences_and_environment(tmp_path, capsys, monkeypatch):
    a = _run(tmp_path, "a")
    monkeypatch.setattr(
        "glassbox.v7.trace.environment_attributes",
        lambda: {"glassbox.env.python": "0.0.0", "glassbox.env.platform": "t"},
    )
    b = _run(tmp_path, "b", tool=False, model="gpt-y")
    assert main([str(a), str(b)]) == 1
    out = capsys.readouterr().out
    assert "[removed] execute_tool get_order" in out
    assert "Total divergences: 3" in out
    assert "Environment differs: glassbox.env.python" in out
    assert "not proof of why" in out


def test_cli_json_output(tmp_path, capsys):
    a, b = _run(tmp_path, "a"), _run(tmp_path, "b", top_k=3)
    assert main(["--json", str(a), str(b)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["first_divergence"]["details"]["gen_ai.retrieval.top_k"] == [5, 3]
    assert main(["--json", str(a), str(a)]) == 0
