"""Tests for the V7 trace recorder (docs/v7/TRACE_SPEC.md v0.1)."""

from __future__ import annotations

import hashlib
import json

import pytest

from glassbox.v7.trace import (
    SCHEMA,
    Recorder,
    TraceIntegrityError,
    canonical_json,
    read_trace,
    redact,
)

PROMPT = [
    {"role": "user", "parts": [{"type": "text", "content": "Where is order 42?"}]}
]
SECRET_EMAIL = "alice@example.com"


def _sha(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _record_simple(tmp_path, policy="hash_only"):
    with Recorder(
        out_dir=tmp_path, run_id="run1", label="baseline", content_policy=policy
    ) as rec:
        with rec.span("retrieval", "kb-prod", attributes={"gen_ai.retrieval.top_k": 5}):
            pass
        with rec.span(
            "chat",
            "gpt-x",
            kind="CLIENT",
            attributes={"gen_ai.request.model": "gpt-x"},
            content={"gen_ai.input.messages": PROMPT},
        ):
            with rec.span(
                "execute_tool",
                "get_order",
                kind="INTERNAL",
                attributes={"gen_ai.tool.name": "get_order"},
            ):
                pass
    return rec.path


# --- structure -----------------------------------------------------------------


def test_round_trip_preserves_spans(tmp_path):
    path = _record_simple(tmp_path)
    spans, summary = read_trace(path)
    assert summary["span_count"] == len(spans) == 4  # root + 3
    assert [s["attributes"]["glassbox.step.index"] for s in spans] == [0, 1, 2, 3]


def test_root_span_carries_run_metadata(tmp_path):
    spans, _ = read_trace(_record_simple(tmp_path))
    root = spans[0]
    assert root["parent_span_id"] is None
    attrs = root["attributes"]
    assert attrs["glassbox.trace.schema"] == SCHEMA
    assert attrs["glassbox.run.id"] == "run1"
    assert attrs["glassbox.run.label"] == "baseline"
    assert attrs["glassbox.content.policy"] == "hash_only"
    assert attrs["glassbox.semconv.snapshot"].startswith("otel-genai@")
    assert "glassbox.env.python" in attrs


def test_span_names_follow_otel_rule(tmp_path):
    spans, _ = read_trace(_record_simple(tmp_path))
    names = [s["name"] for s in spans[1:]]
    assert names == ["retrieval kb-prod", "chat gpt-x", "execute_tool get_order"]
    assert spans[1]["attributes"]["gen_ai.operation.name"] == "retrieval"


def test_nesting_sets_parent_ids_and_one_trace_id(tmp_path):
    spans, _ = read_trace(_record_simple(tmp_path))
    root, retrieval, chat, tool = spans
    assert retrieval["parent_span_id"] == root["span_id"]
    assert chat["parent_span_id"] == root["span_id"]
    assert tool["parent_span_id"] == chat["span_id"]
    assert len({s["trace_id"] for s in spans}) == 1
    assert len({s["span_id"] for s in spans}) == 4


def test_timestamps_are_ordered(tmp_path):
    spans, _ = read_trace(_record_simple(tmp_path))
    for s in spans:
        assert s["end_time_unix_nano"] >= s["start_time_unix_nano"]


# --- privacy (spec §6) -----------------------------------------------------------


def test_hash_only_is_default_and_stores_no_content(tmp_path):
    with Recorder(out_dir=tmp_path, run_id="r") as rec:
        with rec.span("chat", "m", content={"gen_ai.input.messages": PROMPT}):
            pass
    raw = rec.path.read_text(encoding="utf-8")
    assert "Where is order 42" not in raw
    spans, _ = read_trace(rec.path)
    attrs = spans[1]["attributes"]
    assert "gen_ai.input.messages" not in attrs
    assert attrs["glassbox.content.gen_ai.input.messages.sha256"] == _sha(PROMPT)
    assert attrs["glassbox.content.gen_ai.input.messages.length"] == 1


def test_full_policy_stores_content(tmp_path):
    spans, _ = read_trace(_record_simple(tmp_path, policy="full"))
    assert spans[2]["attributes"]["gen_ai.input.messages"] == PROMPT


def test_redacted_policy_removes_email_before_writing(tmp_path):
    msg = [
        {"role": "user", "parts": [{"type": "text", "content": f"mail {SECRET_EMAIL}"}]}
    ]
    with Recorder(out_dir=tmp_path, run_id="r", content_policy="redacted") as rec:
        with rec.span("chat", "m", content={"gen_ai.input.messages": msg}):
            pass
    raw = rec.path.read_text(encoding="utf-8")
    assert SECRET_EMAIL not in raw
    spans, _ = read_trace(rec.path)
    assert "email" in spans[0]["attributes"]["glassbox.redaction.rules"]
    assert "[REDACTED:email]" in json.dumps(
        spans[1]["attributes"]["gen_ai.input.messages"]
    )


def test_content_attribute_cannot_bypass_policy_via_attributes(tmp_path):
    with Recorder(out_dir=tmp_path, run_id="r") as rec:
        with pytest.raises(ValueError, match="content"):
            with rec.span("chat", "m", attributes={"gen_ai.input.messages": PROMPT}):
                pass


def test_unknown_policy_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        Recorder(out_dir=tmp_path, run_id="r", content_policy="everything")


# --- errors ----------------------------------------------------------------------


def test_exception_marks_span_error_and_propagates(tmp_path):
    with pytest.raises(RuntimeError):
        with Recorder(out_dir=tmp_path, run_id="r") as rec:
            with rec.span("execute_tool", "flaky"):
                raise RuntimeError("boom")
    spans, _ = read_trace(rec.path)
    tool = spans[1]
    assert tool["status"]["code"] == "ERROR"
    assert tool["attributes"]["error.type"] == "RuntimeError"
    assert spans[0]["status"]["code"] == "ERROR"


def test_custom_step_uses_glassbox_name(tmp_path):
    with Recorder(out_dir=tmp_path, run_id="r") as rec:
        with rec.step("route"):
            pass
    spans, _ = read_trace(rec.path)
    assert spans[1]["name"] == "glassbox.step route"
    assert spans[1]["attributes"]["glassbox.step.kind"] == "route"


# --- integrity -------------------------------------------------------------------


def test_tampering_is_detected(tmp_path):
    path = _record_simple(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace("kb-prod", "kb-evil")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(TraceIntegrityError):
        read_trace(path)


def test_recorder_refuses_to_overwrite_existing_trace(tmp_path):
    _record_simple(tmp_path)
    with pytest.raises(FileExistsError):
        _record_simple(tmp_path)


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_unknown_content_attribute_is_rejected(tmp_path):
    with Recorder(out_dir=tmp_path, run_id="r") as rec:
        with pytest.raises(ValueError, match="not content attributes"):
            with rec.span("chat", "m", content={"my.secret": "x"}):
                pass


def test_span_outside_recorder_context_is_an_error(tmp_path):
    rec = Recorder(out_dir=tmp_path, run_id="r")
    with pytest.raises(RuntimeError):
        with rec.span("chat", "m"):
            pass


def test_redaction_recurses_and_leaves_non_strings():
    value = {"a": ["key sk-ABCDEFGHIJKLMNOPQRST1234"], "n": 3, "t": ("x@y.io",)}
    out = redact(value)
    assert out["n"] == 3
    assert out["a"] == ["key [REDACTED:api_key]"]
    assert out["t"] == ["[REDACTED:email]"]


def test_hash_length_of_scalar_content_uses_canonical_json(tmp_path):
    with Recorder(out_dir=tmp_path, run_id="r") as rec:
        with rec.span(
            "execute_tool", "calc", content={"gen_ai.tool.call.result": 12345}
        ):
            pass
    spans, _ = read_trace(rec.path)
    assert (
        spans[1]["attributes"]["glassbox.content.gen_ai.tool.call.result.length"] == 5
    )


def test_empty_or_summaryless_file_fails_integrity(tmp_path):
    empty = tmp_path / "a.trace.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(TraceIntegrityError):
        read_trace(empty)
    nosum = tmp_path / "b.trace.jsonl"
    nosum.write_text('{"span_id": "x"}\n', encoding="utf-8")
    with pytest.raises(TraceIntegrityError):
        read_trace(nosum)


def test_add_content_applies_policy_to_an_open_span(tmp_path):
    docs = [{"id": "d1", "score": 0.9}]
    with Recorder(out_dir=tmp_path, run_id="r") as rec:
        with rec.span("retrieval", "kb") as s:
            rec.add_content(s, {"gen_ai.retrieval.documents": docs})
    spans, _ = read_trace(rec.path)
    attrs = spans[1]["attributes"]
    assert "gen_ai.retrieval.documents" not in attrs  # hash_only default
    assert attrs["glassbox.content.gen_ai.retrieval.documents.length"] == 1
    with pytest.raises(ValueError, match="not content attributes"):
        rec.add_content(s, {"x": 1})
