"""Local trace recorder and reader (docs/v7/TRACE_SPEC.md v0.1).

Records one AI-system run as a tree of OpenTelemetry-shaped spans using the
OpenTelemetry GenAI semantic conventions (Development status upstream; see the spec,
§2) plus ``glassbox.*`` extension attributes. Content attributes are never stored raw
unless the run opts in (spec §6). Output is one JSON Lines file per run, ending in a
summary line carrying a SHA-256 of the preceding lines (integrity only, not
correctness).

Example:
    >>> with Recorder(out_dir="traces", run_id="r1", label="baseline") as rec:
    ...     with rec.span("chat", "gpt-x", content={"gen_ai.input.messages": msgs}):
    ...         reply = call_model(msgs)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)

SCHEMA = "glassbox.trace/0.1"
SEMCONV_SNAPSHOT = "otel-genai@main:2026-10-05"
POLICIES = ("hash_only", "redacted", "full")

#: OTel GenAI attributes marked Opt-In / sensitive (spec §4). They may only be passed
#: through the ``content`` channel so that the run's content policy always applies.
CONTENT_ATTRIBUTES = frozenset(
    {
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.tool.definitions",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
        "gen_ai.retrieval.query.text",
        "gen_ai.retrieval.documents",
    }
)

#: Best-effort redaction rules (spec §6). Not a guarantee of personal-data removal.
REDACTION_RULES: Dict[str, "re.Pattern[str]"] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "api_key": re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16})\b"),
}


class TraceIntegrityError(ValueError):
    """Raised when a trace file's summary hash does not match its content."""


def canonical_json(value: Any) -> str:
    """Serialise ``value`` deterministically (sorted keys, no whitespace)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _content_length(value: Any) -> int:
    if isinstance(value, (str, list, tuple, dict)):
        return len(value)
    return len(canonical_json(value))


def redact(value: Any) -> Any:
    """Return a copy of ``value`` with every string passed through the redaction rules."""
    if isinstance(value, str):
        for name, pattern in REDACTION_RULES.items():
            value = pattern.sub(f"[REDACTED:{name}]", value)
        return value
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


@dataclass
class _Span:
    """In-memory span; serialised to the OTLP-JSON-like dict of spec §8."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    kind: str
    start_time_unix_nano: int
    attributes: Dict[str, Any] = field(default_factory=dict)
    end_time_unix_nano: int = 0
    status: Dict[str, str] = field(default_factory=lambda: {"code": "OK"})

    def to_dict(self) -> Dict[str, Any]:
        """Return the span as a JSON-serialisable dict."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind,
            "start_time_unix_nano": self.start_time_unix_nano,
            "end_time_unix_nano": self.end_time_unix_nano,
            "status": self.status,
            "attributes": self.attributes,
        }


def environment_attributes() -> Dict[str, str]:
    """Reproduction context recorded on the root span (spec §7, minimal subset)."""
    return {
        "glassbox.env.python": sys.version.split()[0],
        "glassbox.env.platform": platform.platform(),
    }


class Recorder:
    """Record one run to ``<out_dir>/<run_id>.trace.jsonl``.

    Args:
        out_dir: Directory for the trace file (created if missing).
        run_id: Stable run identifier; a random UUID4 hex if omitted.
        label: Role in a comparison, e.g. ``"baseline"`` / ``"candidate"``.
        content_policy: ``"hash_only"`` (default), ``"redacted"`` or ``"full"``.
        root_name: Name of the root span.

    Raises:
        ValueError: If ``content_policy`` is unknown.
    """

    def __init__(
        self,
        out_dir: Union[str, "os.PathLike[str]"],
        run_id: Optional[str] = None,
        label: Optional[str] = None,
        content_policy: str = "hash_only",
        root_name: str = "glassbox.run",
    ) -> None:
        if content_policy not in POLICIES:
            raise ValueError(
                f"content_policy must be one of {POLICIES}, got {content_policy!r}"
            )
        self.run_id = run_id or uuid.uuid4().hex
        self.label = label
        self.policy = content_policy
        self.root_name = root_name
        self.path = Path(out_dir) / f"{self.run_id}.trace.jsonl"
        self._trace_id = uuid.uuid4().hex
        self._spans: List[_Span] = []
        self._stack: List[_Span] = []
        #: Extra reproduction context (spec §7), e.g. package versions. Keys must
        #: start with ``glassbox.env.``; set before entering the context manager.
        self.environment: Dict[str, Any] = {}

    # -- lifecycle ---------------------------------------------------------------

    def __enter__(self) -> "Recorder":
        if self.path.exists():
            raise FileExistsError(f"{self.path} exists; traces are never overwritten")
        root = self._open(self.root_name, "INTERNAL", self._root_attributes())
        self._stack.append(root)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> Literal[False]:
        root = self._stack.pop()
        self._close(root, exc)
        self._write()
        return False

    def _root_attributes(self) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {
            "glassbox.trace.schema": SCHEMA,
            "glassbox.semconv.snapshot": SEMCONV_SNAPSHOT,
            "glassbox.run.id": self.run_id,
            "glassbox.content.policy": self.policy,
        }
        if self.label is not None:
            attrs["glassbox.run.label"] = self.label
        if self.policy == "redacted":
            attrs["glassbox.redaction.rules"] = sorted(REDACTION_RULES)
        attrs.update(environment_attributes())
        bad = [k for k in self.environment if not k.startswith("glassbox.env.")]
        if bad:
            raise ValueError(f"environment keys must start with 'glassbox.env.': {bad}")
        attrs.update(self.environment)
        return attrs

    # -- spans -------------------------------------------------------------------

    @contextmanager
    def span(
        self,
        operation: str,
        target: Optional[str] = None,
        kind: str = "CLIENT",
        attributes: Optional[Dict[str, Any]] = None,
        content: Optional[Dict[str, Any]] = None,
    ) -> Iterator[_Span]:
        """Record a GenAI operation span named ``"{operation} {target}"``.

        Args:
            operation: OTel ``gen_ai.operation.name`` (e.g. ``chat``, ``retrieval``,
                ``execute_tool``, ``invoke_agent``).
            target: Model, data source, tool or agent name used in the span name.
            kind: OTel span kind (``CLIENT`` or ``INTERNAL``).
            attributes: Non-content attributes, stored as given.
            content: Opt-In content attributes, stored per the run's content policy.

        Raises:
            ValueError: If a content attribute is passed via ``attributes``.
        """
        attrs = dict(attributes or {})
        leaked = CONTENT_ATTRIBUTES.intersection(attrs)
        if leaked:
            raise ValueError(f"pass {sorted(leaked)} via content=, not attributes=")
        attrs["gen_ai.operation.name"] = operation
        attrs.update(self._apply_policy(content or {}))
        name = f"{operation} {target}" if target else operation
        with self._child(name, kind, attrs) as s:
            yield s

    @contextmanager
    def step(
        self, kind: str, attributes: Optional[Dict[str, Any]] = None
    ) -> Iterator[_Span]:
        """Record a non-GenAI step (routing, parsing, ...) as ``glassbox.step {kind}``."""
        attrs = dict(attributes or {})
        attrs["glassbox.step.kind"] = kind
        with self._child(f"glassbox.step {kind}", "INTERNAL", attrs) as s:
            yield s

    @contextmanager
    def _child(self, name: str, kind: str, attrs: Dict[str, Any]) -> Iterator[_Span]:
        if not self._stack:
            raise RuntimeError("Recorder must be entered (use 'with Recorder(...)')")
        s = self._open(name, kind, attrs)
        self._stack.append(s)
        try:
            yield s
        except BaseException as exc:
            self._stack.pop()
            self._close(s, exc)
            raise
        self._stack.pop()
        self._close(s, None)

    def add_content(self, span: _Span, content: Dict[str, Any]) -> None:
        """Attach content to an open span under the run's content policy.

        For content known only after a span starts (e.g. retrieved documents).

        Raises:
            ValueError: If a key is not an OTel GenAI content attribute.
        """
        span.attributes.update(self._apply_policy(content))

    def _apply_policy(self, content: Dict[str, Any]) -> Dict[str, Any]:
        unknown = set(content) - CONTENT_ATTRIBUTES
        if unknown:
            raise ValueError(f"not content attributes: {sorted(unknown)}")
        out: Dict[str, Any] = {}
        for attr, value in content.items():
            out[f"glassbox.content.{attr}.sha256"] = _sha256_text(canonical_json(value))
            out[f"glassbox.content.{attr}.length"] = _content_length(value)
            if self.policy == "full":
                out[attr] = value
            elif self.policy == "redacted":
                out[attr] = redact(value)
        return out

    def _open(self, name: str, kind: str, attrs: Dict[str, Any]) -> _Span:
        parent = self._stack[-1].span_id if self._stack else None
        attrs["glassbox.step.index"] = len(self._spans)
        s = _Span(
            trace_id=self._trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=parent,
            name=name,
            kind=kind,
            start_time_unix_nano=time.time_ns(),
            attributes=attrs,
        )
        self._spans.append(s)
        return s

    def _close(self, s: _Span, exc: Optional[BaseException]) -> None:
        s.end_time_unix_nano = time.time_ns()
        if exc is not None:
            s.status = {"code": "ERROR", "message": type(exc).__name__}
            s.attributes["error.type"] = type(exc).__name__

    # -- output ------------------------------------------------------------------

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [canonical_json(s.to_dict()) for s in self._spans]
        body = "".join(line + "\n" for line in lines)
        summary = {
            "glassbox.trace.summary": {
                "schema": SCHEMA,
                "span_count": len(lines),
                "sha256": _sha256_text(body),
            }
        }
        with open(self.path, "x", encoding="utf-8") as fh:
            fh.write(body + canonical_json(summary) + "\n")
        logger.info("wrote trace %s (%d spans)", self.path, len(lines))


def read_trace(
    path: Union[str, "os.PathLike[str]"],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read a trace file and verify its integrity summary.

    Returns:
        ``(spans, summary)``; spans in recording order.

    Raises:
        TraceIntegrityError: If the summary is missing or its hash or count mismatches.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        raise TraceIntegrityError(f"{path}: empty trace")
    summary = json.loads(lines[-1]).get("glassbox.trace.summary")
    if summary is None:
        raise TraceIntegrityError(f"{path}: missing summary line")
    body = "".join(line + "\n" for line in lines[:-1])
    if (
        _sha256_text(body) != summary["sha256"]
        or len(lines) - 1 != summary["span_count"]
    ):
        raise TraceIntegrityError(f"{path}: content does not match its summary")
    return [json.loads(line) for line in lines[:-1]], summary
