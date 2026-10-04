"""Compare two traces and locate their first divergence (docs/v7/TRACE_SPEC.md §9).

Steps of both runs are aligned by a *step signature*: the GenAI operation (plus the
tool name for ``execute_tool``) or the custom step kind. Alignment uses
``difflib.SequenceMatcher``, so an inserted or dropped step does not shift every
later comparison. Aligned steps are then compared on their attributes, content hashes
and status.

Scope of the claim: a diff says **where** two runs differ, never **why**. A
divergence is an observation; whether it caused a behaviour change needs a controlled
experiment (V7 Intervention, not built). Raw content is never placed in a report,
only attribute names and hash prefixes.

Usage:
    python -m glassbox.v7.diff baseline.trace.jsonl candidate.trace.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from glassbox.v7.trace import CONTENT_ATTRIBUTES, read_trace

PathLike = Union[str, "os.PathLike[str]"]
Span = Dict[str, Any]

#: Attributes that differ between any two runs by construction.
VOLATILE = frozenset({"glassbox.step.index", "glassbox.run.id", "glassbox.run.label"})
_CONTENT_PREFIX = "glassbox.content."
_HASH_SUFFIX = ".sha256"
_LEN_SUFFIX = ".length"


@dataclass
class Divergence:
    """One difference between aligned (or unaligned) steps.

    ``kind`` is one of ``added``, ``removed``, ``status``, ``attribute``, ``content``.
    ``step_a`` / ``step_b`` are ``glassbox.step.index`` values (``None`` if absent).
    ``details`` maps attribute name to ``(value_a, value_b)``. For content, the values
    are 12-character hash prefixes, never content.
    """

    kind: str
    step_a: Optional[int]
    step_b: Optional[int]
    name_a: Optional[str]
    name_b: Optional[str]
    details: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)


@dataclass
class DiffReport:
    """Result of :func:`diff_traces`.

    ``divergences`` are behavioural differences in run order. ``environment`` holds
    root-span differences (environment, schema, content policy). They are reported
    separately and never counted as the first divergence.
    """

    divergences: List[Divergence]
    environment: Dict[str, Tuple[Any, Any]]

    @property
    def first_divergence(self) -> Optional[Divergence]:
        """Earliest behavioural divergence, or ``None``."""
        return self.divergences[0] if self.divergences else None

    @property
    def identical(self) -> bool:
        """True only if neither behaviour nor environment differ."""
        return not self.divergences and not self.environment

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form (tuples become lists)."""
        first = self.first_divergence
        return {
            "identical": self.identical,
            "first_divergence": asdict(first) if first else None,
            "divergences": [asdict(d) for d in self.divergences],
            "environment": {k: list(v) for k, v in self.environment.items()},
        }


def step_signature(span: Span) -> str:
    """Alignment key for a span: operation (+ tool name) or custom step kind."""
    attrs = span["attributes"]
    op = attrs.get("gen_ai.operation.name")
    if op == "execute_tool":
        return f"execute_tool:{attrs.get('gen_ai.tool.name')}"
    if op:
        return str(op)
    return f"step:{attrs.get('glassbox.step.kind', span['name'])}"


def _comparable(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop volatile keys, raw content and error.type (reported with status)."""
    return {
        k: v
        for k, v in attrs.items()
        if k not in VOLATILE and k not in CONTENT_ATTRIBUTES and k != "error.type"
    }


def _pair_diffs(a: Span, b: Span) -> List[Divergence]:
    """Divergences between two aligned spans, ordered status → content → attribute."""
    out: List[Divergence] = []
    ia, ib = (
        a["attributes"]["glassbox.step.index"],
        b["attributes"]["glassbox.step.index"],
    )

    def make(kind: str, details: Dict[str, Tuple[Any, Any]]) -> Divergence:
        return Divergence(kind, ia, ib, a["name"], b["name"], details)

    sa, sb = a["status"]["code"], b["status"]["code"]
    if sa != sb:
        details = {"status": (sa, sb)}
        details["error.type"] = (
            a["attributes"].get("error.type"),
            b["attributes"].get("error.type"),
        )
        out.append(make("status", details))
    ca, cb = _comparable(a["attributes"]), _comparable(b["attributes"])
    content: Dict[str, Tuple[Any, Any]] = {}
    plain: Dict[str, Tuple[Any, Any]] = {}
    for key in sorted(set(ca) | set(cb)):
        va, vb = ca.get(key), cb.get(key)
        if va == vb:
            continue
        if key.startswith(_CONTENT_PREFIX) and key.endswith(_HASH_SUFFIX):
            attr = key[len(_CONTENT_PREFIX) : -len(_HASH_SUFFIX)]
            content[attr] = (str(va)[:12] if va else None, str(vb)[:12] if vb else None)
        elif not (key.startswith(_CONTENT_PREFIX) and key.endswith(_LEN_SUFFIX)):
            plain[key] = (va, vb)
    if content:
        out.append(make("content", content))
    if plain:
        out.append(make("attribute", plain))
    return out


def _unaligned(kind: str, span: Span) -> Divergence:
    idx = span["attributes"]["glassbox.step.index"]
    if kind == "removed":
        return Divergence("removed", idx, None, span["name"], None)
    return Divergence("added", None, idx, None, span["name"])


def _environment(root_a: Span, root_b: Span) -> Dict[str, Tuple[Any, Any]]:
    ca, cb = _comparable(root_a["attributes"]), _comparable(root_b["attributes"])
    return {
        k: (ca.get(k), cb.get(k))
        for k in sorted(set(ca) | set(cb))
        if ca.get(k) != cb.get(k) and k != "gen_ai.operation.name"
    }


def diff_traces(path_a: PathLike, path_b: PathLike) -> DiffReport:
    """Compare two trace files (both are integrity-checked on read).

    Raises:
        TraceIntegrityError: If either file fails its integrity check.
    """
    spans_a, _ = read_trace(path_a)
    spans_b, _ = read_trace(path_b)
    root_a, steps_a = spans_a[0], spans_a[1:]
    root_b, steps_b = spans_b[0], spans_b[1:]
    sig_a = [step_signature(s) for s in steps_a]
    sig_b = [step_signature(s) for s in steps_b]
    matcher = difflib.SequenceMatcher(a=sig_a, b=sig_b, autojunk=False)
    divergences: List[Divergence] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for a, b in zip(steps_a[i1:i2], steps_b[j1:j2]):
                divergences.extend(_pair_diffs(a, b))
            continue
        divergences.extend(_unaligned("removed", s) for s in steps_a[i1:i2])
        divergences.extend(_unaligned("added", s) for s in steps_b[j1:j2])
    return DiffReport(divergences, _environment(root_a, root_b))


def _describe(d: Divergence) -> str:
    where = f"step {d.step_a if d.step_a is not None else '-'}→{d.step_b if d.step_b is not None else '-'}"
    name = d.name_a or d.name_b
    if d.kind in ("added", "removed"):
        return f"{where} [{d.kind}] {name}"
    changes = ", ".join(f"{k}: {va!r} → {vb!r}" for k, (va, vb) in d.details.items())
    return f"{where} [{d.kind}] {name}: {changes}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: exit 0 if no behavioural divergence, 1 otherwise."""
    parser = argparse.ArgumentParser(description="Diff two Glassbox V7 traces.")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument(
        "--json", action="store_true", help="print the full report as JSON"
    )
    args = parser.parse_args(argv)
    report = diff_traces(args.baseline, args.candidate)
    if args.json:
        sys.stdout.write(json.dumps(report.to_dict(), indent=2) + "\n")
        return 1 if report.divergences else 0
    first = report.first_divergence
    if first is None:
        sys.stdout.write("No behavioural divergence.\n")
    else:
        sys.stdout.write(f"First divergence: {_describe(first)}\n")
        sys.stdout.write(f"Total divergences: {len(report.divergences)}\n")
        for d in report.divergences[1:]:
            sys.stdout.write(f"  {_describe(d)}\n")
    for k, (va, vb) in report.environment.items():
        sys.stdout.write(f"Environment differs: {k}: {va!r} → {vb!r}\n")
    sys.stdout.write("Note: a divergence is where runs differ, not proof of why.\n")
    return 1 if report.divergences else 0


if __name__ == "__main__":
    sys.exit(main())
