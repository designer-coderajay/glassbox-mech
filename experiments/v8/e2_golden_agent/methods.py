"""The four methods of E2 (PROTOCOL.md §6) on one shared run budget.

GB is ``glassbox.v7.investigate`` with target ``correct`` and the required tools per
question; its blame set is the *harmful* list, and its fault label for a factor is the
most common label among the failing runs of the ``only_<f>`` cell (``None`` below 5
failing runs). Its undirected ``attributed`` list is kept for ``undirected_fix``.
B0, B1 and B2 reuse runs from GB's cells (run 0, plus 1–5 for B2's flip check).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from glassbox.v7.diff import diff_traces
from glassbox.v7.intervene import InterventionSettings
from glassbox.v7.investigate import Investigation, investigate
from glassbox.v7.taxonomy import dominant
from glassbox.v7.trace import read_trace

HERE = Path(__file__).resolve().parent
B2_EXTRA_RUNS = 5
MIN_FAILING = 5


def _system() -> Any:
    key = "e2_system"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / "system.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def _operation(path: str, step: Optional[int]) -> Optional[str]:
    if step is None:
        return None
    spans, _ = read_trace(path)
    for s in spans:
        if s["attributes"].get("glassbox.step.index") == step:
            op = s["attributes"].get("gen_ai.operation.name")
            return str(op) if op else None
    return None


def b1_blame(base_run: str, cand_run: str, changed: Sequence[str]) -> List[str]:
    """B1 (PROTOCOL §6): what the first divergence of one run pair touches.

    A step present in only one run → episode length → ``max_steps``; a divergence in
    an LLM step → every prompt-visible factor; in a tool step → the tool-result
    factors. ``log_level`` is never blamed.
    """
    sysm = _system()
    first = diff_traces(base_run, cand_run).first_divergence
    if first is None:
        return []
    if first.kind in ("added", "removed"):
        owners: Sequence[str] = ("max_steps",)
    else:
        op = _operation(base_run, first.step_a) or _operation(cand_run, first.step_b)
        owners = {
            "chat": sysm.PROMPT_VISIBLE,
            "execute_tool": sysm.TOOL_RESULT_FACTORS,
        }.get(op or "", ())
    return [f for f in changed if f in owners]


def b2_blame(candidate: str, reverted: Dict[str, str]) -> List[str]:
    """B2: with the candidate run wrong, blame factors whose reverted run is correct."""
    if candidate != "wrong":
        return []
    return [f for f, label in reverted.items() if label == "correct"]


def fault_labels(
    fault_counts: Dict[str, Dict[str, int]], changed: Sequence[str]
) -> Dict[str, Optional[str]]:
    """GB's fault label per factor: dominant label of its ``only_<f>`` cell."""
    return {f: dominant(fault_counts[f"only_{f}"], MIN_FAILING) for f in changed}


def _row(inv_finding: Any, changed: Sequence[str]) -> Dict[str, Any]:
    sysm = _system()
    paths = inv_finding.intervention["paths"]

    def label(cell: str, i: int) -> str:
        return str(sysm.outcome(Path(paths[cell][i])))

    def b2_at(i: int) -> List[str]:
        reverted = {f: label(f"all_but_{f}", i) for f in changed}
        return b2_blame(label("candidate", i), reverted)

    first = b2_at(0)
    flips = sum(b2_at(i) != first for i in range(1, B2_EXTRA_RUNS + 1))
    return {
        "GB": list(inv_finding.harmful or []),
        "GB_helpful": list(inv_finding.helpful or []),
        "GB_attributed": list(inv_finding.attributed),
        "GB_reverting_matters": list(inv_finding.reverting_matters),
        "GB_fault": fault_labels(inv_finding.fault_counts, changed),
        "fault_counts": inv_finding.fault_counts,
        "direction": inv_finding.direction,
        "B0": list(changed),
        "B1": b1_blame(paths["baseline"][0], paths["candidate"][0], changed),
        "B2": first,
        "B2_flips": flips,
        "cells": inv_finding.outcome_counts,
        "first_beyond_noise": inv_finding.first_beyond_noise,
    }


def run_methods(
    system: Any,
    questions: Sequence[Dict[str, Any]],
    out_dir: Path,
    settings: InterventionSettings,
) -> Dict[str, Dict[str, Any]]:
    """Run GB on ``questions`` and derive B0–B2 from the same runs.

    Raises:
        ValueError: If ``settings.n_per_cell`` is too small for B2's flip check.
    """
    if settings.n_per_cell < B2_EXTRA_RUNS + 1:
        raise ValueError(f"n_per_cell must be at least {B2_EXTRA_RUNS + 1}")
    sysm = _system()
    inv: Investigation = investigate(
        sysm.make_incident(system, questions), Path(out_dir) / "work", settings
    )
    changed = [f.name for f in sysm.FACTORS]
    rows = {c.case: _row(c, changed) for c in inv.findings}
    for case, timing in inv.timings.items():
        rows[case]["timing"] = timing
    system.clear()
    return rows
