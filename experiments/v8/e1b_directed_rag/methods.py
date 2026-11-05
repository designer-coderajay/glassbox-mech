"""The four methods of E1b (PROTOCOL.md §5) on one shared run budget.

GB is ``glassbox.v7.investigate`` with the incident's target ``correct``; its blame
set is the *harmful* list (PROTOCOL §3). Its undirected ``attributed`` list, the
helpful list and the per-factor direction are kept for the descriptive results.
B0, B1 and B2 are E1's scripted baselines, unchanged, and reuse runs from GB's cells
(run index 0, plus 1–5 for B2's instability check).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from glassbox.v7.diff import diff_traces
from glassbox.v7.intervene import InterventionSettings
from glassbox.v7.investigate import Investigation, investigate
from glassbox.v7.trace import read_trace

HERE = Path(__file__).resolve().parent
B1_OWNERS = {"retrieval": ("index", "chunk_words"), "chat": ("template",)}
B2_EXTRA_RUNS = 5


def _system() -> Any:
    key = "e1b_system"
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
    """B1: owners of the first divergent step of one baseline/candidate run pair."""
    first = diff_traces(base_run, cand_run).first_divergence
    if first is None:
        return []
    op = _operation(base_run, first.step_a) or _operation(cand_run, first.step_b)
    owners = B1_OWNERS.get(op or "", ())
    return [f for f in changed if f in owners]


def b2_blame(candidate: str, reverted: Dict[str, str]) -> List[str]:
    """B2: with the candidate run wrong, blame factors whose reverted run is correct."""
    if candidate != "wrong":
        return []
    return [f for f, label in reverted.items() if label == "correct"]


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
    return rows
