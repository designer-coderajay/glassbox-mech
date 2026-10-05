#!/usr/bin/env python3
"""Evaluate V7 Experiment 2 traces against PROTOCOL.md §5 (main environment).

Usage:
    python3 experiments/v7/llama_dedup/evaluate.py --results experiments/v7/llama_dedup/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from glassbox.v7.diff import diff_traces
from glassbox.v7.trace import read_trace

BUG, FIX = "0.14.17", "0.14.19"
PATHS = ("sync", "async")
RUNS = ("r1", "r2")


def trace_path(results: Path, version: str, path: str, run: str) -> Path:
    """Location written by subject.py."""
    return results / version / path / f"{run}.trace.jsonl"


def _first_is_retrieval(a: Path, b: Path) -> bool:
    first = diff_traces(a, b).first_divergence
    return first is not None and (first.name_a or "").startswith("retrieval")


def _silent(a: Path, b: Path) -> bool:
    return not diff_traces(a, b).divergences


def _n_documents(p: Path) -> Optional[int]:
    spans, _ = read_trace(p)
    for s in spans:
        n = s["attributes"].get("glassbox.content.gen_ai.retrieval.documents.length")
        if n is not None:
            return int(n)
    return None


def evaluate(results: Path) -> Dict[str, Any]:
    """Compute C1–C5 and the overall verdict; no files are written."""

    def t(version: str, path: str, run: str = "r1") -> Path:
        return trace_path(results, version, path, run)

    crit: Dict[str, Any] = {
        "C1_localisation_bug": {
            "pass": _first_is_retrieval(t(BUG, "async"), t(BUG, "sync"))
        },
        "C2_fix_confirmed": {"pass": _silent(t(FIX, "async"), t(FIX, "sync"))},
        "C3a_intervention_sync": {
            "pass": _first_is_retrieval(t(BUG, "sync"), t(FIX, "sync"))
        },
        "C3b_specificity_async": {"pass": _silent(t(BUG, "async"), t(FIX, "async"))},
    }
    det = {
        f"{v}/{p}": _silent(t(v, p, "r1"), t(v, p, "r2"))
        for v in (BUG, FIX)
        for p in PATHS
    }
    crit["C4_determinism"] = {"per_condition": det, "pass": all(det.values())}
    crit["C5_n_documents"] = {
        f"{v}/{p}": _n_documents(t(v, p)) for v in (BUG, FIX) for p in PATHS
    }
    gated = [k for k in crit if k != "C5_n_documents"]
    env = diff_traces(t(BUG, "sync"), t(FIX, "sync")).environment
    return {
        "protocol": "experiments/v7/llama_dedup/PROTOCOL.md v1.0",
        "issue": "https://github.com/run-llama/llama_index/issues/21033",
        "versions": {"bug": BUG, "fix": FIX},
        "criteria": crit,
        "environment_differences_bug_vs_fix": {k: list(v) for k, v in env.items()},
        "overall": "PASS" if all(crit[k]["pass"] for k in gated) else "FAIL",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: evaluate and write results.json (never overwrites)."""
    parser = argparse.ArgumentParser(description="Evaluate V7 Experiment 2")
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args(argv)
    out = args.results / "results.json"
    if out.exists():
        raise FileExistsError(f"{out} exists; results are never overwritten")
    results = evaluate(args.results)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for name, c in results["criteria"].items():
        sys.stdout.write(f"{name}: {json.dumps(c)}\n")
    sys.stdout.write(f"OVERALL: {results['overall']}\n")
    return 0 if results["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
