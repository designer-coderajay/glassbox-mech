"""Gated criteria G1–G6 and descriptive metrics for E1 (PROTOCOL.md §7).

Inputs, per test question:
- ``labels[q][factor]`` ∈ {large, null, small} from the reference run;
- ``methods[q][m]`` = factors blamed by method m ∈ {GB, B0, B1, B2};
- ``recovery[q]`` = {"fixed": bool | None, "wrong_fix": bool}.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

METHODS = ("GB", "B0", "B1", "B2")
MIN_LARGE = 6


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(mid - half, 4), round(mid + half, 4))


def _pairs(labels: Dict[str, Dict[str, str]], kind: str) -> List[Tuple[str, str]]:
    return [
        (q, f) for q, row in labels.items() for f, lab in row.items() if lab == kind
    ]


def _frac(hits: int, n: int) -> float:
    return round(hits / n, 4) if n else 0.0


def f1(
    labels: Dict[str, Dict[str, str]], blamed: Dict[str, Sequence[str]]
) -> Dict[str, Any]:
    """F1 over large (positive) and null (negative) pairs; small pairs are skipped."""
    tp = sum(f in blamed[q] for q, f in _pairs(labels, "large"))
    fn = len(_pairs(labels, "large")) - tp
    fp = sum(f in blamed[q] for q, f in _pairs(labels, "null"))
    score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "f1": round(score, 4)}


def _g4_g5(
    labels: Dict[str, Dict[str, str]],
    gb: Dict[str, Sequence[str]],
    recovery: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    g4_q = [
        q
        for q, row in labels.items()
        if gb[q] and any(v == "large" for v in row.values())
    ]
    g4_hits = sum(bool(recovery[q].get("fixed")) for q in g4_q)
    g5_q = [
        q
        for q, row in labels.items()
        if row.get("template") != "large"
        and any(v == "large" for f, v in row.items() if f != "template")
    ]
    g5_hits = sum(not recovery[q].get("wrong_fix") for q in g5_q)
    g4 = {
        "n": len(g4_q),
        "verified": g4_hits,
        "value": _frac(g4_hits, len(g4_q)),
        "testable": bool(g4_q),
    }
    g4["pass"] = g4["testable"] and g4["value"] >= 0.80
    g5 = {
        "n": len(g5_q),
        "rejected": g5_hits,
        "value": _frac(g5_hits, len(g5_q)),
        "testable": bool(g5_q),
    }
    g5["pass"] = g5["testable"] and g5["value"] >= 0.80
    return g4, g5


def criteria(
    labels: Dict[str, Dict[str, str]],
    methods: Dict[str, Dict[str, Sequence[str]]],
    recovery: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """G1–G6 and the overall verdict (generator gate is applied by the runner)."""
    gb = {q: list(methods[q]["GB"]) for q in labels}
    large = _pairs(labels, "large")
    g1_hits = sum(f in gb[q] for q, f in large)
    g1 = {
        "n_large": len(large),
        "attributed": g1_hits,
        "value": _frac(g1_hits, len(large)),
    }
    g1["pass"] = g1["value"] >= 0.75
    g2_n = sum("log_level" in gb[q] for q in labels)
    g2 = {"value": g2_n, "n_questions": len(labels), "pass": g2_n <= 2}
    tnull = [(q, f) for q, f in _pairs(labels, "null") if f == "template"]
    g3_hits = sum("template" in gb[q] for q, _ in tnull)
    g3 = {
        "n": len(tnull),
        "attributed": g3_hits,
        "value": _frac(g3_hits, len(tnull)),
        "testable": len(tnull) >= 4,
    }
    g3["pass"] = g3["value"] <= 0.10 if g3["testable"] else None
    g4, g5 = _g4_g5(labels, gb, recovery)
    scores = {m: f1(labels, {q: methods[q][m] for q in labels}) for m in METHODS}
    best_other = max(scores[m]["f1"] for m in METHODS if m != "GB")
    g6 = {
        "f1": {m: s["f1"] for m, s in scores.items()},
        "counts": scores,
        "margin": round(scores["GB"]["f1"] - best_other, 4),
    }
    g6["pass"] = g6["margin"] >= 0.10
    gates = {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6}
    if len(large) < MIN_LARGE:
        overall = "INCONCLUSIVE"
    else:
        tested = [g["pass"] for k, g in gates.items() if k != "G3" or g3["testable"]]
        overall = "PASS" if all(bool(p) for p in tested) else "FAIL"
    return {**gates, "overall": overall}


def descriptive(
    labels: Dict[str, Dict[str, str]],
    methods: Dict[str, Dict[str, Sequence[str]]],
) -> Dict[str, Any]:
    """Misses and false alarms per method, with Wilson 95% intervals."""
    large, null = _pairs(labels, "large"), _pairs(labels, "null")
    out: Dict[str, Any] = {}
    for m in METHODS:
        misses = sum(f not in methods[q][m] for q, f in large)
        alarms = sum(f in methods[q][m] for q, f in null)
        out[m] = {
            "misses": misses,
            "n_large": len(large),
            "miss_ci": wilson(misses, len(large)),
            "false_alarms": alarms,
            "n_null": len(null),
            "false_alarm_ci": wilson(alarms, len(null)),
        }
    return out
