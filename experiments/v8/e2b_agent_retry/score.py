"""Gated criteria G1–G10 and descriptive metrics for E2b (PROTOCOL.md §8). The
criteria and bars are E2's; their input is the two-sided labels of
``reference.py``, after key-suspect questions are removed by the runner. Pairs
labelled ``mixed`` or ``small`` are not scored.

Inputs, per test question:
- ``labels[q][factor]`` ∈ {harmful, helpful, null, small} from the reference run;
- ``regressed[q]``: the reference run's regressed flag;
- ``methods[q][m]`` = factors blamed by method m ∈ {GB, B0, B1, B2} (GB = harmful
  list), plus ``GB_fault[q][factor]`` (GB's fault label or ``None``);
- ``recovery[q]`` with ``arms`` (``gb_fix``, ``wrong_fix``, ``b2_fix``,
  ``undirected_fix``) and ``gb_minus_undirected_ci``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

METHODS = ("GB", "B0", "B1", "B2")
MIN_HARMFUL = 6
MIN_REGRESSED = 6
MIN_TESTABLE = 4
MIN_LABELLED = 6
MAX_KEY_SUSPECT = 4
VISIBLE = "persona"
FAULT_CLASS = {
    "invoice_tool_desc": "wrong_tool",
    "eligibility_tool": "skipped_tool",
    "quote_example": "wrong_order",
    "max_steps": "premature_finish",
    "answer_policy": "no_action",
}


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(mid - half, 4), round(mid + half, 4))


def _pairs(labels: Dict[str, Dict[str, str]], *kinds: str) -> List[Tuple[str, str]]:
    return [
        (q, f) for q, row in labels.items() for f, lab in row.items() if lab in kinds
    ]


def _frac(hits: int, n: int) -> float:
    return round(hits / n, 4) if n else 0.0


def f1(
    labels: Dict[str, Dict[str, str]],
    blamed: Dict[str, Sequence[str]],
    positive: Sequence[str] = ("harmful",),
) -> Dict[str, Any]:
    """F1 over ``positive`` pairs vs null pairs; other pairs are skipped."""
    pos = _pairs(labels, *positive)
    tp = sum(f in blamed[q] for q, f in pos)
    fn = len(pos) - tp
    fp = sum(f in blamed[q] for q, f in _pairs(labels, "null"))
    score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "f1": round(score, 4)}


def _rate_gate(hits: int, n: int, bar: float, at_most: bool, min_n: int = 1):
    out: Dict[str, Any] = {"n": n, "hits": hits, "value": _frac(hits, n)}
    out["testable"] = n >= min_n
    ok = out["value"] <= bar if at_most else out["value"] >= bar
    out["pass"] = ok if out["testable"] else None
    return out


def _verified(arm: Dict[str, Any]) -> bool:
    return bool(arm.get("run") and arm.get("verified"))


def criteria(
    labels: Dict[str, Dict[str, str]],
    regressed: Dict[str, bool],
    methods: Dict[str, Dict[str, Sequence[str]]],
    recovery: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """G1–G10 and the overall verdict (the generator gate is applied by the runner)."""
    gb = {q: list(methods[q]["GB"]) for q in labels}
    harm = _pairs(labels, "harmful")
    g1 = _rate_gate(sum(f in gb[q] for q, f in harm), len(harm), 0.75, False)
    g1["pass"] = bool(g1["pass"])
    n_log = sum("log_level" in gb[q] for q in labels)
    g2 = {"value": n_log, "n_questions": len(labels), "pass": n_log <= 2}
    tnull = [(q, f) for q, f in _pairs(labels, "null") if f == VISIBLE]
    g3 = _rate_gate(
        sum(VISIBLE in gb[q] for q, _ in tnull), len(tnull), 0.10, True, MIN_TESTABLE
    )
    helpful = _pairs(labels, "helpful")
    g4 = _rate_gate(
        sum(f in gb[q] for q, f in helpful), len(helpful), 0.10, True, MIN_TESTABLE
    )
    reg_q = [q for q in labels if regressed.get(q)]
    g5 = _rate_gate(
        sum(_verified(recovery[q]["arms"]["gb_fix"]) for q in reg_q),
        len(reg_q),
        0.80,
        False,
    )
    g5["verified"] = g5["hits"]
    g5["pass"] = bool(g5["pass"])
    g6_q = [q for q in reg_q if labels[q].get(VISIBLE) != "harmful"]
    g6 = _rate_gate(
        sum(not _verified(recovery[q]["arms"]["wrong_fix"]) for q in g6_q),
        len(g6_q),
        0.80,
        False,
    )
    g6["rejected"] = g6["hits"]
    g6["pass"] = bool(g6["pass"])  # always counted (PROTOCOL §7); none to test = fail
    scores = {m: f1(labels, {q: methods[q][m] for q in labels}) for m in METHODS}
    best_other = max(scores[m]["f1"] for m in METHODS if m != "GB")
    g7 = {
        "f1": {m: s["f1"] for m, s in scores.items()},
        "counts": scores,
        "margin": round(scores["GB"]["f1"] - best_other, 4),
    }
    g7["pass"] = g7["margin"] >= 0.10
    g8 = _g8(labels, recovery)
    g9 = _g9(labels, methods)
    null = _pairs(labels, "null")
    g10 = _rate_gate(sum(f in gb[q] for q, f in null), len(null), 0.10, True)
    gates = {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6, "G7": g7,
             "G8": g8, "G9": g9, "G10": g10}
    if len(harm) < MIN_HARMFUL or len(reg_q) < MIN_REGRESSED:
        overall = "INCONCLUSIVE"
    else:
        tested = [g["pass"] for g in gates.values() if g["pass"] is not None]
        overall = "PASS" if all(bool(p) for p in tested) else "FAIL"
    return {
        **gates,
        "n_harmful_large": len(harm),
        "n_regressed": len(reg_q),
        "overall": overall,
    }


def _g8(labels: Dict[str, Dict[str, str]], recovery: Dict[str, Dict[str, Any]]):
    """Directed fix keeps the gain (PROTOCOL §8; Addendum 1 item 6)."""
    eligible = [
        q
        for q in labels
        if "helpful" in labels[q].values()
        and recovery[q].get("gb_minus_undirected_ci") is not None
    ]
    hits = sum(recovery[q]["gb_minus_undirected_ci"][0] >= 0.20 for q in eligible)
    out = _rate_gate(hits, len(eligible), 0.75, False, MIN_TESTABLE)
    out["questions"] = eligible
    return out


def _g9(
    labels: Dict[str, Dict[str, str]], methods: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Fault type named, over harmful-large pairs of the five fault factors."""
    pairs = [(q, f) for q, f in _pairs(labels, "harmful") if f in FAULT_CLASS]
    named = [(q, f, methods[q]["GB_fault"].get(f)) for q, f in pairs]
    labelled = [(q, f, lab) for q, f, lab in named if lab is not None]
    hits = sum(lab == FAULT_CLASS[f] for _, f, lab in labelled)
    out = _rate_gate(hits, len(labelled), 0.75, False, MIN_LABELLED)
    out["unlabelled"] = len(named) - len(labelled)
    out["confusion"] = [
        {"question": q, "factor": f, "label": lab} for q, f, lab in named
    ]
    return out


def coverage(labels: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Per fault class: harmful-large test pairs of its factor; produced if ≥ 2."""
    out = {}
    for f, cls in FAULT_CLASS.items():
        n = sum(row.get(f) == "harmful" for row in labels.values())
        out[cls] = {"factor": f, "harmful_large_pairs": n, "produced": n >= 2}
    return out


def descriptive(
    labels: Dict[str, Dict[str, str]],
    regressed: Dict[str, bool],
    methods: Dict[str, Dict[str, Sequence[str]]],
    recovery: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Misses/false alarms, fix rates for GB and B2, E1's two-sided rule, the
    undirected list scored E1-style, and small pairs left in place by a failed fix."""
    harm, null = _pairs(labels, "harmful"), _pairs(labels, "null")
    per_method: Dict[str, Any] = {}
    for m in METHODS:
        misses = sum(f not in methods[q][m] for q, f in harm)
        alarms = sum(f in methods[q][m] for q, f in null)
        per_method[m] = {
            "misses": misses,
            "n_harmful": len(harm),
            "miss_ci": wilson(misses, len(harm)),
            "false_alarms": alarms,
            "n_null": len(null),
            "false_alarm_ci": wilson(alarms, len(null)),
        }
    reg_q = [q for q in labels if regressed.get(q)]
    fix = {
        m: {
            "verified": sum(_verified(recovery[q]["arms"][arm]) for q in reg_q),
            "n": len(reg_q),
        }
        for m, arm in (("GB", "gb_fix"), ("B2", "b2_fix"),
                       ("undirected", "undirected_fix"))
    }
    two_sided = {
        arm: sum(
            bool(recovery[q]["arms"][arm].get("verified_two_sided")) for q in reg_q
        )
        for arm in ("gb_fix", "wrong_fix", "b2_fix", "undirected_fix")
    }
    per_factor = {}
    factors = sorted({f for row in labels.values() for f in row})
    for f in factors:
        per_factor[f] = {
            lab: sum(row.get(f) == lab for row in labels.values())
            for lab in ("harmful", "helpful", "null", "small", "mixed")
        }
        per_factor[f]["GB_blamed"] = sum(f in methods[q]["GB"] for q in labels)
        per_factor[f]["GB_blamed_on_null"] = sum(
            f in methods[q]["GB"] for q in labels if labels[q].get(f) == "null"
        )
    small_left = [
        q
        for q in reg_q
        if not _verified(recovery[q]["arms"]["gb_fix"])
        and any(
            lab == "small" and f not in methods[q]["GB"] for f, lab in labels[q].items()
        )
    ]
    undirected = f1(
        labels,
        {q: methods[q].get("GB_attributed", methods[q]["GB"]) for q in labels},
        positive=("harmful", "helpful"),
    )
    return {
        "methods": per_method,
        "fix_verified": fix,
        "two_sided_verified": {**two_sided, "n": len(reg_q)},
        "n_small": len(_pairs(labels, "small")),
        "n_mixed": len(_pairs(labels, "mixed")),
        "n_helpful": len(_pairs(labels, "helpful")),
        "failed_fix_with_small_factor_left": small_left,
        "undirected_attributed": undirected,
        "per_factor": per_factor,
        "coverage": coverage(labels),
    }
