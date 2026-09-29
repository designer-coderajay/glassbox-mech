"""V6 distance definitions: D_P, D_M (operational) and D_B.

Pure NumPy/SciPy, no torch. Every function here implements a definition fixed in
``experiments/v6/PREREGISTRATION.md`` §3; the version string below is recorded in every
experiment record so a reproducer knows which definitions produced a number.

Naming is deliberate: :func:`operational_mechanistic_distance` is *one operational
measurement* of mechanistic divergence under a defined attribution procedure. It is not
"the mechanistic distance" between two models.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Hashable, Mapping, Sequence

import numpy as np
from scipy import stats

# 1.1.0 (2026-09-28): TOST with SD = 0 no longer reports p = 0; binary data use an
# exact Clopper-Pearson bound on the discordance rate. Runs recorded under 1.0.0 keep
# their stored values (see PREREGISTRATION.md amendment log).
METRICS_VERSION = "v6.distances/1.1.0"

__all__ = [
    "METRICS_VERSION",
    "paired_tost",
    "exact_paired_equivalence",
    "performance_distance",
    "operational_mechanistic_distance",
    "topk_jaccard_distance",
    "js_divergence",
    "behavioral_distance",
]


def paired_tost(
    diffs: Sequence[float], margin: float, alpha: float = 0.05
) -> Dict[str, Any]:
    """Two one-sided t-tests for equivalence of a paired mean difference.

    H0: |mean(diffs)| >= margin.  H1: -margin < mean(diffs) < margin.
    Equivalence is declared when both one-sided p-values are below ``alpha``.

    Args:
        diffs: Per-item paired differences (e.g. correct_A - correct_B in {-1, 0, 1}).
        margin: Equivalence margin on the same scale as ``diffs`` (> 0).
        alpha: One-sided significance level of each test.

    If every difference is identical (SD = 0) the t statistic is undefined: p-values are
    None and the decision uses an exact bound (binary data) or the known constant gap.

    Returns:
        Dict with ``method``, ``mean_diff``, ``se``, ``df``, ``p_lower``, ``p_upper``,
        ``p_tost`` (None when undefined), ``discordance_upper_bound``, ``equivalent``.
    """
    if margin <= 0:
        raise ValueError("margin must be > 0")
    d = np.asarray(list(diffs), dtype=float)
    n = int(d.size)
    if n < 2:
        raise ValueError("paired_tost needs at least 2 paired observations")
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    out: Dict[str, Any] = {"mean_diff": mean, "df": n - 1, "n": n, "margin": margin,
                           "alpha": alpha, "discordance_upper_bound": None}
    if not np.all(d == d[0]):  # exact test; float SD of equal values can be ~1e-18
        se = sd / math.sqrt(n)
        p_lower = float(stats.t.sf((mean + margin) / se, n - 1))  # H0: mean <= -margin
        p_upper = float(stats.t.cdf((mean - margin) / se, n - 1))  # H0: mean >= +margin
        out.update(method="paired_t_tost", se=se, p_lower=p_lower, p_upper=p_upper,
                   p_tost=max(p_lower, p_upper),
                   equivalent=bool(max(p_lower, p_upper) < alpha))
        return out
    # SD = 0: the t statistic is undefined, so no p-value is reported.
    out.update(se=0.0, p_lower=None, p_upper=None, p_tost=None)
    if set(np.unique(d)) <= {-1.0, 0.0, 1.0}:
        # Paired binary correctness: |accuracy gap| <= discordance rate. Reject
        # H0 (|gap| >= margin) iff the exact one-sided Clopper-Pearson upper bound
        # on the discordance rate is below the margin. Valid, conservative.
        k = int(np.count_nonzero(d))
        bound = 1.0 if k == n else float(stats.beta.ppf(1 - alpha, k + 1, n - k))
        out.update(method="exact_discordance_bound", discordance_upper_bound=bound,
                   equivalent=bool(bound < margin))
    else:
        # Constant continuous differences: the gap is known exactly.
        out.update(method="degenerate_constant", equivalent=bool(abs(mean) < margin))
    return out


def exact_paired_equivalence(
    correct_a: Sequence[bool], correct_b: Sequence[bool], margin: float,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Exact equivalence test for paired binary correctness (candidate for Amendment 2).

    |acc_A - acc_B| <= discordance rate, so H0 (|gap| >= margin) is rejected iff the
    one-sided (1 - alpha) Clopper-Pearson upper bound on the discordance rate is below
    the margin. Valid for any n, monotone in the number of discordant items, and
    conservative (it ignores sign cancellation). Not yet used by ``performance_distance``.
    """
    a = np.asarray(list(correct_a), dtype=bool)
    b = np.asarray(list(correct_b), dtype=bool)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("both models must be evaluated on the same, non-empty items")
    n, k = int(a.size), int((a != b).sum())
    bound = 1.0 if k == n else float(stats.beta.ppf(1 - alpha, k + 1, n - k))
    return {"n": n, "n_discordant": k, "discordance_upper_bound": bound,
            "margin": margin, "alpha": alpha, "equivalent": bool(bound < margin),
            "method": "exact_discordance_bound"}


def performance_distance(
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
    ld_a: Sequence[float],
    ld_b: Sequence[float],
    margin: float,
) -> Dict[str, Any]:
    """D_P: accuracy gap and mean logit-difference gap, with a paired TOST on accuracy.

    Both models must be evaluated on the same items in the same order. The TOST level is
    fixed at alpha = 0.05 per one-sided test (PREREGISTRATION.md §3.1).

    Returns:
        ``accuracy_a``, ``accuracy_b``, ``accuracy_diff`` (A - B), ``mean_ld_a``,
        ``mean_ld_b``, ``mean_ld_diff`` and ``tost`` (see :func:`paired_tost`).
        ``matched`` is True only if the TOST declares equivalence.
    """
    ca = np.asarray(list(correct_a), dtype=float)
    cb = np.asarray(list(correct_b), dtype=float)
    la = np.asarray(list(ld_a), dtype=float)
    lb = np.asarray(list(ld_b), dtype=float)
    if not (ca.shape == cb.shape == la.shape == lb.shape):
        raise ValueError("both models must be evaluated on the same items")
    tost = paired_tost(ca - cb, margin=margin, alpha=0.05)
    return {
        "accuracy_a": float(ca.mean()),
        "accuracy_b": float(cb.mean()),
        "accuracy_diff": float(ca.mean() - cb.mean()),
        "mean_ld_a": float(la.mean()),
        "mean_ld_b": float(lb.mean()),
        "mean_ld_diff": float(la.mean() - lb.mean()),
        "n_items": int(ca.size),
        "n_a_only": int(((ca == 1) & (cb == 0)).sum()),
        "n_b_only": int(((ca == 0) & (cb == 1)).sum()),
        "tost": tost,
        "matched": tost["equivalent"],
    }


def _aligned(
    attr_a: Mapping[Hashable, float], attr_b: Mapping[Hashable, float]
) -> tuple:
    if set(attr_a) != set(attr_b):
        raise ValueError(
            "attribution vectors cover different units; D_M requires the same "
            "architecture (identical head sets)"
        )
    keys = sorted(attr_a, key=str)
    a = np.array([float(attr_a[k]) for k in keys])
    b = np.array([float(attr_b[k]) for k in keys])
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("attribution vectors contain non-finite values")
    return keys, a, b


def operational_mechanistic_distance(
    attr_a: Mapping[Hashable, float], attr_b: Mapping[Hashable, float]
) -> Dict[str, Any]:
    """Primary D_M: ``1 - Spearman(attr_a, attr_b)`` over identical head sets.

    Range [0, 2]: 0 = identical head ranking, 1 = unrelated, 2 = reversed.
    Returns ``value`` = NaN with a ``reason`` when the correlation is undefined
    (a constant vector); NaN is never silently turned into 0 or 1.
    """
    keys, a, b = _aligned(attr_a, attr_b)
    if np.all(a == a[0]) or np.all(b == b[0]):
        return {"value": float("nan"), "rho": float("nan"), "n_units": len(keys),
                "reason": "constant attribution vector; Spearman undefined"}
    rho = float(stats.spearmanr(a, b).correlation)
    return {"value": 1.0 - rho, "rho": rho, "n_units": len(keys), "reason": None}


def topk_jaccard_distance(
    attr_a: Mapping[Hashable, float], attr_b: Mapping[Hashable, float], k: int
) -> Dict[str, Any]:
    """Secondary D_M: ``1 - Jaccard`` of the top-k heads ranked by |attribution|.

    Ties are broken by the head key's string form, so the result is deterministic.
    """
    keys, a, b = _aligned(attr_a, attr_b)
    if not 1 <= k <= len(keys):
        raise ValueError(f"k must be in [1, {len(keys)}]")

    def top(v: np.ndarray) -> set:
        order = sorted(range(len(keys)), key=lambda i: (-abs(v[i]), str(keys[i])))
        return {keys[i] for i in order[:k]}

    ta, tb = top(a), top(b)
    jac = len(ta & tb) / len(ta | tb)
    return {"value": 1.0 - jac, "jaccard": jac, "k": k,
            "top_a": sorted(map(str, ta)), "top_b": sorted(map(str, tb))}


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence in bits (log base 2), bounded in [0, 1].

    Inputs are renormalised to sum to 1; negative entries are rejected.
    """
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    if p_arr.shape != q_arr.shape:
        raise ValueError("distributions must have the same support")
    if np.any(p_arr < 0) or np.any(q_arr < 0):
        raise ValueError("probabilities must be non-negative")
    p_arr = p_arr / p_arr.sum()
    q_arr = q_arr / q_arr.sum()
    m = 0.5 * (p_arr + q_arr)

    def kl(x: np.ndarray) -> float:
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / m[mask])))

    return float(min(1.0, max(0.0, 0.5 * kl(p_arr) + 0.5 * kl(q_arr))))


def behavioral_distance(
    dists_a: np.ndarray, dists_b: np.ndarray
) -> Dict[str, Any]:
    """D_B: mean Jensen-Shannon divergence over a fixed probe set.

    Args:
        dists_a, dists_b: ``[n_probes, vocab]`` next-token distributions of models A
            and B on the *same* probes in the same order (outputs only).

    Returns:
        ``value`` (mean JSD, bits), ``per_probe`` list and ``n_probes``.
    """
    a = np.asarray(dists_a, dtype=np.float64)
    b = np.asarray(dists_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("expected two [n_probes, vocab] arrays of the same shape")
    per = [js_divergence(a[i], b[i]) for i in range(a.shape[0])]
    return {"value": float(np.mean(per)), "per_probe": per, "n_probes": len(per)}
