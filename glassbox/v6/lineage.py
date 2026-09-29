"""Run-level inference for the revised H1 (attribution-profile divergence across runs).

Estimand (revised H1, Amendment 3 draft):

    Delta = E_{r != r'} D*(r@143k, r'@143k)  -  E_r mean_{s<t in S} D*(r@s, r@t)

where D* is the attribution-profile distance modulo within-layer head permutation and S is
the pre-registered checkpoint set. The within term measures **within-training-lineage
variation**; it is not a "same mechanism" baseline.

Dependence. Training runs are the inferential unit. Every pair distance involving run r
depends on r, and all distances share the same prompts. Uncertainty is therefore
estimated by resampling **runs** (pairs of a run with itself are dropped) and, jointly,
**prompts** (the caller supplies pair distances recomputed under each prompt resample).

Inputs:
    within : [R, P]   within-run distances for R runs and P pre-registered checkpoint pairs
    cross  : [R, R]   symmetric cross-run distances at the final checkpoint (diag ignored)
Bootstrap inputs carry a leading axis B (one prompt resample per row).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy import stats

__all__ = ["delta", "delta_for_resample", "jackknife", "two_way_bootstrap"]


def _check(within: np.ndarray, cross: np.ndarray) -> None:
    r = within.shape[0]
    if cross.shape != (r, r):
        raise ValueError("cross must be [R, R] with R = within.shape[0]")
    off = ~np.eye(r, dtype=bool)
    if not np.allclose(cross[off], cross.T[off]):
        raise ValueError("cross-run distance matrix must be symmetric")


def delta(within: np.ndarray, cross: np.ndarray) -> float:
    """Point estimate: mean over distinct run pairs minus mean within-run distance."""
    within, cross = np.asarray(within, float), np.asarray(cross, float)
    _check(within, cross)
    iu = np.triu_indices(within.shape[0], k=1)
    return float(cross[iu].mean() - within.mean())


def delta_for_resample(within: np.ndarray, cross: np.ndarray, runs: np.ndarray) -> float:
    """Delta for a run resample (with multiplicity); self-pairs are dropped.

    Returns NaN if fewer than two distinct runs were drawn.
    """
    runs = np.asarray(runs)
    if np.unique(runs).size < 2:
        return float("nan")
    w = within[runs].mean()
    i, j = np.triu_indices(runs.size, k=1)
    keep = runs[i] != runs[j]
    return float(cross[runs[i][keep], runs[j][keep]].mean() - w)


def jackknife(within: np.ndarray, cross: np.ndarray) -> Dict[str, float]:
    """Leave-one-run-out jackknife SE and one-sided t test (fallback procedure)."""
    within, cross = np.asarray(within, float), np.asarray(cross, float)
    _check(within, cross)
    r = within.shape[0]
    loo = np.array([delta(np.delete(within, k, 0),
                          np.delete(np.delete(cross, k, 0), k, 1)) for k in range(r)])
    se = float(np.sqrt((r - 1) / r * np.sum((loo - loo.mean()) ** 2)))
    d = delta(within, cross)
    t = d / se if se > 0 else float("inf") if d > 0 else float("-inf")
    return {"delta": d, "se": se, "t": float(t), "df": r - 1,
            "p_one_sided": float(stats.t.sf(t, r - 1)), "reject_h0": bool(t > stats.t.ppf(0.95, r - 1))}


def two_way_bootstrap(within_b: np.ndarray, cross_b: np.ndarray, point_within: np.ndarray,
                      point_cross: np.ndarray, seed: int = 0,
                      alpha: float = 0.05, runs_draws: Optional[np.ndarray] = None
                      ) -> Dict[str, Any]:
    """Primary procedure: runs and prompts resampled together; one-sided lower bound.

    within_b / cross_b hold pair distances recomputed under B prompt resamples; replicate b
    uses prompt resample b and an independent run resample. Reject H0 (Delta <= 0) iff the
    alpha-quantile of the bootstrap distribution is > 0.
    """
    within_b, cross_b = np.asarray(within_b, float), np.asarray(cross_b, float)
    n_boot, r = within_b.shape[0], within_b.shape[1]
    rng = np.random.default_rng(seed)
    draws = runs_draws if runs_draws is not None else rng.integers(0, r, size=(n_boot, r))
    reps = np.array([delta_for_resample(within_b[b], cross_b[b], draws[b])
                     for b in range(n_boot)])
    valid = reps[~np.isnan(reps)]
    lower = float(np.quantile(valid, alpha))
    return {"delta": delta(point_within, point_cross), "lower_95": lower,
            "reject_h0": bool(lower > 0), "n_boot": n_boot, "n_valid": int(valid.size),
            "seed": seed}
