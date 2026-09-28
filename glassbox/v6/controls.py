"""V6 Control B (resampling null for D_M) and the model-inclusion check.

Control B: the same model measured on two disjoint halves of the item set. The spread of
D_M between the two half-means is the measurement-noise floor. It is computed from
per-item attribution vectors, so it costs no extra forward passes.

Known limitation (PREREGISTRATION.md §5): each half uses n/2 items while the pair D_M uses
n items, so this null overstates the noise at n. That makes the "divergent" rule
conservative (fewer pairs flagged), never anti-conservative.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Hashable, Iterator, List, Sequence, Tuple

import numpy as np
from scipy import stats

from glassbox.v6.distances import operational_mechanistic_distance

__all__ = ["split_indices", "split_half_null", "is_divergent", "above_chance"]


def split_indices(n_items: int, n_splits: int,
                  seed: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield ``n_splits`` pairs of disjoint, equal-size index sets (floor(n/2) each)."""
    rng = np.random.default_rng(seed)
    half = n_items // 2
    for _ in range(n_splits):
        perm = rng.permutation(n_items)
        yield perm[:half], perm[half:2 * half]


def split_half_null(per_item: np.ndarray, heads: Sequence[Hashable], n_splits: int,
                    seed: int) -> Dict[str, Any]:
    """Control B: D_M between the attribution means of disjoint item halves.

    Args:
        per_item: ``[n_items, n_heads]`` per-item attribution matrix of ONE model.
        heads: Head keys, in column order.
        n_splits: Number of random half-splits.
        seed: RNG seed (recorded).

    Returns:
        ``values`` (finite D_M values), ``n_undefined``, ``p50``, ``p95``, ``n_splits``,
        ``half_size``, ``seed``.
    """
    m = np.asarray(per_item, dtype=float)
    if m.ndim != 2 or m.shape[1] != len(heads):
        raise ValueError("per_item must be [n_items, len(heads)]")
    if m.shape[0] < 4:
        raise ValueError("Control B needs at least 4 items")
    values: List[float] = []
    undefined = 0
    for s1, s2 in split_indices(m.shape[0], n_splits, seed):
        a = dict(zip(heads, m[s1].mean(axis=0)))
        b = dict(zip(heads, m[s2].mean(axis=0)))
        v = operational_mechanistic_distance(a, b)["value"]
        if math.isnan(v):
            undefined += 1
        else:
            values.append(float(v))
    if not values:
        raise ValueError("Control B undefined: every split gave a constant vector")
    return {
        "values": values, "n_undefined": undefined,
        "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
        "n_splits": n_splits, "half_size": m.shape[0] // 2, "seed": seed,
    }


def is_divergent(d_m: float, null_p95s: Sequence[float]) -> Any:
    """Draft rule: divergent iff D_M exceeds the larger of the two models' Control-B p95.

    Returns None when D_M is undefined (NaN).
    """
    if d_m != d_m:
        return None
    return bool(d_m > max(null_p95s))


def above_chance(correct: Sequence[bool], chance: float = 0.5,
                 alpha: float = 0.05) -> Dict[str, Any]:
    """One-sided binomial test of task accuracy against chance (inclusion check)."""
    k, n = int(sum(bool(c) for c in correct)), len(correct)
    p = float(stats.binomtest(k, n, chance, alternative="greater").pvalue)
    return {"k": k, "n": n, "accuracy": k / n, "chance": chance, "p_value": p,
            "above_chance": bool(p < alpha)}
