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

__all__ = ["split_indices", "split_half_null", "is_divergent", "above_chance",
           "relabel_within_layers", "relabelling_null", "sorted_within_layer_dm"]


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


# ── Head-relabelling diagnostics (added after the 2026-09-29 seed audit) ─────────
# Heads within a layer are interchangeable: permuting them leaves the model's function
# unchanged. The primary D_M compares heads by index, so it is only meaningful when two
# models share head correspondence (e.g. checkpoints of one training run). These
# diagnostics quantify that limitation; they do not replace the primary D_M.

def _layer_index(heads: Sequence[Hashable]) -> np.ndarray:
    return np.array([h[0] for h in heads])  # heads are (layer, head) tuples


def relabel_within_layers(v: np.ndarray, heads: Sequence[Hashable],
                          rng: np.random.Generator) -> np.ndarray:
    """Randomly permute attribution values among the heads of each layer."""
    v = np.asarray(v, dtype=float)
    out = v.copy()
    layers = _layer_index(heads)
    for layer in np.unique(layers):
        idx = np.flatnonzero(layers == layer)
        out[idx] = v[rng.permutation(idx)]
    return out


def relabelling_null(v: np.ndarray, heads: Sequence[Hashable], n: int,
                     seed: int) -> Dict[str, Any]:
    """Distribution of primary D_M between one model and relabelled copies of itself."""
    rng = np.random.default_rng(seed)
    keys = list(heads)
    base = dict(zip(keys, np.asarray(v, dtype=float)))
    vals = [operational_mechanistic_distance(
        base, dict(zip(keys, relabel_within_layers(v, heads, rng))))["value"]
        for _ in range(n)]
    vals = [x for x in vals if x == x]
    return {"n": n, "seed": seed, "p05": float(np.percentile(vals, 5)),
            "p50": float(np.percentile(vals, 50)), "p95": float(np.percentile(vals, 95))}


def sorted_within_layer_dm(a: np.ndarray, b: np.ndarray,
                           heads: Sequence[Hashable]) -> float:
    """1 - Spearman after sorting each layer's head values (label-free diagnostic).

    Sorting aligns heads optimistically by attribution value, so this approximately
    lower-bounds any within-layer-label-invariant version of D_M. It discards which
    head does what; it is a diagnostic, not a replacement definition.
    """
    layers = _layer_index(heads)

    def canon(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.concatenate([np.sort(x[layers == layer]) for layer in np.unique(layers)])

    keys = list(range(len(heads)))
    return float(operational_mechanistic_distance(
        dict(zip(keys, canon(a))), dict(zip(keys, canon(b))))["value"])


def is_divergent(d_m: float, null_p95s: Sequence[float]) -> Any:
    """Draft (positional, superseded for cross-run pairs) rule: attribution-profile-divergent
    iff D_M exceeds the larger of the two models' Control-B p95. Record key: ``divergent``.

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
