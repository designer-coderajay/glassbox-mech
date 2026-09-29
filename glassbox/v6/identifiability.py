"""Head identifiability for V6 attribution-profile distances.

Setting. Model m has a per-item, per-head attribution tensor ``X`` with shape
``[N items, L layers, H heads]`` and mean vector ``a = mean_i X[i]`` (shape ``[L, H]``).

Symmetry. Let ``G = S_H^L``: an independent permutation ``pi_l`` of the H heads in every
layer. Permuting head ``h``'s parameter slices (W_Q, W_K, W_V, b_Q, b_K, b_V along the head
axis and W_O along its head axis; b_O is shared) by ``pi_l`` leaves the network function
unchanged. Per-head attribution is then *equivariant*: ``X(pi . theta) = pi . X(theta)``,
where ``(pi . X)[i, l, h] = X[i, l, pi_l(h)]``. ``permute_heads_`` implements the weight-level
action; the tests verify output invariance and attribution equivariance on a real model.

Identifiability requirement. A distance between *models* must not depend on arbitrary head
labels: ``d(A, B) = d(A, pi . B)`` for all ``pi`` in ``G``.

Candidate distances (defined here before being applied to any seed data):

``positional_dm``  ``1 - rho_s(a_A, a_B)`` over the L*H positions (the pre-registered D_M).
    Not G-invariant.

``scalar_quotient_dm``  ``min_{pi in G} [1 - rho_s(a_A, pi . a_B)]``, attained (by the
    rearrangement inequality, up to rank ties) by sorting each layer's H values: the 1-D
    optimal-transport / multiset comparison of scalar mean attributions. G-invariant, but it
    only compares per-layer *distributions of attribution magnitudes*; two different circuits
    with the same magnitude histogram are indistinguishable. Not a profile distance either;
    only a magnitude-distribution comparison.

``profile_orbit_distance``  Represent head (l, h) by its per-item profile ``X[:, l, h]``
    (how its causal effect varies across inputs). After scaling each model's tensor to unit
    Frobenius norm (scale-free), ``D_orbit(A, B) = min_{pi in G} ||X~_A - pi . X~_B||_F^2``.
    The minimum decomposes into one assignment problem per layer (exact via Hungarian).
    ``G`` is finite and acts by isometries, so ``min_pi ||A - pi.B||`` is a metric on
    orbits (the quotient space); range [0, 4]. Equals 0 iff B is a relabelled copy of A (up
    to scale). The minimisation is data-driven, so it is biased low under noise; it must be
    judged against a same-model split-half floor.

``crossfit_aligned_dm``  Positional D_M after aligning heads with a *held-out* fit split:
    for each of K random splits of the items, ``pi_hat`` minimises the profile cost on the
    fit half (per-layer Hungarian on ``1 - Pearson`` of profiles), then ``1 - rho_s`` is
    evaluated on the other half's means. Averaged over splits. G-invariant (the assignment
    is equivariant, ties aside) and not fitted on the data it is evaluated on.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment

__all__ = [
    "act_on", "random_group_element", "permute_heads_",
    "positional_dm", "scalar_quotient_dm", "profile_orbit_distance",
    "crossfit_aligned_dm", "positional_permutation_null",
    "crossfit_orbit_distance", "orbit_prompt_bootstrap", "alignment_recovery",
]


# ── Group action ───────────────────────────────────────────────────────────────

def random_group_element(n_layers: int, n_heads: int,
                         rng: np.random.Generator) -> np.ndarray:
    """``[L, H]`` array; row l is the permutation pi_l."""
    return np.stack([rng.permutation(n_heads) for _ in range(n_layers)])


def act_on(pi: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``(pi . x)[..., l, h] = x[..., l, pi_l(h)]`` for x of shape ``[..., L, H]``."""
    x = np.asarray(x)
    layers = np.arange(pi.shape[0])[:, None]
    return x[..., layers, pi]


def permute_heads_(model: Any, pi: np.ndarray) -> None:
    """Apply the function-preserving head permutation to a HookedTransformer in place.

    Head h of layer l in the new model is head ``pi_l(h)`` of the old one.
    """
    import torch

    with torch.no_grad():
        for layer, perm in enumerate(pi):
            idx = torch.as_tensor(perm, dtype=torch.long)
            attn = model.blocks[layer].attn
            for name in ("W_Q", "W_K", "W_V", "W_O", "b_Q", "b_K", "b_V"):
                p = getattr(attn, name, None)
                if p is not None:
                    p.copy_(p[idx].clone())


# ── Distances ──────────────────────────────────────────────────────────────────

def _rho(x: np.ndarray, y: np.ndarray) -> float:
    return float(stats.spearmanr(np.ravel(x), np.ravel(y)).correlation)


def positional_dm(a: np.ndarray, b: np.ndarray) -> float:
    """Pre-registered D_M: ``1 - Spearman`` over positions (a, b: ``[L, H]``)."""
    return 1.0 - _rho(a, b)


def scalar_quotient_dm(a: np.ndarray, b: np.ndarray) -> float:
    """``min over G`` of positional D_M on scalar means = sort within each layer."""
    return 1.0 - _rho(np.sort(np.asarray(a), axis=1), np.sort(np.asarray(b), axis=1))


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x)
    if n == 0:
        raise ValueError("attribution tensor is all zeros")
    return x / n


def profile_orbit_distance(xa: np.ndarray, xb: np.ndarray) -> Dict[str, Any]:
    """Orbit distance between unit-normalised profile tensors ``[N, L, H]``."""
    a, b = _unit(xa), _unit(xb)
    if a.shape != b.shape:
        raise ValueError("tensors must share [N, L, H]")
    total, pi = 0.0, np.zeros(a.shape[1:], dtype=int)
    for layer in range(a.shape[1]):
        pa, pb = a[:, layer, :].T, b[:, layer, :].T  # [H, N]
        cost = ((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)
        r, c = linear_sum_assignment(cost)
        pi[layer, r] = c
        total += float(cost[r, c].sum())
    return {"value": total, "pi": pi}


def _profile_cost(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """``1 - Pearson`` between head profiles; constant profiles get cost 1."""
    za = pa - pa.mean(1, keepdims=True)
    zb = pb - pb.mean(1, keepdims=True)
    na = np.linalg.norm(za, axis=1, keepdims=True)
    nb = np.linalg.norm(zb, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = (za @ zb.T) / (na * nb.T)
    return 1.0 - np.nan_to_num(corr, nan=0.0)


def crossfit_aligned_dm(xa: np.ndarray, xb: np.ndarray, n_splits: int = 20,
                        seed: int = 0) -> Dict[str, Any]:
    """Positional D_M after head alignment fitted on a disjoint half of the items."""
    xa, xb = np.asarray(xa, dtype=float), np.asarray(xb, dtype=float)
    n, n_layers, _ = xa.shape
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        fit, ev = perm[: n // 2], perm[n // 2:]
        pi = np.zeros(xa.shape[1:], dtype=int)
        for layer in range(n_layers):
            r, c = linear_sum_assignment(
                _profile_cost(xa[fit, layer, :].T, xb[fit, layer, :].T))
            pi[layer, r] = c
        vals.append(positional_dm(xa[ev].mean(0), act_on(pi, xb[ev].mean(0))))
    return {"value": float(np.mean(vals)), "sd": float(np.std(vals)), "n_splits": n_splits}


# ── Null for the positional metric ─────────────────────────────────────────────

def positional_permutation_null(a: np.ndarray, b: np.ndarray, n_perm: int = 10000,
                                seed: int = 0, observed: Optional[float] = None
                                ) -> Dict[str, Any]:
    """Distribution of positional D_M(a, pi . b) over uniformly random pi in G.

    Ranks are permutation-equivariant, so ``rho(a, pi.b)`` is a Pearson correlation of
    ``rank(a)`` with ``pi . rank(b)`` (vectorised). The observed percentile answers: do the
    two models' head labels line up better than arbitrary labels would?
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ra = stats.rankdata(a.ravel()).reshape(a.shape)
    rb = stats.rankdata(b.ravel()).reshape(b.shape)
    za = (ra - ra.mean()) / ra.std()
    rng = np.random.default_rng(seed)
    n_l, n_h = a.shape
    perms = np.argsort(rng.random((n_perm, n_l, n_h)), axis=2)
    pb = np.take_along_axis(np.broadcast_to(rb, (n_perm, n_l, n_h)), perms, axis=2)
    zb = (pb - rb.mean()) / rb.std()
    null = 1.0 - (za[None] * zb).mean(axis=(1, 2))
    obs = positional_dm(a, b) if observed is None else observed
    return {
        "observed": float(obs), "n_perm": n_perm, "seed": seed,
        "mean": float(null.mean()), "median": float(np.median(null)),
        "p01": float(np.percentile(null, 1)), "p05": float(np.percentile(null, 5)),
        "p95": float(np.percentile(null, 95)), "p99": float(np.percentile(null, 99)),
        "observed_percentile": float(100.0 * np.mean(null <= obs)),
    }


# ── Baseline / uncertainty tools for profile_orbit_distance (Amendment 3 drafting) ──

def _orbit_with_fixed_pi(xa: np.ndarray, xb: np.ndarray, pi: np.ndarray) -> float:
    a, b = _unit(xa), _unit(xb)
    return float(((a - act_on(pi, b)) ** 2).sum())


def crossfit_orbit_distance(xa: np.ndarray, xb: np.ndarray, n_splits: int = 20,
                            seed: int = 0) -> Dict[str, Any]:
    """Orbit distance with the head matching fitted on held-out prompts.

    Per split: pi_hat = argmin on half S1; report ||X~_A,S2 - pi_hat . X~_B,S2||^2 and the
    in-sample minimum on S2. By construction in-sample(S2) <= cross-fitted(S2): the
    in-sample value is biased toward similarity (selection of pi on the same prompts), the
    cross-fitted one toward dissimilarity (matching error). Averaged over splits.
    """
    xa, xb = np.asarray(xa, dtype=float), np.asarray(xb, dtype=float)
    n = xa.shape[0]
    rng = np.random.default_rng(seed)
    cf, ins = [], []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        s1, s2 = perm[: n // 2], perm[n // 2: 2 * (n // 2)]
        pi = profile_orbit_distance(xa[s1], xb[s1])["pi"]
        cf.append(_orbit_with_fixed_pi(xa[s2], xb[s2], pi))
        ins.append(profile_orbit_distance(xa[s2], xb[s2])["value"])
    return {"crossfit": float(np.mean(cf)), "insample_half": float(np.mean(ins)),
            "n_splits": n_splits, "half_size": n // 2}


def orbit_prompt_bootstrap(xa: np.ndarray, xb: np.ndarray, n_boot: int = 1000,
                           seed: int = 0, alpha: float = 0.05) -> Dict[str, Any]:
    """Paired prompt bootstrap: resample prompts jointly for both models, re-solve.

    Quantifies sampling uncertainty of the plug-in estimate with respect to the prompt
    distribution. It is not a null distribution. Whole [L, H] slices are resampled
    together, so no independence across heads is assumed.
    """
    xa, xb = np.asarray(xa, dtype=float), np.asarray(xb, dtype=float)
    n = xa.shape[0]
    rng = np.random.default_rng(seed)
    est = profile_orbit_distance(xa, xb)["value"]
    boots = np.array([profile_orbit_distance(xa[i], xb[i])["value"]
                      for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": est, "boot_mean": float(boots.mean()),
            "bias": float(boots.mean() - est), "ci_percentile": [float(lo), float(hi)],
            "n_boot": n_boot, "seed": seed}


def alignment_recovery(xa: np.ndarray, xb: np.ndarray) -> Dict[str, float]:
    """For pairs with known correspondence (same run): does the optimal matching recover
    the identity? Reported unweighted and weighted by head magnitude."""
    pi = profile_orbit_distance(xa, xb)["pi"]
    ident = pi == np.arange(pi.shape[1])[None, :]
    w = np.sqrt((_unit(xa) ** 2).sum(0))  # [L, H] head magnitude in A
    return {"fraction_identity": float(ident.mean()),
            "magnitude_weighted_identity": float((w * ident).sum() / w.sum())}
