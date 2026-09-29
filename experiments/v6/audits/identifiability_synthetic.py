"""Synthetic validation of candidate D_M definitions (head-identifiability audit).

Generator (fixed before any metric was applied to seed data). A model's attribution
structure S = (m, W): per-head mean effect m[l, h] ~ N(0, s_l^2) and per-head loading
W[l, h] ~ N(0, I_k) on k item factors F (shared by all models, since items are shared).
Observed per-item attribution: X[i, l, h] = m[l, h] + s_l * F[i] . W[l, h] + noise.
Layer scales s_l rise then fall, mimicking real attribution concentrating in mid-late layers.

Scenarios (B_model compared with A_model = S + noise):
  F  same structure, fresh noise, no relabelling      (measurement floor)
  A  same structure, arbitrary within-layer relabelling
  B  same structure + small perturbation (eps), no relabelling
  C  independent structure from the same distribution  (different mechanism, same stats)
  D  C + arbitrary relabelling
  E  same per-head MEAN effects as A (relabelled), but new per-item behaviour (new W):
     a different mechanism with an identical multiset of scalar attributions
Desired: A ~ F (low), B low/moderate, C high, D == C, E high.
Regimes: "dense" (all heads active) and "sparse" (60 % of heads scaled by 0.01, matching
the ~60 % near-zero heads observed in Pythia-410M).

Run:  python experiments/v6/audits/identifiability_synthetic.py
"""
from __future__ import annotations

import json
import sys
from typing import Dict

import numpy as np

from glassbox.v6 import identifiability as idf

L, H, N, K = 12, 16, 200, 8
NOISE, EPS = 0.5, 0.25


def structure(rng: np.random.Generator, sparse: bool = False) -> Dict[str, np.ndarray]:
    s = np.exp(-((np.arange(L) - 0.7 * L) ** 2) / (2 * (L / 4) ** 2))[:, None]
    g = np.ones((L, H))
    if sparse:
        g[rng.random((L, H)) < 0.6] = 0.01
    return {"m": rng.normal(size=(L, H)) * s * g,
            "W": rng.normal(size=(L, H, K)) * g[..., None], "s": s, "g": g}


def observe(st: Dict[str, np.ndarray], f: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    signal = st["m"][None] + st["s"][None] * np.einsum("ik,lhk->ilh", f, st["W"])
    return signal + NOISE * st["s"].mean() * rng.normal(size=(N, L, H))


def relabel(st: Dict[str, np.ndarray], pi: np.ndarray) -> Dict[str, np.ndarray]:
    lay = np.arange(L)[:, None]
    return {"m": st["m"][lay, pi], "W": st["W"][lay, pi], "s": st["s"],
            "g": st["g"][lay, pi]}


def perturb(st: Dict[str, np.ndarray], rng: np.random.Generator) -> Dict[str, np.ndarray]:
    return {"m": st["m"] + EPS * st["s"] * rng.normal(size=(L, H)),
            "W": st["W"] + EPS * rng.normal(size=(L, H, K)) * st["g"][..., None],
            "s": st["s"], "g": st["g"]}


def metrics(xa: np.ndarray, xb: np.ndarray) -> Dict[str, float]:
    a, b = xa.mean(0), xb.mean(0)
    return {
        "positional": idf.positional_dm(a, b),
        "crossfit_aligned": idf.crossfit_aligned_dm(xa, xb, n_splits=10, seed=0)["value"],
        "profile_orbit": idf.profile_orbit_distance(xa, xb)["value"],
        "scalar_quotient": idf.scalar_quotient_dm(a, b),
    }


def replicate(seed: int, sparse: bool) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(N, K))
    sa, sc = structure(rng, sparse), structure(rng, sparse)
    xa = observe(sa, f, rng)
    pi1, pi2 = idf.random_group_element(L, H, rng), idf.random_group_element(L, H, rng)
    xc = observe(sc, f, rng)
    pi3 = idf.random_group_element(L, H, rng)
    re = relabel(sa, pi3)  # E: same multiset of mean effects and activity pattern ...
    se = {"m": re["m"], "s": sa["s"], "g": re["g"],  # ... but new per-item behaviour
          "W": rng.normal(size=(L, H, K)) * re["g"][..., None]}
    return {
        "F": metrics(xa, observe(sa, f, rng)),
        "A": metrics(xa, observe(relabel(sa, pi1), f, rng)),
        "B": metrics(xa, observe(perturb(sa, rng), f, rng)),
        "C": metrics(xa, xc),
        "D": metrics(xa, idf.act_on(pi2, xc)),  # exactly C's data, relabelled
        "E": metrics(xa, observe(se, f, rng)),
    }


def summarise(reps: list) -> Dict[str, Dict[str, list]]:
    return {sc: {k: [float(np.mean([r[sc][k] for r in reps])),
                     float(np.std([r[sc][k] for r in reps]))] for k in reps[0][sc]}
            for sc in "FABCDE"}


def main(n_rep: int = 30) -> int:
    result = {"n_rep": n_rep, "params": {"L": L, "H": H, "N": N, "K": K,
                                         "noise": NOISE, "eps": EPS}}
    for regime in ("dense", "sparse"):
        reps = [replicate(s, regime == "sparse") for s in range(n_rep)]
        out = summarise(reps)
        dc = max(abs(r["D"][k] - r["C"][k]) for r in reps
                 for k in ("crossfit_aligned", "profile_orbit", "scalar_quotient"))
        print(f"\n[{regime}] {'metric':<18}" + "".join(f"{s:>15}" for s in "FABCDE"))
        for k in reps[0]["F"]:
            print(f"{'':>9}{k:<18}" + "".join(
                f"{out[s][k][0]:>8.3f}±{out[s][k][1]:<5.3f}" for s in "FABCDE"))
        print(f"{'':>9}max |D - C| (invariant metrics): {dc:.1e}")
        result[regime] = {"summary": out, "max_abs_D_minus_C": dc}
    with open("experiments/v6/audits/identifiability_synthetic.json", "w") as fh:
        json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
