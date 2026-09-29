"""Gate 5: validity of run-level inference for Delta (criteria: amendment3_gate_criteria.md).

Worlds (synthetic generator identifiability_synthetic.py, unchanged; L=12, H=16, N=200):
  H0      Every checkpoint of every run is an independent perturbation of ONE shared
          structure S0; each run gets its own random head relabelling (applied to all its
          checkpoints). Within-run and cross-run pairs are then identically distributed,
          so Delta = 0 exactly (the H0 boundary).
  H1lam   Run structure S_r = S0 + lam * (independent draw - S0), lam in {0.15, 0.3};
          checkpoints perturb S_r. (lam set before any full run; the smoke run showed
          lam = 0.5 gives Delta ~ 0.6, too large to say anything about power.)
  H1strong Independent structure per run; checkpoints perturb S_r.
For each simulated experiment: 3 checkpoints per run, plug-in orbit distances, the primary
two-way bootstrap (runs + prompts, B = 200) and the jackknife fallback.
Validity (pre-declared): type-I error at H0 <= 0.081 for R = 6 and R = 10 runs.

Run (Mac; uses all cores):  python experiments/v6/audits/gate5_simulation.py
Smoke test:                 GATE_SIMS=2 python experiments/v6/audits/gate5_simulation.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
from itertools import combinations
from multiprocessing import Pool
from typing import Dict, List

import numpy as np

from glassbox.v6 import identifiability as idf
from glassbox.v6 import lineage

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("idsyn", HERE / "identifiability_synthetic.py")
syn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(syn)

SIMS = int(os.environ.get("GATE_SIMS", 200))
OUT = HERE / "gate5_results.json" if SIMS >= 200 else pathlib.Path("/tmp/gate5_smoke.json")
N, BOOT, T = 200, 200, 3
PAIRS = list(combinations(range(T), 2))  # checkpoint pairs (0,1), (0,2), (1,2); 2 = final
TYPE1_MAX = 0.05 + 2 * np.sqrt(0.95 * 0.05 / 200)
CELLS = [("H0", "sparse", 6), ("H0", "sparse", 10), ("H0", "dense", 6), ("H0", "dense", 10),
         ("H1lam0.15", "sparse", 6), ("H1lam0.15", "sparse", 10),
         ("H1lam0.3", "sparse", 6), ("H1strong", "sparse", 6)]


def _observe(st, f, rng):
    old = syn.N
    syn.N = f.shape[0]
    try:
        return syn.observe(st, f, rng)
    finally:
        syn.N = old


def _world(kind: str, regime: str, n_runs: int, rng) -> List[List[np.ndarray]]:
    sparse = regime == "sparse"
    s0 = syn.structure(rng, sparse)
    f = rng.normal(size=(N, syn.K))
    runs = []
    for _ in range(n_runs):
        if kind == "H0":
            base = s0
        elif kind == "H1strong":
            base = syn.structure(rng, sparse)
        else:  # "H1lam<value>"
            lam = float(kind[len("H1lam"):])
            z = syn.structure(rng, sparse)
            base = {"m": s0["m"] + lam * (z["m"] - s0["m"]),
                    "W": s0["W"] + lam * (z["W"] - s0["W"]), "s": s0["s"], "g": s0["g"]}
        pi = idf.random_group_element(syn.L, syn.H, rng)
        runs.append([_observe(syn.relabel(syn.perturb(base, rng), pi), f, rng)
                     for _ in range(T)])
    return runs


def _experiment(args) -> Dict[str, float]:
    kind, regime, n_runs, sim = args
    rng = np.random.default_rng([sim, n_runs, sum(map(ord, kind + regime))])
    runs = _world(kind, regime, n_runs, rng)
    counts = rng.multinomial(N, np.full(N, 1.0 / N), size=BOOT)
    within_b = np.zeros((BOOT, n_runs, len(PAIRS)))
    cross_b = np.zeros((BOOT, n_runs, n_runs))
    within = np.zeros((n_runs, len(PAIRS)))
    cross = np.zeros((n_runs, n_runs))
    for r in range(n_runs):
        for p, (s, t) in enumerate(PAIRS):
            within[r, p] = idf.profile_orbit_distance(runs[r][s], runs[r][t])["value"]
            within_b[:, r, p] = idf.orbit_distance_weighted(runs[r][s], runs[r][t], counts)
    for r, q in combinations(range(n_runs), 2):
        cross[r, q] = cross[q, r] = idf.profile_orbit_distance(runs[r][-1], runs[q][-1])["value"]
        d = idf.orbit_distance_weighted(runs[r][-1], runs[q][-1], counts)
        cross_b[:, r, q] = cross_b[:, q, r] = d
    boot = lineage.two_way_bootstrap(within_b, cross_b, within, cross, seed=sim)
    jack = lineage.jackknife(within, cross)
    return {"delta": boot["delta"], "boot_reject": boot["reject_h0"],
            "boot_lower": boot["lower_95"], "jack_reject": jack["reject_h0"]}


def main() -> int:
    workers = int(os.environ.get("WORKERS", os.cpu_count() or 1))
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    with Pool(workers) as pool:
        for kind, regime, n_runs in CELLS:
            key = f"{kind}/{regime}/R{n_runs}"
            rows = pool.map(_experiment, [(kind, regime, n_runs, s) for s in range(SIMS)])
            rate_b = float(np.mean([r["boot_reject"] for r in rows]))
            rate_j = float(np.mean([r["jack_reject"] for r in rows]))
            res = {"sims": SIMS, "mean_delta": float(np.mean([r["delta"] for r in rows])),
                   "reject_rate_bootstrap": rate_b, "reject_rate_jackknife": rate_j,
                   "mc_se_bootstrap": float(np.sqrt(rate_b * (1 - rate_b) / SIMS))}
            if kind == "H0":
                res["type1_max"] = float(TYPE1_MAX)
                res["bootstrap_valid"] = bool(rate_b <= TYPE1_MAX)
                res["jackknife_valid"] = bool(rate_j <= TYPE1_MAX)
            results[key] = res
            OUT.write_text(json.dumps(results, indent=2, sort_keys=True))
            print(key, json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
