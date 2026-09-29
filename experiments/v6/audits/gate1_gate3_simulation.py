"""Gates 1 and 3 for Amendment 3 (criteria: amendment3_gate_criteria.md, fixed beforehand).

Gate 1: coverage of three prompt-level interval procedures for the orbit distance D*,
    in 9 conditions (regime x case), R replicates each.
Gate 3: synthetic discrimination (F, A, B, C, D, E) on the plug-in orbit distance.

Run one unit at a time (results are merged into gate1_gate3_results.json):
    python experiments/v6/audits/gate1_gate3_simulation.py gate1 dense similar
    python experiments/v6/audits/gate1_gate3_simulation.py gate3 sparse
    python experiments/v6/audits/gate1_gate3_simulation.py gate1v2   (all 9 cells; Mac)
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
from multiprocessing import Pool
from typing import Dict

import numpy as np

from glassbox.v6 import identifiability as idf

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("idsyn", HERE / "identifiability_synthetic.py")
syn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(syn)

R, N, N_POP, BOOT, K_CF = 200, 200, 20000, 200, 10
Z_SE = np.sqrt(0.95 * 0.05 / R)
THRESH = 0.95 - 2 * Z_SE
OUT = HERE / "gate1_gate3_results.json"
# Smoke-test override (never used for gate decisions): GATE_R=4 writes to a scratch file.
if os.environ.get("GATE_R"):
    R = int(os.environ["GATE_R"])
    OUT = pathlib.Path("/tmp/gate_smoke.json")
V2_SEED_OFFSET = 10000  # dated note 1: fresh seeds for the single allowed revision


def _structure(rng, regime: str):
    return syn.structure(rng, sparse=regime == "sparse", twins=regime == "unstable")


def _observe(st, f, rng):
    old = syn.N
    syn.N = f.shape[0]
    try:
        return syn.observe(st, f, rng)
    finally:
        syn.N = old


def _pair(case: str, regime: str, rng):
    sa = _structure(rng, regime)
    if case == "similar":
        sb = syn.relabel(sa, idf.random_group_element(syn.L, syn.H, rng))
    elif case == "perturbed":
        sb = syn.perturb(sa, rng)
    elif case == "different":
        sb = _structure(rng, regime)
    else:
        raise ValueError(case)
    return sa, sb


def gate1(regime: str, case: str) -> Dict[str, float]:
    hits = {"percentile": [], "basic": [], "bracketed": []}
    width = {k: [] for k in hits}
    bias_in, bias_cf = [], []
    for rep in range(R):
        rng = np.random.default_rng([rep, sum(map(ord, regime + case))])
        sa, sb = _pair(case, regime, rng)
        fp = rng.normal(size=(N_POP, syn.K))
        d_star = idf.profile_orbit_distance(_observe(sa, fp, rng), _observe(sb, fp, rng))["value"]
        f = rng.normal(size=(N, syn.K))
        xa, xb = _observe(sa, f, rng), _observe(sb, f, rng)
        b = idf.orbit_prompt_bootstrap(xa, xb, n_boot=BOOT, seed=rep)
        cf = idf.crossfit_orbit_distance(xa, xb, n_splits=K_CF, seed=rep)["crossfit"]
        est, (lo, hi) = b["estimate"], b["ci_percentile"]
        ints = {"percentile": (lo, hi), "basic": (2 * est - hi, 2 * est - lo),
                "bracketed": (lo, cf + (hi - est))}
        for k, (a, z) in ints.items():
            hits[k].append(a <= d_star <= z)
            width[k].append(z - a)
        bias_in.append(est - d_star)
        bias_cf.append(cf - d_star)
    res = {"R": R, "nominal": 0.95, "mc_se": float(Z_SE), "threshold": float(THRESH),
           "bias_insample": float(np.mean(bias_in)), "bias_crossfit": float(np.mean(bias_cf))}
    for k in hits:
        cov = float(np.mean(hits[k]))
        res[k] = {"coverage": cov, "mc_se_of_estimate": float(np.sqrt(cov * (1 - cov) / R)),
                  "mean_width": float(np.mean(width[k])), "pass": bool(cov >= THRESH)}
    return res


def _v2_rep(args):
    regime, case, rep = args
    rng = np.random.default_rng([V2_SEED_OFFSET + rep, sum(map(ord, regime + case))])
    sa, sb = _pair(case, regime, rng)
    fp = rng.normal(size=(N_POP, syn.K))
    d_star = idf.profile_orbit_distance(_observe(sa, fp, rng), _observe(sb, fp, rng))["value"]
    f = rng.normal(size=(N, syn.K))
    xa, xb = _observe(sa, f, rng), _observe(sb, f, rng)
    r = idf.bracketed_v2_interval(xa, xb, n_boot=BOOT, n_splits=K_CF, seed=V2_SEED_OFFSET + rep)
    return (r["lower"] <= d_star <= r["upper"], r["upper"] - r["lower"],
            d_star < r["lower"], d_star > r["upper"])


def gate1v2(regime: str, case: str, workers: int) -> Dict[str, object]:
    with Pool(workers) as pool:
        rows = pool.map(_v2_rep, [(regime, case, rep) for rep in range(R)])
    cov = float(np.mean([r[0] for r in rows]))
    return {"R": R, "nominal": 0.95, "threshold": float(THRESH), "coverage": cov,
            "mc_se_of_estimate": float(np.sqrt(cov * (1 - cov) / R)),
            "mean_width": float(np.mean([r[1] for r in rows])),
            "misses_below": int(sum(r[2] for r in rows)),
            "misses_above": int(sum(r[3] for r in rows)), "pass": bool(cov >= THRESH),
            "seed_offset": V2_SEED_OFFSET}


def gate3(regime: str) -> Dict[str, object]:
    rows = []
    for rep in range(R):
        rng = np.random.default_rng([rep, 7, sum(map(ord, regime))])
        f = rng.normal(size=(N, syn.K))
        sa, sc = _structure(rng, regime), _structure(rng, regime)
        xa = _observe(sa, f, rng)
        pi1, pi2 = (idf.random_group_element(syn.L, syn.H, rng) for _ in range(2))
        xc = _observe(sc, f, rng)
        re = syn.relabel(sa, idf.random_group_element(syn.L, syn.H, rng))
        se = {"m": re["m"], "s": sa["s"], "g": re["g"],
              "W": rng.normal(size=sa["W"].shape) * re["g"][..., None]}
        d = lambda y: idf.profile_orbit_distance(xa, y)["value"]  # noqa: E731
        rows.append({"F": d(_observe(sa, f, rng)),
                     "A": d(_observe(syn.relabel(sa, pi1), f, rng)),
                     "B": d(_observe(syn.perturb(sa, rng), f, rng)),
                     "C": d(xc), "D": d(idf.act_on(pi2, xc)),
                     "E": d(_observe(se, f, rng))})
    v = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    af = v["A"] - v["F"]
    crit = {
        "A_approx_F": bool(abs(af.mean()) <= 3 * af.std(ddof=1) / np.sqrt(R)),
        "B_gt_A": float(np.mean(v["B"] > v["A"])),
        "C_gt_B": float(np.mean(v["C"] > v["B"])),
        "D_eq_C_max_abs": float(np.max(np.abs(v["D"] - v["C"]))),
        "E_gt_B": float(np.mean(v["E"] > v["B"])),
    }
    passed = (crit["A_approx_F"] and crit["B_gt_A"] >= 0.95 and crit["C_gt_B"] >= 0.95
              and crit["D_eq_C_max_abs"] <= 1e-9 and crit["E_gt_B"] >= 0.95)
    return {"R": R, "means": {k: float(x.mean()) for k, x in v.items()},
            "sds": {k: float(x.std(ddof=1)) for k, x in v.items()},
            "mean_A_minus_F": float(af.mean()), "se_A_minus_F": float(af.std(ddof=1) / np.sqrt(R)),
            "criteria": crit, "pass": bool(passed)}


def _save(key: str, value: Dict[str, object]) -> None:
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data[key] = value
    OUT.write_text(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "gate1v2":
        workers = int(os.environ.get("WORKERS", os.cpu_count() or 1))
        for regime in ("dense", "sparse", "unstable"):
            for case in ("similar", "perturbed", "different"):
                r = gate1v2(regime, case, workers)
                _save(f"gate1v2/{regime}/{case}", r)
                print(regime, case, "coverage %.3f (below %d, above %d) width %.4f %s" % (
                    r["coverage"], r["misses_below"], r["misses_above"], r["mean_width"],
                    "PASS" if r["pass"] else "FAIL"), flush=True)
    elif which == "gate1":
        regime, case = sys.argv[2], sys.argv[3]
        r = gate1(regime, case)
        _save(f"gate1/{regime}/{case}", r)
        print(regime, case, {k: (r[k]["coverage"], round(r[k]["mean_width"], 4), r[k]["pass"])
                             for k in ("percentile", "basic", "bracketed")},
              "bias in/cf %.4f/%.4f" % (r["bias_insample"], r["bias_crossfit"]))
    else:
        regime = sys.argv[2]
        r = gate3(regime)
        _save(f"gate3/{regime}", r)
        print(regime, r["criteria"], "PASS" if r["pass"] else "FAIL")
