"""Does a paired prompt bootstrap give valid CIs for profile_orbit_distance?

The orbit distance minimises over head matchings, so it is a non-smooth function of the
data, and the bootstrap can be unreliable for such statistics. This checks coverage on the
fixed synthetic generator (identifiability_synthetic.py; unchanged). No real-model data.

Estimand D*: orbit distance on a large fresh prompt sample (N_POP) from the same generator,
approximating the population value for the pair. Per replicate: draw N prompts, compute
the plug-in estimate, a percentile bootstrap CI, and the cross-fitted estimate.

Run: python experiments/v6/audits/orbit_baseline_simulation.py [dense|sparse]
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np

from glassbox.v6 import identifiability as idf

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("idsyn", HERE / "identifiability_synthetic.py")
syn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(syn)

N, N_POP, REPS, BOOT = 200, 20000, 40, 200


def pair(scenario: str, sparse: bool, rng: np.random.Generator):
    """Return two structures (model A, model B) for a scenario."""
    sa = syn.structure(rng, sparse)
    if scenario == "A":
        sb = syn.relabel(sa, idf.random_group_element(syn.L, syn.H, rng))
    elif scenario == "B":
        sb = syn.perturb(sa, rng)
    else:  # "C"
        sb = syn.structure(rng, sparse)
    return sa, sb


def draw(sa, sb, n: int, rng: np.random.Generator):
    old = syn.N
    syn.N = n  # observe() draws N prompts
    f = rng.normal(size=(n, syn.K))
    xa, xb = syn.observe(sa, f, rng), syn.observe(sb, f, rng)
    syn.N = old
    return xa, xb


def run(regime: str) -> dict:
    out = {}
    for sc in "ABC":
        cover, bias_in, bias_cf, width = [], [], [], []
        cover_basic, cover_bracket, cover_wide = [], [], []
        for rep in range(REPS):
            rng = np.random.default_rng(1000 * rep + ord(sc))
            sa, sb = pair(sc, regime == "sparse", rng)
            d_star = idf.profile_orbit_distance(*draw(sa, sb, N_POP, rng))["value"]
            xa, xb = draw(sa, sb, N, rng)
            b = idf.orbit_prompt_bootstrap(xa, xb, n_boot=BOOT, seed=rep)
            cf = idf.crossfit_orbit_distance(xa, xb, n_splits=10, seed=rep)["crossfit"]
            lo, hi = b["ci_percentile"]
            cover.append(lo <= d_star <= hi)
            est = b["estimate"]
            cover_basic.append(2 * est - hi <= d_star <= 2 * est - lo)  # basic bootstrap
            cover_bracket.append(est <= d_star <= cf)  # in-sample <= D* <= cross-fit
            cover_wide.append(lo <= d_star <= cf + (hi - est))  # bracket + sampling error
            bias_in.append(b["estimate"] - d_star)
            bias_cf.append(cf - d_star)
            width.append(hi - lo)
        out[sc] = {"coverage": float(np.mean(cover)),
                   "coverage_basic": float(np.mean(cover_basic)),
                   "coverage_bracket": float(np.mean(cover_bracket)),
                   "coverage_bracket_plus_sampling": float(np.mean(cover_wide)),
                   "bias_insample": float(np.mean(bias_in)),
                   "bias_crossfit": float(np.mean(bias_cf)), "mean_ci_width": float(np.mean(width))}
        print(f"[{regime}] {sc}: percentile {out[sc]['coverage']:.2f} "
              f"basic {out[sc]['coverage_basic']:.2f} "
              f"bracket {out[sc]['coverage_bracket']:.2f} "
              f"bracket+samp {out[sc]['coverage_bracket_plus_sampling']:.2f} | "
              f"bias in-sample {out[sc]['bias_insample']:+.4f}  "
              f"bias cross-fit {out[sc]['bias_crossfit']:+.4f}  "
              f"CI width {out[sc]['mean_ci_width']:.4f}", flush=True)
    return out


if __name__ == "__main__":
    regime = sys.argv[1] if len(sys.argv) > 1 else "dense"
    res = run(regime)
    path = HERE / "orbit_baseline_simulation.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[regime] = {"N": N, "N_POP": N_POP, "REPS": REPS, "BOOT": BOOT, "results": res}
    path.write_text(json.dumps(data, indent=2))
