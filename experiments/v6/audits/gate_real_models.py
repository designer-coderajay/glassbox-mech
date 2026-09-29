"""Gates 2 and 7 on real models (criteria: amendment3_gate_criteria.md, fixed beforehand).

Gate 2 (invariance under genuine function-preserving head permutations):
    X = attribution tensor of model M; X_pi = same after permuting M's head parameter
    slices; Y = a different checkpoint. Pass: D(X, X_pi) <= 1e-4 and
    |D(X_pi, Y) - D(X, Y)| <= sqrt(D(X,X_pi)) * (sqrt(D(X,Y)) + sqrt(D(X_pi,Y)))
    (triangle bound; dated note 3 replaced a mis-specified fixed 1e-4 tolerance).
Gate 7 (controlled perturbation sensitivity; NOT mechanism validation):
    target heads = top-10 by mean |attribution| in a held-out model on held-out prompts;
    scale their W_O by c in {1, 0.75, 0.5, 0.25, 0} in the target model; D(M, M_c) must be
    0 at c = 1 and strictly increase as c decreases. Low-attribution heads (rank > 100) at
    the same doses are reported only.
None of these models is in the confirmatory run set.

Mac (defaults as pre-declared):
    python experiments/v6/audits/gate_real_models.py
Sandbox validation on small models (not a gate result):
    python experiments/v6/audits/gate_real_models.py --small
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import stats

from glassbox.v6 import identifiability as idf
from glassbox.v6 import measure, tasks
from glassbox.v6.claims import canonical_json

HERE = pathlib.Path(__file__).resolve().parent
DOSES = [1.0, 0.75, 0.5, 0.25, 0.0]
TOL = 1e-4


def attr_tensor(model, items) -> np.ndarray:
    heads, mat = measure.head_attribution_matrix(model, items)
    n_l, n_h = model.cfg.n_layers, model.cfg.n_heads
    assert heads == [(layer, h) for layer in range(n_l) for h in range(n_h)]
    return mat.reshape(len(items), n_l, n_h)


def items_for(model, n: int, seed: int):
    return tasks.build_ioi(n, seed, measure.single_token_predicate(model),
                           model.tokenizer.name_or_path).items


def gate2(spec_m: Tuple[str, int], spec_y: Tuple[str, int], n: int) -> Dict[str, object]:
    m = measure.load_model(*spec_m)
    items = items_for(m, n, seed=0)
    x = attr_tensor(m, items)
    pi = idf.random_group_element(m.cfg.n_layers, m.cfg.n_heads, np.random.default_rng(0))
    idf.permute_heads_(m, pi)
    x_pi = attr_tensor(m, items)
    del m
    y = attr_tensor(measure.load_model(*spec_y), items)
    d_self = idf.profile_orbit_distance(x, x_pi)["value"]
    d_xy = idf.profile_orbit_distance(x, y)["value"]
    d_pi_y = idf.profile_orbit_distance(x_pi, y)["value"]
    return {"model": spec_m, "other": spec_y, "n_prompts": n,
            "D_self_vs_permuted": d_self, "D_x_y": d_xy, "D_xpi_y": d_pi_y,
            "abs_diff": abs(d_pi_y - d_xy),
            "equivariance_spearman": float(stats.spearmanr(
                idf.act_on(pi, x).ravel(), x_pi.ravel()).correlation),
            "triangle_bound": float(np.sqrt(d_self) * (np.sqrt(d_xy) + np.sqrt(d_pi_y))),
            # Dated note 3: cross condition is the metric (triangle) bound, not a fixed 1e-4.
            "pass": bool(d_self <= TOL and abs(d_pi_y - d_xy)
                         <= np.sqrt(d_self) * (np.sqrt(d_xy) + np.sqrt(d_pi_y)) + 1e-12)}


def _scale_heads(model, heads: List[Tuple[int, int]], c: float) -> List[torch.Tensor]:
    saved = []
    with torch.no_grad():
        for layer, h in heads:
            w = model.blocks[layer].attn.W_O
            saved.append(w[h].clone())
            w[h].mul_(c)
    return saved


def _restore(model, heads, saved) -> None:
    with torch.no_grad():
        for (layer, h), w0 in zip(heads, saved):
            model.blocks[layer].attn.W_O[h].copy_(w0)


def gate7(spec_heldout: Tuple[str, int], spec_target: Tuple[str, int], n: int) -> Dict[str, object]:
    ho = measure.load_model(*spec_heldout)
    held = np.abs(attr_tensor(ho, items_for(ho, n, seed=1)).mean(0))  # [L, H]
    n_h = ho.cfg.n_heads
    del ho
    order = np.argsort(-held.ravel())
    top = [(int(i) // n_h, int(i) % n_h) for i in order[:10]]
    rng = np.random.default_rng(0)
    low_pool = order[100:] if order.size > 110 else order[order.size // 2:]
    low = [(int(i) // n_h, int(i) % n_h) for i in rng.choice(low_pool, 10, replace=False)]
    m = measure.load_model(*spec_target)
    items = items_for(m, n, seed=0)
    x0 = attr_tensor(m, items)
    out: Dict[str, object] = {"heldout": spec_heldout, "target": spec_target,
                              "n_prompts": n, "top_heads": top, "low_heads": low}
    for label, heads in (("targeted", top), ("low_attribution", low)):
        ds = []
        for c in DOSES:
            saved = _scale_heads(m, heads, c)
            ds.append(idf.profile_orbit_distance(x0, attr_tensor(m, items))["value"])
            _restore(m, heads, saved)
        out[label] = dict(zip([str(c) for c in DOSES], ds))
    t = [out["targeted"][str(c)] for c in DOSES]
    out["pass"] = bool(t[0] <= 1e-12 and all(b > a for a, b in zip(t, t[1:])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="sandbox validation on pythia-70m")
    args = ap.parse_args()
    if args.small:
        g2 = gate2(("pythia-70m", 143000), ("pythia-70m", 71000), n=20)
        g7 = gate7(("pythia-70m", 71000), ("pythia-70m", 143000), n=20)
        out = pathlib.Path("/tmp/gate_real_small.json")
    else:
        g2 = gate2(("pythia-410m-deduped", 143000), ("pythia-410m-deduped", 133000), n=50)
        g7 = gate7(("pythia-410m-deduped", 123000), ("pythia-410m-deduped", 143000), n=100)
        out = HERE / "gate2_gate7_results.json"
    res = {"gate2": g2, "gate7": g7}
    out.write_text(json.dumps(json.loads(canonical_json(res)), indent=2, sort_keys=True))
    print("GATE 2", "PASS" if g2["pass"] else "FAIL",
          {k: g2[k] for k in ("D_self_vs_permuted", "abs_diff", "triangle_bound",
                              "equivariance_spearman")})
    print("GATE 7", "PASS" if g7["pass"] else "FAIL", "targeted", g7["targeted"])
    print("       low-attribution (report only)", g7["low_attribution"])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
