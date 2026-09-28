"""Scientific audit of one V6 pilot pair (read-only with respect to the library).

Given a saved pilot run directory, this script:

1. Reproduction: re-runs the identical ``run_diff`` config and compares every field of the
   new record with the saved one, excluding only volatile fields (run_id, timestamps,
   wall-clock seconds, environment).
2. Recomputes per-item attribution matrices and checks their mean equals the saved D_M input.
3. Cross-model split-half control: D_M between A and B computed on DISJOINT item halves,
   compared with the within-model split-half distances (Control B). If the between-model
   distance is not a sample artefact, between >> within on every split.
4. Instrument check: Taylor attribution vs exact last-position activation patching per head
   (the quantity Taylor approximates), per model, and D_M recomputed from exact patching.

Usage:
    python experiments/v6/audits/audit_pair.py RUN_DIR [--n-exact 20] [--skip-rerun]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy import stats

from glassbox.core import GlassboxV2, _decision_value
from glassbox.v6 import controls, measure
from glassbox.v6.claims import canonical_json
from glassbox.v6.diff import DiffConfig, run_diff
from glassbox.v6.distances import operational_mechanistic_distance
from glassbox.v6.tasks import build_ioi

VOLATILE = {"run_id", "started_utc", "seconds", "environment"}


def _strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in VOLATILE}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def _first_diff(a: Any, b: Any, path: str = "") -> str:
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}/{k}: missing on one side"
            r = _first_diff(a[k], b[k], f"{path}/{k}")
            if r:
                return r
        return ""
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            r = _first_diff(x, y, f"{path}[{i}]")
            if r:
                return r
        return ""
    return "" if a == b else f"{path}: {a!r} vs {b!r}"


def reproduce(run_dir: Path, cfg: DiffConfig) -> Dict[str, Any]:
    """Step 1: identical rerun, exact comparison of the non-volatile record."""
    old = json.loads((run_dir / "record.json").read_text())
    rerun_dir = run_dir / "audit_rerun"
    run_diff(cfg, rerun_dir)
    new = json.loads((rerun_dir / "record.json").read_text())
    a, b = _strip(old), _strip(new)
    same = canonical_json(a) == canonical_json(b)
    return {"identical_excluding_volatile": same, "first_difference": _first_diff(a, b),
            "volatile_fields_excluded": sorted(VOLATILE),
            "rerun_dir": str(rerun_dir)}


def exact_last_position_patching(model: Any, items: List[Any]) -> np.ndarray:
    """Per item, per head: LD(clean) - LD(clean with head z at last pos from corrupted).

    This is the exact quantity the Taylor attribution approximates to first order.
    """
    n_l, n_h = model.cfg.n_layers, model.cfg.n_heads
    out = np.zeros((len(items), n_l * n_h))
    for i, it in enumerate(items):
        clean = model.to_tokens(it.prompt)
        corr = model.to_tokens(GlassboxV2._name_swap(it.prompt, it.name_a, it.name_b))
        t, d = model.to_single_token(it.target), model.to_single_token(it.distractor)
        cache: Dict[str, torch.Tensor] = {}
        names = [f"blocks.{layer}.attn.hook_z" for layer in range(n_l)]

        def save(act, hook):
            cache[hook.name] = act.detach().clone()

        with torch.no_grad():
            base = float(_decision_value(model(clean), t, d, fp32_each=True))
            model.run_with_hooks(corr, fwd_hooks=[(n, save) for n in names])
            batch = clean.repeat(n_h, 1)
            for layer in range(n_l):
                src = cache[names[layer]][0, -1]  # [n_heads, d_head]

                def patch(act, hook, src=src):
                    for h in range(n_h):
                        act[h, -1, h, :] = src[h]
                    return act

                logits = model.run_with_hooks(batch, fwd_hooks=[(names[layer], patch)])
                ld = (logits[:, -1, t].float() - logits[:, -1, d].float()).cpu().numpy()
                out[i, layer * n_h:(layer + 1) * n_h] = base - ld
    return out


def _dm(x: np.ndarray, y: np.ndarray, heads: List[Tuple[int, int]]) -> float:
    return operational_mechanistic_distance(dict(zip(heads, x)), dict(zip(heads, y)))["value"]


def cross_split(pa: np.ndarray, pb: np.ndarray, heads: List[Tuple[int, int]],
                n_splits: int, seed: int) -> Dict[str, Any]:
    """Step 3: between-model D_M on disjoint halves vs within-model D_M."""
    between, within_a, within_b = [], [], []
    for s1, s2 in controls.split_indices(pa.shape[0], n_splits, seed):
        between.append(_dm(pa[s1].mean(0), pb[s2].mean(0), heads))
        within_a.append(_dm(pa[s1].mean(0), pa[s2].mean(0), heads))
        within_b.append(_dm(pb[s1].mean(0), pb[s2].mean(0), heads))
    between_arr = np.array(between)
    within_max = np.maximum(within_a, within_b)
    return {
        "n_splits": n_splits,
        "between_min": float(between_arr.min()), "between_median": float(np.median(between)),
        "within_a_max": float(np.max(within_a)), "within_b_max": float(np.max(within_b)),
        "splits_where_between_exceeds_within": int((between_arr > within_max).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--n-exact", type=int, default=20)
    ap.add_argument("--n-splits", type=int, default=200)
    ap.add_argument("--skip-rerun", action="store_true")
    args = ap.parse_args()
    rec = json.loads((args.run_dir / "record.json").read_text())
    fields = {f.name for f in dataclasses.fields(DiffConfig)}
    cfg = DiffConfig(**{k: v for k, v in rec["config"].items() if k in fields})
    report: Dict[str, Any] = {"run_dir": str(args.run_dir), "config": rec["config"]}

    if not args.skip_rerun:
        report["reproduction"] = reproduce(args.run_dir, cfg)
        print("reproduction:", report["reproduction"], flush=True)

    torch.manual_seed(cfg.seed)
    ma = measure.load_model(cfg.model_a, cfg.checkpoint_a, cfg.device)
    mb = measure.load_model(cfg.model_b, cfg.checkpoint_b, cfg.device)
    pa_, pb_ = measure.single_token_predicate(ma), measure.single_token_predicate(mb)
    ds = build_ioi(cfg.n_prompts, cfg.seed, lambda s: pa_(s) and pb_(s),
                   rec["dataset"]["tokenizer_id"])
    report["dataset_hash_matches"] = ds.dataset_hash == rec["dataset"]["hash"]

    heads, per_a = measure.head_attribution_matrix(ma, ds.items)
    _, per_b = measure.head_attribution_matrix(mb, ds.items)
    keys = [f"L{layer}H{h}" for layer, h in heads]
    saved_a = np.array([rec["model_a"]["attribution"][k] for k in keys])
    saved_b = np.array([rec["model_b"]["attribution"][k] for k in keys])
    report["mean_attribution_matches_saved"] = {
        "a_max_abs_diff": float(np.abs(per_a.mean(0) - saved_a).max()),
        "b_max_abs_diff": float(np.abs(per_b.mean(0) - saved_b).max()),
    }
    report["cross_model_split_half"] = cross_split(per_a, per_b, heads,
                                                   args.n_splits, cfg.seed)
    print("cross split:", report["cross_model_split_half"], flush=True)

    n = min(args.n_exact, len(ds.items))
    ex_a = exact_last_position_patching(ma, ds.items[:n])
    ex_b = exact_last_position_patching(mb, ds.items[:n])
    ta, tb = per_a[:n].mean(0), per_b[:n].mean(0)
    report["taylor_vs_exact"] = {
        "n_items": n,
        "spearman_taylor_exact_a": float(stats.spearmanr(ta, ex_a.mean(0)).correlation),
        "spearman_taylor_exact_b": float(stats.spearmanr(tb, ex_b.mean(0)).correlation),
        "pearson_taylor_exact_a": float(np.corrcoef(ta, ex_a.mean(0))[0, 1]),
        "pearson_taylor_exact_b": float(np.corrcoef(tb, ex_b.mean(0))[0, 1]),
        "D_M_taylor_same_items": _dm(ta, tb, heads),
        "D_M_exact_same_items": _dm(ex_a.mean(0), ex_b.mean(0), heads),
    }
    print("taylor vs exact:", report["taylor_vs_exact"], flush=True)
    out = args.run_dir / "audit.json"
    out.write_text(json.dumps(json.loads(canonical_json(report)), indent=2, sort_keys=True))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
