"""``glassbox-ai diff``: one model pair -> D_P, D_M, D_B, Controls A+B, record + Finding.

Milestone-1 scope (experiments/v6/AUDIT.md §E): Arm A IOI only, one pair per run.
A single pair cannot test H1-H4 (they need many model pairs), so
every hypothesis is recorded as UNRESOLVED and the run is labelled ``smoke`` or ``pilot``,
never ``confirmatory``.
"""
from __future__ import annotations

import dataclasses
import gc
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from glassbox.v6 import controls
from glassbox.v6 import distances as dist
from glassbox.v6.claims import (
    EvidenceStatus,
    Finding,
    HypothesisResult,
    Measurement,
    Scope,
    canonical_json,
)
from glassbox.v6.provenance import environment_record, sha256_json
from glassbox.v6.tasks import build_ioi

logger = logging.getLogger(__name__)

__all__ = ["DiffConfig", "run_diff", "HYPOTHESES"]

PROVISIONAL_MARGIN = 0.02

# Estimands follow PREREGISTRATION.md §4 (= PLAN_V6_FINAL.md §5, primary endpoints).
HYPOTHESES = [
    ("H1", "median D_M over D_P-matched pairs minus median D_M under Control B",
     "difference <= 0 (matched pairs look like measurement noise)",
     "permutation test over model labels; model-bootstrap CI"),
    ("H2", "AUROC of D_B separating divergent pairs from Control-B null pairs",
     "AUROC <= pre-registered threshold", "AUROC; model-bootstrap CI"),
    ("H3", "Mantel r between the D_B and D_M distance matrices",
     "r <= 0", "Mantel permutation test + sensitivity set (partial | D_P, LOMO, "
     "Spearman/Pearson, alternative definitions)"),
    ("H4", "Spearman(D_B, D_M) on held-out models",
     "rho <= 0", "held-out-by-model evaluation; model-bootstrap CI"),
]


@dataclasses.dataclass
class DiffConfig:
    """All run parameters; hashed into the record."""

    model_a: str
    model_b: str
    task: str = "ioi"
    checkpoint_a: Optional[int] = None
    checkpoint_b: Optional[int] = None
    n_prompts: int = 20
    seed: int = 0
    k: int = 10
    margin: float = PROVISIONAL_MARGIN
    label: str = "smoke"
    device: str = "cpu"
    n_splits: int = 200


def _measure(model: Any, ds: Any) -> Dict[str, Any]:
    from glassbox.v6 import measure

    t0 = time.perf_counter()
    correct, lds = measure.evaluate_items(model, ds.items)
    probes = measure.probe_distributions(model, ds.probes)
    heads, per_item = measure.head_attribution_matrix(model, ds.items)
    attr = dict(zip(heads, per_item.mean(axis=0).tolist()))
    return {"correct": correct, "ld": lds, "probes": probes, "attr": attr,
            "heads": heads, "per_item": per_item, "seconds": time.perf_counter() - t0}


def _control_b(cfg: DiffConfig, res_a: Dict[str, Any], res_b: Dict[str, Any],
               d_m: float) -> Dict[str, Any]:
    null_a = controls.split_half_null(res_a["per_item"], res_a["heads"],
                                      cfg.n_splits, cfg.seed)
    null_b = controls.split_half_null(res_b["per_item"], res_b["heads"],
                                      cfg.n_splits, cfg.seed + 1)
    return {
        "name": "Control B (resampling null: same model, disjoint item halves)",
        "model_a": null_a,
        "model_b": null_b,
        "threshold": max(null_a["p95"], null_b["p95"]),
        "rule": "divergent iff D_M > max(Control-B p95 of A, of B) [draft, "
                "PREREGISTRATION.md §4]",
        "pair_divergent": controls.is_divergent(d_m, [null_a["p95"], null_b["p95"]]),
        "limitation": "halves use n/2 items; the null overstates noise at n "
                      "(conservative)",
    }


def _control_a(first: Dict[str, Any], again: Dict[str, Any]) -> Dict[str, Any]:
    keys = sorted(first["attr"])
    a1 = np.array([first["attr"][h] for h in keys])
    a2 = np.array([again["attr"][h] for h in keys])
    diffs = {
        "attr_max_abs": float(np.max(np.abs(a1 - a2))),
        "ld_max_abs": float(np.max(np.abs(np.subtract(first["ld"], again["ld"])))),
        "probe_max_abs": float(np.max(np.abs(first["probes"] - again["probes"]))),
    }
    bitwise = (np.array_equal(a1, a2) and first["ld"] == again["ld"]
               and np.array_equal(first["probes"], again["probes"]))
    dm = dist.operational_mechanistic_distance(first["attr"], again["attr"])
    db = dist.behavioral_distance(first["probes"], again["probes"])
    return {
        "name": "Control A (pipeline null: same model, reloaded)",
        "bitwise_identical": bool(bitwise),
        "max_abs_diff": diffs,
        "D_M_self": dm["value"],
        "D_B_self": db["value"],
        "epsilon_det": None,
        "epsilon_det_status": "PENDING: set from pilot runs, see PREREGISTRATION.md §5",
        "status": "PASS_BITWISE" if bitwise else "RECORDED_TOLERANCE_PENDING",
    }


def _serialisable(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "correct": m["correct"], "ld": m["ld"],
        "attribution": {f"L{l}H{h}": v for (l, h), v in sorted(m["attr"].items())},
        "per_item_attribution_sha256": sha256_json(np.round(m["per_item"], 12).tolist()),
        "probe_dist_sha256": sha256_json(np.round(m["probes"], 12).tolist()),
        "inclusion": controls.above_chance(m["correct"]),
        "seconds": m["seconds"],
    }


def run_diff(cfg: DiffConfig, out_dir: Path) -> Finding:
    """Run one pair end to end and write ``record.json`` and ``finding.json``."""
    if cfg.task != "ioi":
        raise NotImplementedError(
            "--task credit needs the Arm B known-positive models, which are not built "
            "yet (milestone 2). Only --task ioi is available.")
    if cfg.label == "confirmatory":
        raise ValueError("a single-pair diff cannot be a confirmatory run")
    import torch

    from glassbox.v6 import measure

    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    model_a = measure.load_model(cfg.model_a, cfg.checkpoint_a, cfg.device)
    model_b = measure.load_model(cfg.model_b, cfg.checkpoint_b, cfg.device)
    if model_a.cfg.d_vocab != model_b.cfg.d_vocab:
        raise ValueError("D_B needs a shared output vocabulary")
    pa, pb = measure.single_token_predicate(model_a), measure.single_token_predicate(model_b)
    tok_id = f"{model_a.tokenizer.name_or_path}|{model_b.tokenizer.name_or_path}"
    ds = build_ioi(cfg.n_prompts, cfg.seed, lambda s: pa(s) and pb(s), tok_id)

    res_a, res_b = _measure(model_a, ds), _measure(model_b, ds)
    del model_a, model_b
    gc.collect()
    model_a2 = measure.load_model(cfg.model_a, cfg.checkpoint_a, cfg.device)
    control_a = _control_a(res_a, _measure(model_a2, ds))
    del model_a2
    distances = _distances(cfg, ds, res_a, res_b)
    control_b = _control_b(cfg, res_a, res_b, distances["D_M"]["value"])

    record = {
        "run_id": uuid.uuid4().hex,
        "started_utc": started,
        "config": dataclasses.asdict(cfg),
        "config_hash": sha256_json(dataclasses.asdict(cfg)),
        "margin_status": "PROVISIONAL (+-2pp), pending pre-registration lock",
        "environment": environment_record(),
        "dataset": {"version": ds.version, "hash": ds.dataset_hash, "seed": ds.seed,
                    "tokenizer_id": ds.tokenizer_id, "n_items": len(ds.items),
                    "n_probes": len(ds.probes)},
        "metrics_version": dist.METRICS_VERSION,
        "model_a": _serialisable(res_a),
        "model_b": _serialisable(res_b),
        "distances": distances,
        "controls": {"A": control_a, "B": control_b},
    }
    finding = _finding(cfg, ds, record, sha256_json(record))
    _write(out_dir, record, finding)
    return finding


def _distances(cfg: DiffConfig, ds: Any, res_a: Dict[str, Any],
               res_b: Dict[str, Any]) -> Dict[str, Any]:
    d_b = dist.behavioral_distance(res_a["probes"], res_b["probes"])
    kinds = [p.kind for p in ds.probes]
    d_b["by_kind"] = {k: float(np.mean([v for v, kk in zip(d_b["per_probe"], kinds)
                                        if kk == k])) for k in sorted(set(kinds))}
    return {
        "D_P": dist.performance_distance(res_a["correct"], res_b["correct"],
                                         res_a["ld"], res_b["ld"], margin=cfg.margin),
        "D_M": dist.operational_mechanistic_distance(res_a["attr"], res_b["attr"]),
        "D_M_topk": dist.topk_jaccard_distance(res_a["attr"], res_b["attr"], cfg.k),
        "D_B": d_b,
    }


def _write(out_dir: Path, record: Dict[str, Any], finding: Finding) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "record.json").write_text(json.dumps(json.loads(canonical_json(record)),
                                                    indent=2, sort_keys=True))
    payload = finding.to_dict()
    payload["finding_hash"] = finding.content_hash()
    (out_dir / "finding.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def _finding(cfg: DiffConfig, ds: Any, record: Dict[str, Any], rhash: str) -> Finding:
    d = record["distances"]
    cb = record["controls"]["B"]
    inc_a, inc_b = record["model_a"]["inclusion"], record["model_b"]["inclusion"]
    scope = Scope(task="ioi", models=[f"{cfg.model_a}@{cfg.checkpoint_a}",
                                      f"{cfg.model_b}@{cfg.checkpoint_b}"],
                  dataset_hash=ds.dataset_hash, metric_definitions=dist.METRICS_VERSION)
    measurements = [
        Measurement("D_P.accuracy_diff", d["D_P"]["accuracy_diff"],
                    {"tost_p": d["D_P"]["tost"]["p_tost"], "matched": d["D_P"]["matched"],
                     "margin": cfg.margin, "margin_status": "provisional"}),
        Measurement("D_M.one_minus_spearman", d["D_M"]["value"],
                    {"n_units": d["D_M"]["n_units"]}, reason=d["D_M"]["reason"]),
        Measurement("D_M.one_minus_topk_jaccard", d["D_M_topk"]["value"], {"k": cfg.k}),
        Measurement("D_B.mean_jsd_bits", d["D_B"]["value"], {"by_kind": d["D_B"]["by_kind"]}),
        Measurement("control_B.divergence_threshold", cb["threshold"],
                    {"pair_divergent": cb["pair_divergent"], "rule_status": "draft"}),
        Measurement("inclusion.accuracy_a", inc_a["accuracy"],
                    {"p_value": inc_a["p_value"], "above_chance": inc_a["above_chance"]}),
        Measurement("inclusion.accuracy_b", inc_b["accuracy"],
                    {"p_value": inc_b["p_value"], "above_chance": inc_b["above_chance"]}),
    ]
    hyps = [HypothesisResult(h, est, null, test, EvidenceStatus.UNRESOLVED,
                             reason="single-pair run; H1-H4 need many model pairs")
            for h, est, null, test in HYPOTHESES]
    return Finding(run_id=record["run_id"], scope=scope, measurements=measurements,
                   hypotheses=hyps, controls=record["controls"], record_hash=rhash,
                   label=cfg.label)
