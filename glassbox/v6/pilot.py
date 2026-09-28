"""``glassbox-ai pilot``: measure N models once each, then compute every pair.

``diff`` re-measures both models for every pair, so N models cost N(N-1) measurements.
The pilot runner measures each model once (plus one Control A reload of the first
model), then computes D_P, D_M, D_B and the Control B divergence rule for all pairs,
and writes the D_M / D_B / D_P matrices needed later for H1-H4 (Mantel etc.).

Pilot data are excluded from the confirmatory analysis (PREREGISTRATION.md §7), so every
hypothesis stays UNRESOLVED and the label can only be ``pilot`` or ``smoke``.
"""
from __future__ import annotations

import dataclasses
import gc
import time
import uuid
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from glassbox.v6 import controls, diff
from glassbox.v6 import distances as dist
from glassbox.v6.claims import EvidenceStatus, Finding, HypothesisResult, Measurement, Scope
from glassbox.v6.provenance import environment_record, sha256_json
from glassbox.v6.tasks import build_ioi

__all__ = ["PilotConfig", "parse_model_spec", "run_pilot"]


@dataclasses.dataclass
class PilotConfig:
    """Pilot parameters; hashed into the record. Models are ``name[@checkpoint]``."""

    models: List[str]
    n_prompts: int = 200
    seed: int = 0
    k: int = 10
    margin: float = diff.PROVISIONAL_MARGIN
    n_splits: int = 200
    device: str = "cpu"
    label: str = "pilot"


def parse_model_spec(spec: str) -> Tuple[str, Optional[int]]:
    """``"pythia-410m@143000"`` -> ``("pythia-410m", 143000)``; no ``@`` -> checkpoint None."""
    name, sep, ckpt = spec.partition("@")
    if not sep:
        return name, None
    if not ckpt.isdigit():
        raise ValueError(f"checkpoint must be an integer training step: {spec!r}")
    return name, int(ckpt)


# Thin seams over the model code (patched in tests).
def _load(name: str, checkpoint: Optional[int], device: str) -> Any:
    from glassbox.v6 import measure

    return measure.load_model(name, checkpoint, device)


def _predicate(model: Any) -> Callable[[str], bool]:
    from glassbox.v6 import measure

    return measure.single_token_predicate(model)


def _measure_model(model: Any, ds: Any) -> Dict[str, Any]:
    return diff._measure(model, ds)


def _validate(cfg: PilotConfig) -> None:
    if len(cfg.models) < 2:
        raise ValueError("a pilot needs at least 2 models")
    if len(set(cfg.models)) != len(cfg.models):
        raise ValueError("duplicate model specs")
    if cfg.label not in {"pilot", "smoke"}:
        raise ValueError("pilot runs can only be labelled 'pilot' or 'smoke'")
    for spec in cfg.models:
        parse_model_spec(spec)


def _measure_all(cfg: PilotConfig) -> Tuple[Any, List[Dict[str, Any]]]:
    """Load each model once, measure it, run Control B (and A for the first model)."""
    ds, vocab, results = None, None, []
    for idx, spec in enumerate(cfg.models):
        name, ckpt = parse_model_spec(spec)
        model = _load(name, ckpt, cfg.device)
        pred = _predicate(model)
        if ds is None:
            vocab = model.cfg.d_vocab
            ds = build_ioi(cfg.n_prompts, cfg.seed, pred, model.tokenizer.name_or_path)
        elif model.cfg.d_vocab != vocab:
            raise ValueError(f"{spec}: D_B needs a shared output vocabulary")
        bad = {n for it in ds.items for n in (it.name_a, it.name_b) if not pred(" " + n)}
        if bad:
            raise ValueError(f"{spec}: names not single-token for this tokenizer: {bad}")
        res = _measure_model(model, ds)
        del model
        gc.collect()
        ctrl: Dict[str, Any] = {"B": controls.split_half_null(
            res["per_item"], res["heads"], cfg.n_splits, cfg.seed + idx)}
        if idx == 0:
            again = _measure_model(_load(name, ckpt, cfg.device), ds)
            ctrl["A"] = diff._control_a(res, again)
            gc.collect()
        results.append({"spec": spec, "res": res, "controls": ctrl})
    return ds, results


def _pairs(cfg: PilotConfig, ds: Any, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, j in combinations(range(len(results)), 2):
        ra, rb = results[i], results[j]
        d = diff._distances(cfg, ds, ra["res"], rb["res"])
        p95s = [ra["controls"]["B"]["p95"], rb["controls"]["B"]["p95"]]
        out.append({
            "i": i, "j": j, "a": ra["spec"], "b": rb["spec"],
            "D_P_accuracy_diff": d["D_P"]["accuracy_diff"],
            "D_P_mean_ld_diff": d["D_P"]["mean_ld_diff"],
            "matched": d["D_P"]["matched"], "tost": d["D_P"]["tost"],
            "D_M": d["D_M"]["value"], "D_M_topk": d["D_M_topk"]["value"],
            "D_B": d["D_B"]["value"], "D_B_by_kind": d["D_B"]["by_kind"],
            "control_B_threshold": max(p95s),
            "divergent": controls.is_divergent(d["D_M"]["value"], p95s),
        })
    return out


def _matrix(n: int, pairs: List[Dict[str, Any]], key: str, sign: float = 1.0) -> list:
    m = np.zeros((n, n))
    for p in pairs:
        m[p["i"], p["j"]] = p[key]
        m[p["j"], p["i"]] = sign * p[key]
    return m.tolist()


def _finding(cfg: PilotConfig, ds: Any, record: Dict[str, Any]) -> Finding:
    scope = Scope(task="ioi", models=list(cfg.models), dataset_hash=ds.dataset_hash,
                  metric_definitions=dist.METRICS_VERSION)
    ms = []
    for p in record["pairs"]:
        tag = f"{p['a']}|{p['b']}"
        ms += [
            Measurement(f"{tag}.D_P.accuracy_diff", p["D_P_accuracy_diff"],
                        {"matched": p["matched"], "margin_status": "provisional"}),
            Measurement(f"{tag}.D_M.one_minus_spearman", p["D_M"],
                        {"divergent": p["divergent"], "rule_status": "draft"},
                        reason=None if p["D_M"] == p["D_M"] else "undefined D_M"),
            Measurement(f"{tag}.D_B.mean_jsd_bits", p["D_B"]),
        ]
    hyps = [HypothesisResult(h, est, null, test, EvidenceStatus.UNRESOLVED,
                             reason="pilot data; excluded from confirmatory analysis "
                                    "(PREREGISTRATION.md §7)")
            for h, est, null, test in diff.HYPOTHESES]
    return Finding(run_id=record["run_id"], scope=scope, measurements=ms,
                   hypotheses=hyps, controls={"see": "record.json models[*].controls"},
                   record_hash=sha256_json(record), label=cfg.label)


def run_pilot(cfg: PilotConfig, out_dir: Path) -> Dict[str, Any]:
    """Measure every model once, compute all pairs, write record.json + finding.json."""
    _validate(cfg)
    import torch

    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ds, results = _measure_all(cfg)
    pairs = _pairs(cfg, ds, results)
    n = len(results)
    record = {
        "run_id": uuid.uuid4().hex, "started_utc": started, "kind": "pilot",
        "config": dataclasses.asdict(cfg), "config_hash": sha256_json(dataclasses.asdict(cfg)),
        "margin_status": "PROVISIONAL (+-2pp), pending pre-registration lock",
        "environment": environment_record(),
        "dataset": {"version": ds.version, "hash": ds.dataset_hash, "seed": ds.seed,
                    "tokenizer_id": ds.tokenizer_id, "n_items": len(ds.items),
                    "n_probes": len(ds.probes),
                    "items_hash": sha256_json([dataclasses.asdict(i) for i in ds.items])},
        "metrics_version": dist.METRICS_VERSION,
        "models": [{"spec": r["spec"], **diff._serialisable(r["res"]),
                    "controls": r["controls"]} for r in results],
        "pairs": pairs,
        "matrices": {"order": list(cfg.models), "D_M": _matrix(n, pairs, "D_M"),
                     "D_B": _matrix(n, pairs, "D_B"),
                     "D_P_accuracy_diff": _matrix(n, pairs, "D_P_accuracy_diff", -1.0)},
    }
    diff._write(out_dir, record, _finding(cfg, ds, record))
    return record
