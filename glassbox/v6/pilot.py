"""``glassbox-ai pilot``: measure N models once each, then compute every pair.

``diff`` re-measures both models for every pair, so N models cost N(N-1) measurements.
The pilot runner measures each model once (plus one Control A reload of the first
model), then computes D_P, D_M, D_B and the (draft, positional) attribution-profile divergence flag for all pairs,
and writes the D_M / D_B / D_P matrices needed later for H1-H4 (Mantel etc.).

Pilot data are excluded from the confirmatory analysis (PREREGISTRATION.md §7), so every
hypothesis stays UNRESOLVED and the label can only be ``pilot`` or ``smoke``.
"""
from __future__ import annotations

import dataclasses
import gc
import json
import time
import uuid
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from glassbox.v6 import controls, diff
from glassbox.v6 import distances as dist
from glassbox.v6.claims import EvidenceStatus, Finding, HypothesisResult, Measurement, Scope
from glassbox.v6.provenance import environment_record, git_state, sha256_json
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


def _provenance(spec: str) -> Dict[str, Any]:
    from glassbox.v6 import measure

    return measure.hub_provenance(*parse_model_spec(spec))


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


# ── Per-model measurement cache ───────────────────────────────────────────────
# Each model's measurement is written to disk as soon as it finishes, so a failure
# (e.g. a download error at model 4) does not throw away models 1-3. A cache entry is
# reused only if the dataset hash, device, code commit and metric/task versions match.

def _cache_key(spec: str, ds: Any, cfg: PilotConfig) -> Dict[str, Any]:
    return {"spec": spec, "dataset_hash": ds.dataset_hash, "device": cfg.device,
            "git_sha": git_state()["sha"], "metrics_version": dist.METRICS_VERSION,
            "task_version": ds.version}


def _cache_paths(out_dir: Path, spec: str) -> Tuple[Path, Path]:
    cache = out_dir / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / ".gitignore").write_text("*\n")  # large arrays never go into git
    stem = spec.replace("/", "_").replace("@", "_step")
    return cache / f"{stem}.json", cache / f"{stem}.npz"


def _cache_load(out_dir: Path, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    meta_p, arr_p = _cache_paths(out_dir, key["spec"])
    if not (meta_p.exists() and arr_p.exists()):
        return None
    meta = json.loads(meta_p.read_text())
    if meta["key"] != key:
        return None
    arr = np.load(arr_p)
    heads = [tuple(int(v) for v in h) for h in arr["heads"]]
    res = {"correct": [bool(c) for c in arr["correct"]], "ld": arr["ld"].tolist(),
           "probes": arr["probes"], "per_item": arr["per_item"], "heads": heads,
           "attr": dict(zip(heads, arr["attr"].tolist())), "seconds": meta["seconds"]}
    return {"res": res, "controls": meta["controls"], "d_vocab": meta["d_vocab"]}


def _cache_save(out_dir: Path, key: Dict[str, Any], res: Dict[str, Any],
                ctrl: Dict[str, Any], d_vocab: int) -> None:
    meta_p, arr_p = _cache_paths(out_dir, key["spec"])
    np.savez(arr_p, correct=np.array(res["correct"]), ld=np.array(res["ld"]),
             probes=res["probes"], per_item=res["per_item"], heads=np.array(res["heads"]),
             attr=np.array([res["attr"][h] for h in res["heads"]]))
    meta_p.write_text(json.dumps({"key": key, "controls": ctrl, "d_vocab": d_vocab,
                                  "seconds": res["seconds"]}))


def _dataset(cfg: PilotConfig) -> Tuple[Any, int]:
    """Build the item set from the first model's tokenizer (loads it, no measuring)."""
    name, ckpt = parse_model_spec(cfg.models[0])
    model = _load(name, ckpt, cfg.device)
    ds = build_ioi(cfg.n_prompts, cfg.seed, _predicate(model), model.tokenizer.name_or_path)
    return ds, model.cfg.d_vocab


def _measure_one(cfg: PilotConfig, idx: int, ds: Any, vocab: int) -> Dict[str, Any]:
    spec = cfg.models[idx]
    name, ckpt = parse_model_spec(spec)
    model = _load(name, ckpt, cfg.device)
    if model.cfg.d_vocab != vocab:
        raise ValueError(f"{spec}: D_B needs a shared output vocabulary")
    pred = _predicate(model)
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
    return {"res": res, "controls": ctrl, "d_vocab": vocab}


def _measure_all(cfg: PilotConfig, out_dir: Path) -> Tuple[Any, List[Dict[str, Any]]]:
    """Measure each model once (or reuse its cache entry); Control B for all, A for #0."""
    ds, vocab = _dataset(cfg)
    results = []
    for idx, spec in enumerate(cfg.models):
        key = _cache_key(spec, ds, cfg)
        hit = _cache_load(out_dir, key)
        if hit is not None and hit["d_vocab"] != vocab:
            raise ValueError(f"{spec}: D_B needs a shared output vocabulary")
        entry = hit or _measure_one(cfg, idx, ds, vocab)
        if hit is None:
            _cache_save(out_dir, key, entry["res"], entry["controls"], vocab)
        results.append({"spec": spec, "res": entry["res"], "controls": entry["controls"],
                        "from_cache": hit is not None})
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
    ds, results = _measure_all(cfg, out_dir)
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
        "models": [{"spec": r["spec"], "hub": _provenance(r["spec"]),
                    **diff._serialisable(r["res"]),
                    "controls": r["controls"], "from_cache": r["from_cache"]}
                   for r in results],
        "pairs": pairs,
        "matrices": {"order": list(cfg.models), "D_M": _matrix(n, pairs, "D_M"),
                     "D_B": _matrix(n, pairs, "D_B"),
                     "D_P_accuracy_diff": _matrix(n, pairs, "D_P_accuracy_diff", -1.0)},
    }
    diff._write(out_dir, record, _finding(cfg, ds, record))
    return record
