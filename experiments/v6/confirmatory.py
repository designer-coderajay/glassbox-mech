"""V6 confirmatory runner: implements ONLY the locked protocol of experiments/v6/AMENDMENT3.md.

There are deliberately no options for the distance, attribution method, prompts, runs,
checkpoints, thresholds, bootstrap size, inference method or decision rule. Everything is
fixed in ``PROTOCOL`` and checked against ``EXPECTED_PROTOCOL_SHA256`` before anything runs.

    python experiments/v6/confirmatory.py --out experiments/v6/runs/confirmatory_v1
    python experiments/v6/confirmatory.py --pipeline-validation --out /tmp/pipeline_check
    (``--free-disk`` deletes each checkpoint's cached weights after it is measured.)

Hard failures (``ProtocolError``) stop the run: protocol/hash mismatch, unpinned revision,
resolved commit != pinned commit, weights sha256 mismatch, prompt-set invariant violation,
tokenizer mismatch, or an existing record in the output directory. A checkpoint that fails to
load or measure for any other reason excludes its run, with the error recorded (§4.1).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import re
import sys
import time
from itertools import combinations
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from glassbox.v6 import controls, lineage, tasks
from glassbox.v6.claims import canonical_json
from glassbox.v6.distances import exact_paired_equivalence
from glassbox.v6.identifiability import (
    crossfit_orbit_distance,
    orbit_distance_weighted,
    profile_orbit_distance,
)
from glassbox.v6.provenance import environment_record, git_state

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIRMATORY_LABEL = "CONFIRMATORY"
VALIDATION_LABEL = "PIPELINE VALIDATION ONLY — NOT CONFIRMATORY DATA"
TEST_LABEL = "UNIT TEST — NOT DATA"
CHECKPOINTS = [123000, 133000, 143000]
FINAL = 143000
VOCAB_SHA256 = "9f23fceff4898e6d7b9000708ff3252e3b7e9a282a2e268236e95a32c65809b2"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")

CLAIM_BOUNDARY = (
    "The confirmatory experiment can provide evidence for attribution-profile divergence "
    "under the preregistered estimand. It does not by itself establish mechanistic "
    "divergence, different causal mechanisms, different circuits, or a causal explanation "
    "of training-induced differences. Such stronger claims require additional "
    "intervention or causal evidence.")
GATE1_NOTE = (
    "Gate 1 (pair-level interval validity) FAILED and is CLOSED; no third attempt. "
    "Per-pair D* values are descriptive point estimates only; no pair-level confidence "
    "intervals are reported. H1 inference is at the training-run level only (Gate 5).")


def _runs(repos: Sequence[Tuple[str, bool, Tuple[str, str, str]]]) -> List[Dict[str, Any]]:
    return [{"repo": r, "use_safetensors": st,
             "commits": {str(c): s for c, s in zip(CHECKPOINTS, shas)}} for r, st, shas in repos]


_PROMPTS = {
    "attribution": {"path": "experiments/v6/prompts/ioi_attribution_v1.json", "n": 200,
                    "sha256": "aa3e32c8ab51d4aabe96e83b51d30a947e120b68bfcbf0da1dff87868abee0d2"},
    "matching": {"path": "experiments/v6/prompts/ioi_matching_v1.json", "n": 1000,
                 "sha256": "0849499b8117dac405743c2b33f500545f42aa1f4faff85461feecdef475d296"},
}
_COMMON = {
    "version": "v6-amendment3/1.0",
    "checkpoints": CHECKPOINTS,
    "within_pairs": [[123000, 133000], [123000, 143000], [133000, 143000]],
    "prompts": _PROMPTS,
    "metric": "profile_orbit_distance",
    "estimator": "plug-in",
    "inference": "two_way_bootstrap",
    "n_boot": 200,
    "alpha": 0.05,
    "inclusion": {"test": "one-sided exact binomial vs 0.5", "alpha": 0.05,
                  "set": "matching", "checkpoint": FINAL},
    "matching": {"rule": "exact_paired_equivalence", "design": "star", "margin": 0.02,
                 "alpha": 0.05, "set": "matching", "checkpoint": FINAL},
    "r_min": 4,
    "seeds": {"prompt_resample": 20260929, "run_resample": 20260930, "s3_crossfit": 20260931},
    "s3_n_splits": 20,
}

PROTOCOL: Dict[str, Any] = {
    **copy.deepcopy(_COMMON),
    "label": CONFIRMATORY_LABEL,
    "base_model": "pythia-410m",
    "reference": "EleutherAI/pythia-410m",
    "s2_exclude": "EleutherAI/pythia-410m-seed3",
    "runs": _runs([
        ("EleutherAI/pythia-410m", True, ("51cabaff46863a0421bf3b44194431cc94d3466e",
                                          "45c291dfb81b96ad2522eabf89eb953045cd537a",
                                          "bba6a464f54bbf08fc174cfb351d9794d58af21d")),
        ("EleutherAI/pythia-410m-seed1", False, ("3a55097f9efd407b7e7f780c1860d7514ec7bd86",
                                                 "2c44de039f189452fc0a3c90a17916b223c1b1b4",
                                                 "a77e7d87906f1c77138c009121f9d6439144663c")),
        ("EleutherAI/pythia-410m-seed2", False, ("319875fb868a5cd6636423c0b5ff95f786c5cd18",
                                                 "59534882c2c34c7d7b8a41bb9e251164477de8a1",
                                                 "8f8a4fabfcd5f2504ee550cb33771760d6cec419")),
        ("EleutherAI/pythia-410m-seed3", False, ("1cb2ac3dd2b729f33a208c0c9ed4da16ebd066b8",
                                                 "a2a1327db3a383c0f77c4b53ea483ddd4c6d4ae5",
                                                 "83438f87adb388fb205a83ad516e3b349b5cfcba")),
        ("EleutherAI/pythia-410m-seed4", False, ("b884f27285abb230c4503db0a0c7f0a108651fae",
                                                 "f1dee9051011ae3b7bd191a82db07bb8c71cb702",
                                                 "b122bdb9065e19909d34653da26e06e49fd4ded2")),
        ("EleutherAI/pythia-410m-seed5", False, ("e48b7a7fe656ffc0886b83de9af62a877ccb68bc",
                                                 "8b6f55ae443381e65968483290be8d0c2557db43",
                                                 "b9156c2847b0e50f298706af140656bbf0b7cf06")),
        ("EleutherAI/pythia-410m-seed6", False, ("ce6bdbca13fe93cb6b142721a0801fb7b9ae8611",
                                                 "eba0d2f483d7163dfeac5e89e258af476a1afee2",
                                                 "36a33e1487d89429085aec228a9ccfe831271243")),
        ("EleutherAI/pythia-410m-seed7", False, ("a12420095e08c46806901446f76a01927973fa70",
                                                 "fbad49e4aced1398fab8d94b531c63cbfa60c70a",
                                                 "4c83aca7bf06be6061797566759d4cf370307b7b")),
        ("EleutherAI/pythia-410m-seed8", False, ("52449162281639c3cf5000d31ac5959fa4ec32a8",
                                                 "453deb67190a41b3f4a65080df130480fb62b5a4",
                                                 "b3d0ef9e3205c5f235e44c7b8e8791ce04301452")),
        ("EleutherAI/pythia-410m-seed9", False, ("1933c5b650555211652330d6864ecf50ae7a8bc5",
                                                 "582bc5e55e7535ebb395c0fccefe7c87aa16db88",
                                                 "6155dcb7e1a9080a5cb61d0ef1f8041e91b5cc7a")),
    ]),
}

VALIDATION_PROTOCOL: Dict[str, Any] = {
    **copy.deepcopy(_COMMON),
    "label": VALIDATION_LABEL,
    "base_model": "pythia-70m",
    "reference": "EleutherAI/pythia-70m",
    "s2_exclude": "EleutherAI/pythia-70m-seed3",
    "runs": _runs([
        ("EleutherAI/pythia-70m", True, ("d91619cf1e21f4d2e2a815e76892d5b4be8218c6",
                                         "1dc9e11f4b9373e1f1ce5fc04ea327ef8a551aea",
                                         "de3e4e2d6cbb3b1a51f90fe153ea51fcd8c7c852")),
        ("EleutherAI/pythia-70m-seed1", False, ("006a97b32976c73cb616a385ff54fdf82b3692d0",
                                                "5aaf2b73af9afb87c0c0f1f39062e3109841c011",
                                                "a94dc37ad8326640e664560d32a0d4a229253fce")),
        ("EleutherAI/pythia-70m-seed2", False, ("f08eaeaac518c8c4c6d03236034448b5db240a24",
                                                "56328564e01954f7e7fbc5d2a1a84735fee394e2",
                                                "258e7aedecacc445bdc68229398e49b9e90d59e0")),
        ("EleutherAI/pythia-70m-seed3", False, ("7d5b11b31de3bae319fa2251dc8e6bd7e2c8940e",
                                                "6e10ee19454e4e8f624ba1ac94029333429c20b2",
                                                "d6d278fb9b57ed4ce7faf49ad0aae4ee0805936b")),
    ]),
}

EXPECTED_PROTOCOL_SHA256 = "b3d55e106aaf584c3a2dda3b9ced43607f96b76aa98abf4339ba7ca56c5bd706"
EXPECTED_VALIDATION_SHA256 = "5937ba9c91b68b19c07f8507ce08909567a80938ccea5f9bcb1620af9ef302f0"


class ProtocolError(RuntimeError):
    """A violation of the locked protocol. Always fatal."""


# ── protocol ─────────────────────────────────────────────────────────────────────

def protocol_hash(p: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(p).encode()).hexdigest()


def within_pairs(p: Dict[str, Any]) -> List[Tuple[int, int]]:
    return [tuple(x) for x in p["within_pairs"]]


def validate_protocol(p: Dict[str, Any], expected_sha256: str) -> None:
    """Reject anything that is not exactly the locked protocol."""
    if protocol_hash(p) != expected_sha256:
        raise ProtocolError("protocol hash mismatch: the protocol differs from the locked one")
    checks = [
        (p["metric"] == "profile_orbit_distance", "metric must be profile_orbit_distance"),
        (p["estimator"] == "plug-in", "estimator must be plug-in"),
        (p["inference"] == "two_way_bootstrap", "inference must be two_way_bootstrap"),
        (p["n_boot"] == 200, "n_boot must be 200 (inherited from Gate 5)"),
        (p["alpha"] == 0.05, "alpha must be 0.05"),
        (p["r_min"] == 4, "r_min must be 4 (gate criteria note 6)"),
        (p["checkpoints"] == CHECKPOINTS, "checkpoint set must be 123000/133000/143000"),
        (p["within_pairs"] == [[123000, 133000], [123000, 143000], [133000, 143000]],
         "within-lineage checkpoint pairs are fixed"),
        (p["matching"]["margin"] == 0.02 and p["matching"]["alpha"] == 0.05
         and p["matching"]["design"] == "star", "matching rule is fixed"),
        (p["inclusion"]["alpha"] == 0.05, "inclusion rule is fixed"),
        (p["seeds"] == _COMMON["seeds"], "seeds are fixed"),
        (p["label"] in {CONFIRMATORY_LABEL, VALIDATION_LABEL, TEST_LABEL}, "unknown label"),
        (p["reference"] == p["runs"][0]["repo"], "reference must be the first run"),
    ]
    for ok, msg in checks:
        if not ok:
            raise ProtocolError(msg)
    for run in p["runs"]:
        if sorted(int(c) for c in run["commits"]) != CHECKPOINTS:
            raise ProtocolError(f"{run['repo']}: checkpoint set differs from the protocol")
        for c, sha in run["commits"].items():
            if not isinstance(sha, str) or not _HEX40.match(sha):
                raise ProtocolError(f"{run['repo']}@{c}: revision must be a pinned 40-hex "
                                    f"commit SHA, got {sha!r}")


LOCK_TAG = "v6-amendment3-lock"


def _tags_at_head() -> List[str]:
    import subprocess

    try:
        out = subprocess.run(["git", "--no-optional-locks", "tag", "--points-at", "HEAD"],
                             cwd=ROOT, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return []
    return out.stdout.split()


def require_locked_commit(label: str) -> None:
    """A CONFIRMATORY run must execute exactly the locked, committed code."""
    if label != CONFIRMATORY_LABEL:
        return
    g = git_state()
    if g.get("sha") is None or g.get("dirty") is not False:
        raise ProtocolError("confirmatory run requires a clean git working tree at a known commit")
    if LOCK_TAG not in _tags_at_head():
        raise ProtocolError(f"confirmatory run requires HEAD to carry the tag {LOCK_TAG}")


def distance_function(name: str):
    if name != "profile_orbit_distance":
        raise ProtocolError(f"metric {name!r} is not allowed; only profile_orbit_distance")
    return lambda a, b: profile_orbit_distance(a, b)["value"]


# ── prompts ──────────────────────────────────────────────────────────────────────

def items_hash(rows: List[Dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json(rows).encode()).hexdigest()


def _path(p: str) -> pathlib.Path:
    q = pathlib.Path(p)
    return q if q.is_absolute() else ROOT / q


def load_prompt_sets(p: Dict[str, Any]) -> Dict[str, List[tasks.IOIItem]]:
    """Load the locked prompt files and enforce every invariant (no regeneration)."""
    out, prompts = {}, {}
    for name, prefix in (("attribution", "A"), ("matching", "M")):
        spec = p["prompts"][name]
        rows = json.loads(_path(spec["path"]).read_text())["items"]
        if len(rows) != spec["n"]:
            raise ProtocolError(f"{name}: prompt count {len(rows)} != locked {spec['n']}")
        if items_hash(rows) != spec["sha256"]:
            raise ProtocolError(f"{name}: prompt-set sha256 differs from the locked value")
        if [r["id"] for r in rows] != [f"{prefix}{i:04d}" for i in range(len(rows))]:
            raise ProtocolError(f"{name}: prompt IDs are not in the locked order")
        texts = [r["prompt"] for r in rows]
        if len(set(texts)) != len(texts):
            raise ProtocolError(f"{name}: duplicate prompt")
        prompts[name] = set(texts)
        fields = [f.name for f in tasks.IOIItem.__dataclass_fields__.values()]
        out[name] = [tasks.IOIItem(**{k: r[k] for k in fields}) for r in rows]
    if prompts["attribution"] & prompts["matching"]:
        raise ProtocolError("prompt overlap between attribution and matching sets")
    return out


# ── measurement ──────────────────────────────────────────────────────────────────

def _cache_paths(cache_dir: pathlib.Path, repo: str, sha: str):
    stem = repo.replace("/", "__") + "@" + sha
    return cache_dir / f"{stem}.json", cache_dir / f"{stem}.npz"


def _cache_key(p: Dict[str, Any], repo: str, sha: str) -> Dict[str, str]:
    return {"protocol_sha256": protocol_hash(p), "repo": repo, "pinned_commit": sha,
            "runner_sha256": _file_sha256(pathlib.Path(__file__))}


def _cache_read(cache_dir, p, repo, sha):
    meta_p, arr_p = _cache_paths(cache_dir, repo, sha)
    if not (meta_p.exists() and arr_p.exists()):
        return None
    meta = json.loads(meta_p.read_text())
    if meta.get("key") != _cache_key(p, repo, sha):
        return None
    arr = np.load(arr_p)
    correct = [bool(x) for x in arr["correct"]] if "correct" in arr.files else None
    return {**meta["rec"], "from_cache": True}, arr["attr"], correct


def _cache_write(cache_dir, p, repo, sha, rec, attr, correct) -> None:
    """Atomic: write temp files, then rename (a killed process leaves no partial entry)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_p, arr_p = _cache_paths(cache_dir, repo, sha)
    tmp_arr = arr_p.with_suffix(".tmp.npz")
    arrays = {"attr": attr} if correct is None else {"attr": attr, "correct": np.array(correct)}
    np.savez(tmp_arr, **arrays)
    tmp_arr.replace(arr_p)
    tmp_meta = meta_p.with_suffix(".tmp")
    tmp_meta.write_text(json.dumps({"key": _cache_key(p, repo, sha), "rec": rec}))
    tmp_meta.replace(meta_p)


def _measure_checkpoint(p, backend, run, ckpt, prompts, names, shape, cache_dir):
    """Load at the pinned SHA, verify, measure. ProtocolError is fatal; others exclude.

    A verified, successfully measured checkpoint is cached (keyed by protocol hash, runner
    hash and pinned commit) so an interrupted run resumes with identical inputs; reuse is
    flagged ``from_cache`` in the record. Failures are never cached.
    """
    sha = run["commits"][str(ckpt)]
    hit = _cache_read(cache_dir, p, run["repo"], sha)
    if hit is not None:
        return hit
    rec: Dict[str, Any] = {"pinned_commit": sha, "resolved_commit": None, "weights": None,
                           "status": "failed", "attempts": [], "from_cache": False}
    model = None
    for attempt in (1, 2):  # one retry, identical pinned configuration, recorded
        try:
            model, resolved = backend.load(run["repo"], sha, p["base_model"],
                                           run["use_safetensors"])
            rec["attempts"].append({"attempt": attempt, "ok": True})
            break
        except ProtocolError:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded, then run excluded
            rec["attempts"].append({"attempt": attempt, "ok": False,
                                    "error": f"{type(exc).__name__}: {exc}"})
    if model is None:
        rec["error"] = rec["attempts"][-1]["error"]
        return rec, None, None
    rec["resolved_commit"] = resolved
    if resolved != sha:
        raise ProtocolError(f"{run['repo']}@{ckpt}: commit mismatch (pinned {sha}, "
                            f"resolved {resolved})")
    rec["weights"] = backend.weights_check(run["repo"], sha, run["use_safetensors"])
    if not rec["weights"]["ok"]:
        raise ProtocolError(f"{run['repo']}@{ckpt}: weights sha256 mismatch")
    if not backend.tokenizer_single_token(model, names):
        raise ProtocolError(f"{run['repo']}@{ckpt}: tokenizer mismatch")
    try:
        attr = np.asarray(backend.attributions(model, prompts["attribution"]), dtype=float)
        if shape[0] is not None and attr.shape != shape[0]:
            raise ProtocolError(f"{run['repo']}@{ckpt}: attribution shape {attr.shape} "
                                f"!= {shape[0]}")
        correct = backend.correctness(model, prompts["matching"]) if ckpt == FINAL else None
    except ProtocolError:
        raise
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        backend.release(model)
        return rec, None, None
    backend.release(model)
    rec["status"] = "ok"
    _cache_write(cache_dir, p, run["repo"], sha, rec, attr, correct)
    return rec, attr, correct


def measure_all(p, backend, prompts, cache_dir):
    names = sorted({n for it in prompts["attribution"] + prompts["matching"]
                    for n in (it.name_a, it.name_b)})
    shape: List[Any] = [None]
    runs: Dict[str, Any] = {}
    tensors: Dict[Tuple[str, int], np.ndarray] = {}
    correct: Dict[str, List[bool]] = {}
    for run in p["runs"]:
        ck: Dict[str, Any] = {}
        for c in p["checkpoints"]:
            rec, attr, corr = _measure_checkpoint(p, backend, run, c, prompts, names, shape,
                                                  cache_dir)
            ck[str(c)] = rec
            if attr is not None:
                shape[0] = shape[0] or attr.shape
                tensors[(run["repo"], c)] = attr
            if corr is not None:
                correct[run["repo"]] = [bool(x) for x in corr]
        runs[run["repo"]] = {"checkpoints": ck,
                             "complete": all(v["status"] == "ok" for v in ck.values())}
    return runs, tensors, correct


# ── eligibility (AMENDMENT3.md §4) ───────────────────────────────────────────────

def eligibility(p, runs, correct):
    ref = p["reference"]
    for repo, r in runs.items():
        r["inclusion"] = (controls.above_chance(correct[repo], chance=0.5,
                                                alpha=p["inclusion"]["alpha"])
                          if r["complete"] else None)
        r["matching"] = None
    ref_ok = runs[ref]["complete"] and runs[ref]["inclusion"]["above_chance"]
    for repo, r in runs.items():
        if repo != ref and r["complete"] and runs[ref]["complete"]:
            r["matching"] = exact_paired_equivalence(
                correct[ref], correct[repo], margin=p["matching"]["margin"],
                alpha=p["matching"]["alpha"])
    primary = []
    for run in p["runs"]:
        repo, r = run["repo"], runs[run["repo"]]
        if not r["complete"]:
            bad = [c for c, v in r["checkpoints"].items() if v["status"] != "ok"]
            reason = "incomplete: checkpoint(s) " + ", ".join(
                f"{c} ({r['checkpoints'][c].get('error')})" for c in bad)
        elif not r["inclusion"]["above_chance"]:
            reason = "fails inclusion (IOI accuracy not above chance on the matching set)"
        elif repo != ref and not ref_ok:
            reason = "reference run not eligible; matching not evaluable"
        elif repo != ref and not r["matching"]["equivalent"]:
            reason = "not matched to the reference (exact discordance bound >= 0.02)"
        else:
            reason = None
        r["primary"] = reason is None
        r["exclusion_reason"] = reason
        if reason is None:
            primary.append(repo)
    return primary, ref_ok


# ── distances and run-level inference (AMENDMENT3.md §7) ─────────────────────────

def pairwise(p, tensors, repos):
    d = distance_function(p["metric"])
    within = {r: {f"{s}-{t}": d(tensors[(r, s)], tensors[(r, t)]) for s, t in within_pairs(p)}
              for r in repos}
    cross = {f"{a}|{b}": d(tensors[(a, FINAL)], tensors[(b, FINAL)])
             for a, b in combinations(repos, 2)}
    return within, cross


def run_level(p, tensors, repos, pw):
    """Validated Gate 5 procedure: plug-in D*, two-way bootstrap, B = 200."""
    n_prompts = tensors[(repos[0], FINAL)].shape[0]
    counts = np.random.default_rng(p["seeds"]["prompt_resample"]).multinomial(
        n_prompts, np.full(n_prompts, 1.0 / n_prompts), size=p["n_boot"])
    pairs = within_pairs(p)
    r_n = len(repos)
    within = np.array([[pw[0][r][f"{s}-{t}"] for s, t in pairs] for r in repos])
    cross = np.zeros((r_n, r_n))
    within_b = np.zeros((p["n_boot"], r_n, len(pairs)))
    cross_b = np.zeros((p["n_boot"], r_n, r_n))
    for i, r in enumerate(repos):
        for k, (s, t) in enumerate(pairs):
            within_b[:, i, k] = orbit_distance_weighted(tensors[(r, s)], tensors[(r, t)], counts)
    for i, j in combinations(range(r_n), 2):
        a, b = repos[i], repos[j]
        cross[i, j] = cross[j, i] = pw[1][f"{a}|{b}"]
        cross_b[:, i, j] = cross_b[:, j, i] = orbit_distance_weighted(
            tensors[(a, FINAL)], tensors[(b, FINAL)], counts)
    boot = lineage.two_way_bootstrap(within_b, cross_b, within, cross,
                                     seed=p["seeds"]["run_resample"], alpha=p["alpha"])
    return boot, within, cross


def h1_result(p, tensors, primary, ref_ok, runs, pw):
    base = {"decision_rule": "SUPPORTED_AT_RUN_LEVEL iff one-sided 95% bootstrap lower bound "
                             "of Delta > 0 (Gate 5 procedure, B = 200)",
            "self_pairs": "excluded", "R": len(primary), "runs": list(primary)}
    if not runs[p["reference"]]["complete"]:
        return {**base, "status": "UNTESTED_UNRESOLVED",
                "reason": "reference run incomplete: H1 untestable"}, None
    if not ref_ok:
        return {**base, "status": "UNTESTED_UNRESOLVED",
                "reason": "reference run fails inclusion: H1 untestable"}, None
    if len(primary) < p["r_min"]:
        return {**base, "status": "UNTESTED_UNRESOLVED",
                "reason": f"fewer than r_min={p['r_min']} eligible runs ({len(primary)})"}, None
    boot, within, cross = run_level(p, tensors, primary, pw)
    status = "SUPPORTED_AT_RUN_LEVEL" if boot["reject_h0"] else "NOT_SUPPORTED"
    return {**base, "status": status, "reason": None, "delta": boot["delta"],
            "lower_95": boot["lower_95"], "n_boot": boot["n_boot"],
            "n_valid": boot["n_valid"]}, (within, cross)


def sensitivity(p, tensors, runs, primary, pw, h1_arrays):
    out: Dict[str, Any] = {"note": "Descriptive only; never changes the H1 decision."}
    complete = [r["repo"] for r in p["runs"] if runs[r["repo"]]["complete"]]
    for key, repos in (("S1_all_complete_runs", complete),
                       ("S2_primary_without_" + p["s2_exclude"].split("/")[-1],
                        [r for r in primary if r != p["s2_exclude"]])):
        if key.startswith("S2") and p["s2_exclude"] not in primary:
            out[key] = {"computed": False, "reason": "excluded run not in the primary set"}
        elif len(repos) < p["r_min"]:
            out[key] = {"computed": False, "reason": f"fewer than r_min={p['r_min']} runs"}
        else:
            boot, _, _ = run_level(p, tensors, repos, pw)
            out[key] = {"computed": True, "runs": repos, "delta": boot["delta"],
                        "lower_95_descriptive": boot["lower_95"], "n_valid": boot["n_valid"]}
    if h1_arrays is None:
        out["S3_crossfit_delta"] = {"computed": False, "reason": "H1 not tested"}
        out["J_jackknife"] = {"computed": False, "reason": "H1 not tested",
                              "role": "diagnostic only; never determines H1"}
        return out
    seed, k = p["seeds"]["s3_crossfit"], p["s3_n_splits"]
    cf = lambda a, b: crossfit_orbit_distance(a, b, n_splits=k, seed=seed)["crossfit"]  # noqa
    w = np.array([[cf(tensors[(r, s)], tensors[(r, t)]) for s, t in within_pairs(p)]
                  for r in primary])
    c = np.zeros((len(primary), len(primary)))
    for i, j in combinations(range(len(primary)), 2):
        c[i, j] = c[j, i] = cf(tensors[(primary[i], FINAL)], tensors[(primary[j], FINAL)])
    out["S3_crossfit_delta"] = {"computed": True, "delta": lineage.delta(w, c),
                                "note": "point estimate only"}
    out["J_jackknife"] = {"computed": True, "role": "diagnostic only; never determines H1",
                          **lineage.jackknife(*h1_arrays)}
    return out


# ── record ───────────────────────────────────────────────────────────────────────

_VOLATILE = {"started_utc", "finished_utc", "output_dir", "environment", "git"}


def strip_volatile(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in rec.items() if k not in _VOLATILE}


def _file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_confirmatory(p, expected_sha256, backend, out_dir) -> Dict[str, Any]:
    """Execute the locked protocol end to end and write an immutable-style record."""
    validate_protocol(p, expected_sha256)
    require_locked_commit(p["label"])
    out_dir = pathlib.Path(out_dir)
    if (out_dir / "record.json").exists():
        raise ProtocolError(f"{out_dir}/record.json exists; records are never overwritten")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prompts = load_prompt_sets(p)
    runs, tensors, correct = measure_all(p, backend, prompts, out_dir / "_cache")
    primary, ref_ok = eligibility(p, runs, correct)
    complete = [r["repo"] for r in p["runs"] if runs[r["repo"]]["complete"]]
    pw = pairwise(p, tensors, complete)
    h1, arrays = h1_result(p, tensors, primary, ref_ok, runs, pw)
    warnings = []
    if h1["status"] != "UNTESTED_UNRESOLVED" and 7 <= len(primary) <= 9:
        warnings.append("R in 7-9: calibration assumed between validated R = 6 and R = 10")
    rec = {
        "label": p["label"], "protocol": p, "protocol_sha256": protocol_hash(p),
        "runner_sha256": _file_sha256(pathlib.Path(__file__)), "git": git_state(),
        "lock_tag_at_head": LOCK_TAG in _tags_at_head(),
        "environment": environment_record(), "started_utc": started,
        "output_dir": str(out_dir),
        "prompt_sets": {k: {"sha256": v["sha256"], "n": v["n"], "path": v["path"]}
                        for k, v in p["prompts"].items()},
        "seeds": {**p["seeds"], "model_inference": "deterministic (no RNG)"},
        "runs": runs, "primary_runs": primary, "r_min": p["r_min"],
        "pairwise_d_star": {"note": "D* point estimates only; no pair-level confidence "
                                    "intervals (Gate 1 failed and is closed)",
                            "within": pw[0], "cross": pw[1]},
        "h1": h1,
        "bootstrap": {"method": p["inference"], "n_boot": p["n_boot"], "alpha": p["alpha"],
                      "prompt_seed": p["seeds"]["prompt_resample"],
                      "run_seed": p["seeds"]["run_resample"],
                      "inherited_from": "Gate 5 (validated)", "n_valid": h1.get("n_valid")},
        "sensitivity": sensitivity(p, tensors, runs, primary, pw, arrays),
        "warnings": warnings,
        "errors": [f"{repo}@{c}: {v.get('error')}" for repo, r in runs.items()
                   for c, v in r["checkpoints"].items() if v["status"] != "ok"],
        "claim_boundary": CLAIM_BOUNDARY, "gate1_note": GATE1_NOTE,
    }
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec = json.loads(canonical_json(rec))
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(rec, indent=2, sort_keys=True)
    (out_dir / "record.json").write_text(text)
    (out_dir / "record.sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "\n")
    return rec


# ── real backend (Hugging Face + TransformerLens) ────────────────────────────────

_LEGACY_BUFFERS = (".attention.bias", ".attention.masked_bias", ".rotary_emb.inv_freq")


def check_state_keys(missing: Sequence[str], unexpected: Sequence[str]) -> None:
    """Every model parameter must come from the pinned file; the only tolerated extra keys
    are legacy GPT-NeoX buffers that current transformers recomputes (and that
    ``from_pretrained`` also drops)."""
    if missing:
        raise ProtocolError(f"missing parameters in pinned weights file: {list(missing)[:5]}")
    bad = [k for k in unexpected if not k.endswith(_LEGACY_BUFFERS)]
    if bad:
        raise ProtocolError(f"unexpected keys in pinned weights file: {bad[:5]}")

class HubBackend:
    """Loads by pinned commit SHA only; never by branch name."""

    def __init__(self, free_disk: bool = False):
        self.free_disk = free_disk
        self._current: Tuple[str, str] | None = None

    def load(self, repo, sha, base_model, use_safetensors):
        """Explicit loading of exactly the pinned weights file.

        ``AutoModelForCausalLM.from_pretrained`` is deliberately not used: for repositories
        that ship only ``pytorch_model.bin`` it starts a background thread
        (``Thread-auto_conversion``) that downloads an unmerged SFconvertbot PR's
        ``model.safetensors``. That file is not loaded, but it is an uncontrolled network
        side effect (found in the pipeline-validation run, 2026-09-29).
        """
        import torch
        from huggingface_hub import hf_hub_download
        from transformer_lens import HookedTransformer
        from transformers import AutoConfig, AutoModelForCausalLM

        cfg = AutoConfig.from_pretrained(repo, revision=sha)
        fname = "model.safetensors" if use_safetensors else "pytorch_model.bin"
        path = hf_hub_download(repo, fname, revision=sha)
        if use_safetensors:
            from safetensors.torch import load_file

            state = load_file(path)
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
        hf = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
        res = hf.load_state_dict(state, strict=False)
        check_state_keys(res.missing_keys, res.unexpected_keys)
        hf.eval()
        resolved = cfg._commit_hash
        model = HookedTransformer.from_pretrained(base_model, hf_model=hf, device="cpu")
        model.eval()
        del hf, state
        self._current = (repo, sha)
        return model, resolved

    def weights_check(self, repo, sha, use_safetensors):
        from huggingface_hub import HfApi, hf_hub_download

        fname = "model.safetensors" if use_safetensors else "pytorch_model.bin"
        info = HfApi().model_info(repo, revision=sha, files_metadata=True)
        entry = {s.rfilename: s for s in info.siblings}.get(fname)
        if entry is None or entry.lfs is None:
            return {"file": fname, "expected_sha256": None, "local_sha256": None, "ok": False}
        local = pathlib.Path(hf_hub_download(repo, fname, revision=sha, local_files_only=True))
        h = hashlib.sha256()
        with open(local, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        return {"file": fname, "expected_sha256": entry.lfs.sha256,
                "local_sha256": h.hexdigest(), "ok": h.hexdigest() == entry.lfs.sha256}

    def tokenizer_single_token(self, model, names):
        vocab = hashlib.sha256(json.dumps(sorted(model.tokenizer.get_vocab().items()))
                               .encode()).hexdigest()
        if vocab != VOCAB_SHA256:
            return False
        return all(len(model.to_str_tokens(" " + n, prepend_bos=False)) == 1 for n in names)

    def correctness(self, model, items):
        from glassbox.v6 import measure

        return measure.evaluate_items(model, items)[0]

    def attributions(self, model, items):
        from glassbox.v6 import measure

        heads, mat = measure.head_attribution_matrix(model, items)
        n_l, n_h = model.cfg.n_layers, model.cfg.n_heads
        if heads != [(a, b) for a in range(n_l) for b in range(n_h)]:
            raise ProtocolError("unexpected head ordering from the attribution backend")
        return mat.reshape(len(items), n_l, n_h)

    def release(self, model):
        import gc

        del model
        gc.collect()
        if self.free_disk and self._current is not None:
            from huggingface_hub import scan_cache_dir

            repo, sha = self._current
            info = scan_cache_dir()
            revs = [rv.commit_hash for r in info.repos if r.repo_id == repo
                    for rv in r.revisions if rv.commit_hash == sha]
            if revs:
                info.delete_revisions(*revs).execute()


# ── CLI ──────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="V6 confirmatory runner (locked protocol).")
    ap.add_argument("--out", required=True, help="output directory (must not hold a record)")
    ap.add_argument("--pipeline-validation", action="store_true",
                    help="run the Pythia-70M software check; output is NOT confirmatory data")
    ap.add_argument("--free-disk", action="store_true",
                    help="delete each checkpoint's cached weights after measuring it")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.pipeline_validation:
        p, h = VALIDATION_PROTOCOL, EXPECTED_VALIDATION_SHA256
    else:
        p, h = PROTOCOL, EXPECTED_PROTOCOL_SHA256
    require_locked_commit(p["label"])
    rec = run_confirmatory(p, h, HubBackend(free_disk=args.free_disk), args.out)
    print(rec["label"])
    print("primary runs:", rec["primary_runs"])
    print("H1:", json.dumps(rec["h1"], sort_keys=True))
    print("record:", pathlib.Path(args.out) / "record.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
