"""Tests for the locked V6 confirmatory runner (experiments/v6/confirmatory.py).

A fake backend stands in for model loading/measurement so every rule of Amendment 3 can be
exercised without downloads. Numbered comments map to the required test list.
"""
from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import pathlib

import numpy as np
import pytest

from glassbox.v6 import lineage, tasks
from glassbox.v6.identifiability import profile_orbit_distance

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cf", ROOT / "experiments/v6/confirmatory.py")
cf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cf)

L, H = 3, 4
SHA = "{:040x}"


# ── fixtures ────────────────────────────────────────────────────────────────────

def _write_prompts(tmp, name, items, ids_prefix):
    rows = []
    for i, it in enumerate(items):
        d = dataclasses.asdict(it)
        d["id"] = f"{ids_prefix}{i:04d}"
        rows.append(d)
    path = tmp / f"{name}.json"
    path.write_text(json.dumps({"items": rows, "sha256": cf.items_hash(rows), "n": len(rows)}))
    return {"path": str(path), "sha256": cf.items_hash(rows), "n": len(rows)}


def _protocol(tmp, n_runs=5):
    a = tasks.build_ioi(12, 0).items
    pa = {x.prompt for x in a}
    m = [x for x in tasks.build_ioi(220, 2).items if x.prompt not in pa]
    uniq, seen = [], set()
    for x in m:
        if x.prompt not in seen:
            seen.add(x.prompt)
            uniq.append(x)
    p = copy.deepcopy(cf.PROTOCOL)
    p["label"] = cf.TEST_LABEL
    p["runs"] = [{"repo": f"org/run{r}", "use_safetensors": r == 0,
                  "commits": {str(c): SHA.format(1000 * r + c) for c in p["checkpoints"]}}
                 for r in range(n_runs)]
    p["reference"] = "org/run0"
    p["prompts"] = {"attribution": _write_prompts(tmp, "attr", a, "A"),
                    "matching": _write_prompts(tmp, "match", uniq[:160], "M")}
    return p


class FakeBackend:
    """Run r has lineage structure base_r; checkpoints add small drift."""

    def __init__(self, fail_load=(), resolved=None, weights_bad=(), accuracy=None, seed=0):
        self.fail_load, self.resolved = set(fail_load), resolved or {}
        self.weights_bad, self.accuracy = set(weights_bad), accuracy or {}
        self.seed, self.loaded = seed, []

    def load(self, repo, sha, base_model, use_safetensors):
        self.loaded.append((repo, sha))
        if (repo, sha) in self.fail_load or repo in self.fail_load:
            raise OSError(f"simulated download failure for {repo}@{sha[:8]}")
        return {"repo": repo, "sha": sha}, self.resolved.get((repo, sha), sha)

    def weights_check(self, repo, sha, use_safetensors):
        ok = (repo, sha) not in self.weights_bad
        return {"file": "w", "expected_sha256": "e", "local_sha256": "e" if ok else "x", "ok": ok}

    def tokenizer_single_token(self, model, names):
        return True

    def correctness(self, model, items):
        acc = self.accuracy.get(model["repo"], 1.0)
        rng = np.random.default_rng(sum(map(ord, model["repo"])))
        return [bool(v) for v in rng.random(len(items)) < acc]

    def attributions(self, model, items):
        r = int(model["repo"].replace("org/run", ""))
        base = np.random.default_rng([self.seed, r]).normal(size=(len(items), L, H))
        drift = np.random.default_rng([self.seed, r, int(model["sha"], 16) % 997]).normal(
            size=(len(items), L, H))
        return base + 0.05 * drift

    def release(self, model):
        pass


def _run(tmp, p=None, backend=None, name="out"):
    p = p or _protocol(tmp)
    return cf.run_confirmatory(p, cf.protocol_hash(p), backend or FakeBackend(), tmp / name)


# ── 1-2 model pinning ─────────────────────────────────────────────────────────────

def test_01_wrong_model_sha_is_hard_failure(tmp_path) -> None:
    p = _protocol(tmp_path)
    sha = p["runs"][1]["commits"]["143000"]
    bad = FakeBackend(resolved={("org/run1", sha): SHA.format(999999)})
    with pytest.raises(cf.ProtocolError, match="commit mismatch"):
        _run(tmp_path, p, bad)


def test_01b_weights_hash_mismatch_is_hard_failure(tmp_path) -> None:
    p = _protocol(tmp_path)
    bad = FakeBackend(weights_bad={("org/run2", p["runs"][2]["commits"]["123000"])})
    with pytest.raises(cf.ProtocolError, match="weights"):
        _run(tmp_path, p, bad)


@pytest.mark.parametrize("rev", ["main", "step143000", "abc123", "refs/pr/1"])
def test_02_unpinned_or_moving_revision_is_hard_failure(tmp_path, rev) -> None:
    p = _protocol(tmp_path)
    p["runs"][3]["commits"]["143000"] = rev
    with pytest.raises(cf.ProtocolError, match="40-hex"):
        cf.validate_protocol(p, cf.protocol_hash(p))


# ── 3-5 prompt invariants ─────────────────────────────────────────────────────────

def test_03_wrong_prompt_count_is_hard_failure(tmp_path) -> None:
    p = _protocol(tmp_path)
    p["prompts"]["attribution"]["n"] += 1
    with pytest.raises(cf.ProtocolError, match="count"):
        cf.load_prompt_sets(p)


def test_03b_prompt_hash_mismatch_is_hard_failure(tmp_path) -> None:
    p = _protocol(tmp_path)
    p["prompts"]["matching"]["sha256"] = "0" * 64
    with pytest.raises(cf.ProtocolError, match="sha256"):
        cf.load_prompt_sets(p)


def _rewrite(entry, rows):
    path = pathlib.Path(entry["path"])
    path.write_text(json.dumps({"items": rows}))
    entry["sha256"], entry["n"] = cf.items_hash(rows), len(rows)


def test_04_duplicate_prompt_is_hard_failure(tmp_path) -> None:
    p = _protocol(tmp_path)
    rows = json.loads(pathlib.Path(p["prompts"]["matching"]["path"]).read_text())["items"]
    dup = dict(rows[0], id=f"M{len(rows):04d}")
    _rewrite(p["prompts"]["matching"], rows + [dup])
    with pytest.raises(cf.ProtocolError, match="duplicate"):
        cf.load_prompt_sets(p)


def test_05_prompt_overlap_is_hard_failure(tmp_path) -> None:
    p = _protocol(tmp_path)
    a = json.loads(pathlib.Path(p["prompts"]["attribution"]["path"]).read_text())["items"]
    m = json.loads(pathlib.Path(p["prompts"]["matching"]["path"]).read_text())["items"]
    _rewrite(p["prompts"]["matching"], m + [dict(a[0], id=f"M{len(m):04d}")])
    with pytest.raises(cf.ProtocolError, match="overlap"):
        cf.load_prompt_sets(p)


def test_05b_prompt_ids_must_be_in_locked_order(tmp_path) -> None:
    p = _protocol(tmp_path)
    rows = json.loads(pathlib.Path(p["prompts"]["attribution"]["path"]).read_text())["items"]
    _rewrite(p["prompts"]["attribution"], rows[1:] + rows[:1])
    with pytest.raises(cf.ProtocolError, match="order"):
        cf.load_prompt_sets(p)


# ── 6 checkpoints ────────────────────────────────────────────────────────────────

def test_06_wrong_checkpoint_is_hard_failure(tmp_path) -> None:
    p = _protocol(tmp_path)
    p["checkpoints"] = [113000, 133000, 143000]
    with pytest.raises(cf.ProtocolError, match="checkpoint"):
        cf.validate_protocol(p, cf.protocol_hash(p))
    q = _protocol(tmp_path)
    q["runs"][2]["commits"].pop("133000")
    with pytest.raises(cf.ProtocolError, match="checkpoint"):
        cf.validate_protocol(q, cf.protocol_hash(q))


def test_06b_locked_protocol_hash_is_enforced() -> None:
    cf.validate_protocol(cf.PROTOCOL, cf.EXPECTED_PROTOCOL_SHA256)
    p = copy.deepcopy(cf.PROTOCOL)
    p["seeds"]["run_resample"] += 1
    with pytest.raises(cf.ProtocolError, match="protocol hash"):
        cf.validate_protocol(p, cf.EXPECTED_PROTOCOL_SHA256)


# ── 7-9 exclusions and untestable outcomes ───────────────────────────────────────

def test_07_missing_run_is_excluded_with_reason(tmp_path) -> None:
    rec = _run(tmp_path, backend=FakeBackend(fail_load={"org/run3"}))
    r3 = rec["runs"]["org/run3"]
    assert r3["complete"] is False and r3["primary"] is False
    assert "incomplete" in r3["exclusion_reason"] and "simulated download failure" in json.dumps(r3)
    assert "org/run3" not in rec["primary_runs"]
    assert rec["h1"]["status"] in {"SUPPORTED_AT_RUN_LEVEL", "NOT_SUPPORTED"}


def test_08_reference_failure_makes_h1_untestable(tmp_path) -> None:
    rec = _run(tmp_path, backend=FakeBackend(fail_load={"org/run0"}))
    assert rec["h1"]["status"] == "UNTESTED_UNRESOLVED"
    assert "reference" in rec["h1"]["reason"]
    rec2 = _run(tmp_path, backend=FakeBackend(accuracy={"org/run0": 0.3}), name="o2")
    assert rec2["h1"]["status"] == "UNTESTED_UNRESOLVED"
    # must be untestable *because of the reference rule*, not via another path
    assert rec2["h1"]["reason"] == "reference run fails inclusion: H1 untestable"
    assert rec["h1"]["reason"] == "reference run incomplete: H1 untestable"


def test_09_fewer_than_r_min_makes_h1_untestable(tmp_path) -> None:
    rec = _run(tmp_path, backend=FakeBackend(fail_load={"org/run1", "org/run2"}))
    assert len(rec["primary_runs"]) == 3 < cf.PROTOCOL["r_min"] == 4
    assert rec["h1"]["status"] == "UNTESTED_UNRESOLVED" and "r_min" in rec["h1"]["reason"]


def test_09b_unmatched_run_is_excluded_not_replaced(tmp_path) -> None:
    rec = _run(tmp_path, backend=FakeBackend(accuracy={"org/run4": 0.6}))
    r4 = rec["runs"]["org/run4"]
    assert r4["matching"]["equivalent"] is False and r4["primary"] is False
    assert rec["primary_runs"] == ["org/run0", "org/run1", "org/run2", "org/run3"]


# ── 10-12 method locks ────────────────────────────────────────────────────────────

def test_10_positional_dm_is_hard_failure(tmp_path) -> None:
    p = _protocol(tmp_path)
    p["metric"] = "operational_mechanistic_distance"
    with pytest.raises(cf.ProtocolError, match="metric"):
        cf.validate_protocol(p, cf.protocol_hash(p))
    with pytest.raises(cf.ProtocolError):
        cf.distance_function("operational_mechanistic_distance")
    src = (ROOT / "experiments/v6/confirmatory.py").read_text()
    assert "operational_mechanistic_distance(" not in src


def test_11_b_not_200_is_hard_failure(tmp_path) -> None:
    for b in (199, 2000):
        p = _protocol(tmp_path)
        p["n_boot"] = b
        with pytest.raises(cf.ProtocolError, match="n_boot"):
            cf.validate_protocol(p, cf.protocol_hash(p))


def test_12_alternate_inference_is_hard_failure(tmp_path) -> None:
    for alt in ("jackknife", "percentile_bootstrap"):
        p = _protocol(tmp_path)
        p["inference"] = alt
        with pytest.raises(cf.ProtocolError, match="inference"):
            cf.validate_protocol(p, cf.protocol_hash(p))


def test_12b_cli_exposes_no_method_options() -> None:
    opts = {a.dest for a in cf.build_parser()._actions}
    assert opts <= {"help", "out", "pipeline_validation", "free_disk"}


# ── 13-15 estimator behaviour and reproducibility ───────────────────────────────

def test_13_delta_matches_lineage_and_self_pairs_excluded(tmp_path) -> None:
    rec = _run(tmp_path)
    prim = rec["primary_runs"]
    d = rec["pairwise_d_star"]
    within = np.array([[d["within"][r][f"{s}-{t}"] for s, t in cf.within_pairs(cf.PROTOCOL)]
                       for r in prim])
    cross = np.zeros((len(prim), len(prim)))
    for i, a in enumerate(prim):
        for j, b in enumerate(prim):
            if i != j:
                cross[i, j] = d["cross"][f"{a}|{b}"] if f"{a}|{b}" in d["cross"] else \
                    d["cross"][f"{b}|{a}"]
    assert rec["h1"]["delta"] == pytest.approx(lineage.delta(within, cross))
    assert all(a != b for k in d["cross"] for a, b in [k.split("|")])
    assert rec["h1"]["self_pairs"] == "excluded"


def test_13b_d_star_is_profile_orbit_distance(tmp_path) -> None:
    fb = FakeBackend()
    p = _protocol(tmp_path)
    rec = cf.run_confirmatory(p, cf.protocol_hash(p), fb, tmp_path / "o")
    items = cf.load_prompt_sets(p)["attribution"]
    a = fb.attributions({"repo": "org/run0", "sha": p["runs"][0]["commits"]["143000"]}, items)
    b = fb.attributions({"repo": "org/run1", "sha": p["runs"][1]["commits"]["143000"]}, items)
    assert rec["pairwise_d_star"]["cross"]["org/run0|org/run1"] == pytest.approx(
        profile_orbit_distance(a, b)["value"])


def test_14_fixed_seeds_reproduce_bootstrap(tmp_path) -> None:
    from glassbox.v6.identifiability import orbit_distance_weighted

    p = _protocol(tmp_path)
    fb = FakeBackend()
    r1 = cf.run_confirmatory(p, cf.protocol_hash(p), fb, tmp_path / "a")
    r2 = _run(tmp_path, p, name="b")
    assert r1["h1"]["lower_95"] == r2["h1"]["lower_95"]
    # independent recomputation with the pinned seeds (catches a changed seed)
    items = cf.load_prompt_sets(p)["attribution"]
    runs = r1["primary_runs"]
    ten = {(r, c): fb.attributions({"repo": r, "sha": next(x for x in p["runs"]
                                    if x["repo"] == r)["commits"][str(c)]}, items)
           for r in runs for c in cf.CHECKPOINTS}
    n = len(items)
    counts = np.random.default_rng(20260929).multinomial(n, np.full(n, 1 / n), size=200)
    pairs = cf.within_pairs(p)
    wb = np.stack([[orbit_distance_weighted(ten[(r, s)], ten[(r, t)], counts)
                    for s, t in pairs] for r in runs]).transpose(2, 0, 1)
    cb = np.zeros((200, len(runs), len(runs)))
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            cb[:, i, j] = cb[:, j, i] = orbit_distance_weighted(
                ten[(runs[i], 143000)], ten[(runs[j], 143000)], counts)
    w = np.array([[profile_orbit_distance(ten[(r, s)], ten[(r, t)])["value"]
                   for s, t in pairs] for r in runs])
    c = np.zeros((len(runs), len(runs)))
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            c[i, j] = c[j, i] = profile_orbit_distance(ten[(runs[i], 143000)],
                                                       ten[(runs[j], 143000)])["value"]
    ref = lineage.two_way_bootstrap(wb, cb, w, c, seed=20260930)
    assert r1["h1"]["lower_95"] == pytest.approx(ref["lower_95"], abs=1e-12)
    assert r1["h1"]["delta"] == pytest.approx(ref["delta"], abs=1e-12)
    assert r1["seeds"] == {"prompt_resample": 20260929, "run_resample": 20260930,
                           "s3_crossfit": 20260931, "model_inference": "deterministic (no RNG)"}


def test_15_identical_input_identical_record(tmp_path) -> None:
    r1, r2 = _run(tmp_path, name="a"), _run(tmp_path, name="b")
    assert cf.strip_volatile(r1) == cf.strip_volatile(r2)


def test_15b_record_is_not_overwritten(tmp_path) -> None:
    p = _protocol(tmp_path)
    _run(tmp_path, p, name="same")
    with pytest.raises(cf.ProtocolError, match="exists"):
        _run(tmp_path, p, name="same")


# ── 16 provenance ───────────────────────────────────────────────────────────────

def test_16_record_contains_all_provenance_fields(tmp_path) -> None:
    rec = _run(tmp_path)
    for key in ("label", "protocol", "protocol_sha256", "runner_sha256", "git", "environment",
                "started_utc", "finished_utc", "prompt_sets", "seeds", "runs", "primary_runs",
                "r_min", "pairwise_d_star", "h1", "bootstrap", "sensitivity", "warnings",
                "errors", "claim_boundary", "gate1_note"):
        assert key in rec, key
    ck = rec["runs"]["org/run1"]["checkpoints"]["143000"]
    for key in ("pinned_commit", "resolved_commit", "weights", "status"):
        assert key in ck, key
    assert rec["bootstrap"] == {"method": "two_way_bootstrap", "n_boot": 200, "alpha": 0.05,
                                "prompt_seed": 20260929, "run_seed": 20260930,
                                "inherited_from": "Gate 5 (validated)",
                                "n_valid": rec["h1"]["n_valid"]}
    stored = json.loads((tmp_path / "out" / "record.json").read_text())
    assert stored["h1"] == json.loads(json.dumps(rec["h1"]))
    assert (tmp_path / "out" / "record.sha256").exists()


def test_16b_pairwise_values_are_labelled_point_estimates(tmp_path) -> None:
    rec = _run(tmp_path)
    assert "point estimates only" in rec["pairwise_d_star"]["note"]
    assert "confidence interval" not in json.dumps(rec["pairwise_d_star"]["cross"])
    assert rec["sensitivity"]["J_jackknife"]["role"].startswith("diagnostic only")


# ── resumable cache (crash safety; must not change results) ──────────────────────

class _Crash(Exception):
    pass


class CrashingBackend(FakeBackend):
    """Raises a non-recoverable interrupt after `after` successful loads."""

    def __init__(self, after, **kw):
        super().__init__(**kw)
        self.after = after

    def load(self, repo, sha, base_model, use_safetensors):
        if len(self.loaded) >= self.after:
            raise KeyboardInterrupt  # simulates the process being killed
        return super().load(repo, sha, base_model, use_safetensors)


def test_17_resume_after_interrupt_gives_identical_result(tmp_path) -> None:
    p = _protocol(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        cf.run_confirmatory(p, cf.protocol_hash(p), CrashingBackend(after=7), tmp_path / "r")
    assert not (tmp_path / "r" / "record.json").exists()
    resumed = cf.run_confirmatory(p, cf.protocol_hash(p), FakeBackend(), tmp_path / "r")
    fresh = cf.run_confirmatory(p, cf.protocol_hash(p), FakeBackend(), tmp_path / "f")
    assert resumed["h1"] == fresh["h1"] and resumed["pairwise_d_star"] == fresh["pairwise_d_star"]
    flags = [c["from_cache"] for r in resumed["runs"].values() for c in r["checkpoints"].values()]
    assert sum(flags) == 7 and not any(c["from_cache"] for r in fresh["runs"].values()
                                       for c in r["checkpoints"].values())


def test_18_cache_from_other_protocol_is_not_reused(tmp_path) -> None:
    p = _protocol(tmp_path)
    cf.run_confirmatory(p, cf.protocol_hash(p), FakeBackend(), tmp_path / "x")
    (tmp_path / "x" / "record.json").unlink()
    q = copy.deepcopy(p)
    q["runs"][4]["commits"]["143000"] = SHA.format(424242)
    fb = FakeBackend()
    rec = cf.run_confirmatory(q, cf.protocol_hash(q), fb, tmp_path / "x")
    assert not any(c["from_cache"] for r in rec["runs"].values() for c in r["checkpoints"].values())


# ── confirmatory runs require the locked, clean commit ───────────────────────────

def test_19_confirmatory_label_requires_clean_locked_commit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cf, "git_state", lambda: {"sha": "a" * 40, "dirty": True})
    monkeypatch.setattr(cf, "_tags_at_head", lambda: ["v6-amendment3-lock"])
    with pytest.raises(cf.ProtocolError, match="clean"):
        cf.require_locked_commit(cf.CONFIRMATORY_LABEL)
    monkeypatch.setattr(cf, "git_state", lambda: {"sha": "a" * 40, "dirty": False})
    monkeypatch.setattr(cf, "_tags_at_head", lambda: [])
    with pytest.raises(cf.ProtocolError, match="v6-amendment3-lock"):
        cf.require_locked_commit(cf.CONFIRMATORY_LABEL)
    monkeypatch.setattr(cf, "_tags_at_head", lambda: ["v6-amendment3-lock"])
    cf.require_locked_commit(cf.CONFIRMATORY_LABEL)  # passes
    monkeypatch.setattr(cf, "git_state", lambda: {"sha": None, "dirty": None})
    cf.require_locked_commit(cf.VALIDATION_LABEL)  # validation runs are not gated


def test_19b_main_enforces_lock_before_loading_anything(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(cf, "require_locked_commit",
                        lambda label: (_ for _ in ()).throw(cf.ProtocolError("not locked")))
    monkeypatch.setattr(cf, "HubBackend", lambda **k: calls.append(k))
    with pytest.raises(cf.ProtocolError, match="not locked"):
        cf.main(["--out", str(tmp_path / "x")])
    assert calls == []


# ── explicit weight loading (no transformers auto-conversion side effects) ───────

def test_20_state_key_rules() -> None:
    ok_extra = ["gpt_neox.layers.0.attention.bias", "gpt_neox.layers.3.attention.masked_bias",
                "gpt_neox.layers.5.attention.rotary_emb.inv_freq"]
    cf.check_state_keys([], ok_extra)  # legacy buffers only: fine
    with pytest.raises(cf.ProtocolError, match="missing"):
        cf.check_state_keys(["gpt_neox.layers.0.attention.dense.weight"], [])
    with pytest.raises(cf.ProtocolError, match="unexpected"):
        cf.check_state_keys([], ["gpt_neox.layers.0.mlp.extra.weight"])


def test_20b_hub_backend_never_calls_from_pretrained() -> None:
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(cf.HubBackend.load)))
    calls = {n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    owners = {(n.func.value.id, n.func.attr) for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and isinstance(n.func.value, ast.Name)}
    assert ("AutoModelForCausalLM", "from_pretrained") not in owners
    assert "from_config" in calls and "load_state_dict" in calls
