"""V6 multi-model pilot runner: each model measured once, all pairs computed."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from glassbox.v6 import pilot

BASES = {"m0": 0, "m1": 0, "m2": 1, "m3": 1}  # m0~m1 share a pattern, m2~m3 another


def _fake_model(name: str, ckpt, device: str, d_vocab: int = 12):
    return SimpleNamespace(name=name, cfg=SimpleNamespace(d_vocab=d_vocab),
                           tokenizer=SimpleNamespace(name_or_path="fake-tok"))


def _fake_measure(model, ds):
    n = len(ds.items)
    rng = np.random.default_rng(sum(map(ord, model.name)))  # stable across processes
    base = np.random.default_rng(BASES[model.name]).normal(size=24)
    per_item = base + rng.normal(0, 0.05, size=(n, 24))
    heads = [(i // 4, i % 4) for i in range(24)]
    probes = np.random.default_rng(BASES[model.name]).dirichlet(np.ones(12), 3 * n)
    return {"correct": [True] * n, "ld": [1.0] * n, "probes": probes,
            "attr": dict(zip(heads, per_item.mean(0).tolist())), "heads": heads,
            "per_item": per_item, "seconds": 0.0}


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(pilot, "_load", _fake_model)
    monkeypatch.setattr(pilot, "_predicate", lambda model: (lambda s: True))
    monkeypatch.setattr(pilot, "_measure_model", _fake_measure)
    monkeypatch.setattr(pilot, "_provenance", lambda spec: {"repo": spec, "commit": "x"})


def test_parse_model_spec() -> None:
    assert pilot.parse_model_spec("pythia-410m@143000") == ("pythia-410m", 143000)
    assert pilot.parse_model_spec("gpt2") == ("gpt2", None)
    with pytest.raises(ValueError):
        pilot.parse_model_spec("pythia-410m@final")


def test_pilot_computes_all_pairs_and_matrices(fakes, tmp_path) -> None:
    cfg = pilot.PilotConfig(models=["m0", "m1", "m2", "m3"], n_prompts=40, k=4,
                            n_splits=50)
    rec = pilot.run_pilot(cfg, tmp_path)
    assert len(rec["pairs"]) == 6
    dm = np.array(rec["matrices"]["D_M"])
    assert dm.shape == (4, 4) and np.allclose(dm, dm.T) and np.allclose(np.diag(dm), 0)
    by = {(p["a"], p["b"]): p for p in rec["pairs"]}
    assert by[("m0", "m1")]["divergent"] is False  # same pattern: within noise
    assert by[("m0", "m2")]["divergent"] is True  # different pattern
    assert by[("m0", "m2")]["D_M"] > by[("m0", "m1")]["D_M"]
    assert "A" in rec["models"][0]["controls"] and "A" not in rec["models"][1]["controls"]
    assert rec["models"][0]["controls"]["A"]["bitwise_identical"] is True


def test_pilot_writes_artifacts_with_unresolved_hypotheses(fakes, tmp_path) -> None:
    pilot.run_pilot(pilot.PilotConfig(models=["m0", "m2"], n_prompts=20, k=4,
                                      n_splits=20), tmp_path)
    fnd = json.loads((tmp_path / "finding.json").read_text())
    assert fnd["label"] == "pilot"
    assert all(h["status"] == "UNRESOLVED" for h in fnd["hypotheses"])
    rec = json.loads((tmp_path / "record.json").read_text())
    assert rec["dataset"]["items_hash"] and rec["metrics_version"]
    assert all(m["hub"]["commit"] == "x" for m in rec["models"])  # provenance recorded


def test_pilot_resumes_from_cache_after_failure(fakes, monkeypatch, tmp_path) -> None:
    # Regression (2026-09-29): a download failure at model 3 lost models 1-2.
    calls = []

    def counting(model, ds):
        calls.append(model.name)
        return _fake_measure(model, ds)

    def flaky(name, ckpt, device):
        if name == "m2":
            raise OSError("simulated download failure")
        return _fake_model(name, ckpt, device)

    cfg = pilot.PilotConfig(models=["m0", "m1", "m2"], n_prompts=20, k=4, n_splits=20)
    monkeypatch.setattr(pilot, "_measure_model", counting)
    monkeypatch.setattr(pilot, "_load", flaky)
    with pytest.raises(OSError):
        pilot.run_pilot(cfg, tmp_path)
    assert calls == ["m0", "m0", "m1"]  # m0 measured twice (Control A)

    calls.clear()
    monkeypatch.setattr(pilot, "_load", _fake_model)
    rec = pilot.run_pilot(cfg, tmp_path)
    assert calls == ["m2"]  # m0 (incl. Control A) and m1 came from the cache
    assert [m["from_cache"] for m in rec["models"]] == [True, True, False]
    assert (tmp_path / "_cache" / ".gitignore").read_text().strip() == "*"


def test_pilot_cache_is_invalidated_by_different_data(fakes, monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(pilot, "_measure_model",
                        lambda m, ds: calls.append(m.name) or _fake_measure(m, ds))
    pilot.run_pilot(pilot.PilotConfig(models=["m0", "m1"], n_prompts=20, k=4,
                                      n_splits=20), tmp_path)
    calls.clear()
    pilot.run_pilot(pilot.PilotConfig(models=["m0", "m1"], n_prompts=20, seed=1, k=4,
                                      n_splits=20), tmp_path)
    assert calls == ["m0", "m0", "m1"]  # new seed -> new dataset -> nothing reused


def test_cached_results_equal_fresh_results(fakes, tmp_path) -> None:
    cfg = pilot.PilotConfig(models=["m0", "m2"], n_prompts=20, k=4, n_splits=20)
    fresh = pilot.run_pilot(cfg, tmp_path)
    again = pilot.run_pilot(cfg, tmp_path)
    assert fresh["pairs"] == again["pairs"] and fresh["matrices"] == again["matrices"]


def test_pilot_rejects_bad_configs(fakes, monkeypatch, tmp_path) -> None:
    with pytest.raises(ValueError):
        pilot.run_pilot(pilot.PilotConfig(models=["m0"]), tmp_path)
    with pytest.raises(ValueError):
        pilot.run_pilot(pilot.PilotConfig(models=["m0", "m1"], label="confirmatory"),
                        tmp_path)
    monkeypatch.setattr(pilot, "_load", lambda n, c, d: _fake_model(
        n, c, d, d_vocab=12 if n == "m0" else 13))
    with pytest.raises(ValueError, match="vocabulary"):
        pilot.run_pilot(pilot.PilotConfig(models=["m0", "m1"], n_prompts=10), tmp_path)
