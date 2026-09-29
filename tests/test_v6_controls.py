"""V6 Control B (resampling null) and the model-inclusion check."""
from __future__ import annotations

import numpy as np
import pytest

from glassbox.v6 import controls


def _heads(n):
    return [(i // 4, i % 4) for i in range(n)]


def test_split_half_null_is_deterministic() -> None:
    m = np.random.default_rng(0).normal(size=(40, 12))
    a = controls.split_half_null(m, _heads(12), n_splits=50, seed=1)
    b = controls.split_half_null(m, _heads(12), n_splits=50, seed=1)
    assert a == b and len(a["values"]) == 50


def test_split_halves_are_disjoint_and_cover_items() -> None:
    for s1, s2 in controls.split_indices(11, n_splits=20, seed=0):
        assert not set(s1) & set(s2)
        assert len(s1) == len(s2) == 5


def test_null_is_small_when_mechanism_is_stable() -> None:
    # Positive structure + small per-item noise -> halves agree -> low D_M.
    rng = np.random.default_rng(2)
    base = np.linspace(-1, 1, 24)
    m = base + rng.normal(0, 0.05, size=(60, 24))
    null = controls.split_half_null(m, _heads(24), n_splits=100, seed=0)
    assert null["p95"] < 0.1


def test_null_is_near_one_when_attribution_is_pure_noise() -> None:
    m = np.random.default_rng(3).normal(size=(60, 24))
    null = controls.split_half_null(m, _heads(24), n_splits=200, seed=0)
    assert 0.8 < float(np.median(null["values"])) < 1.2


def test_divergence_rule() -> None:
    assert controls.is_divergent(0.5, [0.1, 0.2]) is True
    assert controls.is_divergent(0.15, [0.1, 0.2]) is False
    assert controls.is_divergent(float("nan"), [0.1, 0.2]) is None


def test_split_half_rejects_too_few_items() -> None:
    with pytest.raises(ValueError):
        controls.split_half_null(np.zeros((3, 4)), _heads(4), n_splits=5, seed=0)


def _layered(n_layers, n_heads, rng):
    heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    return heads, rng.normal(size=n_layers * n_heads) * np.repeat(
        np.linspace(0.1, 3, n_layers), n_heads)


def test_relabelling_preserves_each_layer_multiset() -> None:
    rng = np.random.default_rng(0)
    heads, v = _layered(4, 8, rng)
    p = controls.relabel_within_layers(v, heads, rng)
    for layer in range(4):
        idx = [i for i, (l, _) in enumerate(heads) if l == layer]
        assert sorted(v[idx]) == sorted(p[idx])


def test_head_indexed_dm_cannot_distinguish_a_relabelled_model() -> None:
    # Regression (2026-09-29 seed audit): shuffling one model's heads within layers is a
    # function-preserving relabelling, yet the primary D_M reports ~1 for it. So the
    # primary D_M is only meaningful when models share head correspondence.
    rng = np.random.default_rng(1)
    heads, v = _layered(24, 16, rng)
    null = controls.relabelling_null(v, heads, n=200, seed=0)
    assert null["p50"] > 0.6


def test_sorted_within_layer_dm_is_zero_for_relabelled_self() -> None:
    rng = np.random.default_rng(2)
    heads, v = _layered(6, 8, rng)
    p = controls.relabel_within_layers(v, heads, rng)
    assert controls.sorted_within_layer_dm(v, p, heads) == pytest.approx(0.0)
    w = rng.normal(size=v.size)
    assert controls.sorted_within_layer_dm(v, w, heads) > 0.0


def test_seed4_score_fails_inclusion_regression() -> None:
    # pythia-410m-seed4@143000 scored 84/200 on IOI (below chance); the pre-drafted
    # inclusion rule (PREREGISTRATION §2) must reject it.
    r = controls.above_chance([True] * 84 + [False] * 116)
    assert r["above_chance"] is False and r["p_value"] > 0.9


def test_above_chance() -> None:
    assert controls.above_chance([True] * 45 + [False] * 5)["above_chance"] is True
    r = controls.above_chance([True] * 17 + [False] * 23)
    assert r["above_chance"] is False and 0.0 <= r["p_value"] <= 1.0
