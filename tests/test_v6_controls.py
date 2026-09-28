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


def test_above_chance() -> None:
    assert controls.above_chance([True] * 45 + [False] * 5)["above_chance"] is True
    r = controls.above_chance([True] * 17 + [False] * 23)
    assert r["above_chance"] is False and 0.0 <= r["p_value"] <= 1.0
