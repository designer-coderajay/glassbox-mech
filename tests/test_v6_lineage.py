"""Run-level (lineage) inference for Delta: training runs are the unit."""
from __future__ import annotations

import numpy as np
import pytest

from glassbox.v6 import lineage


def test_delta_by_hand() -> None:
    within = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    cross = np.array([[0, 5, 7], [5, 0, 9], [7, 9, 0]], dtype=float)
    # cross mean over r<r': (5+7+9)/3 = 7; within mean = 2
    assert lineage.delta(within, cross) == pytest.approx(5.0)


def test_delta_ignores_diagonal_and_requires_symmetry() -> None:
    within = np.ones((3, 3))
    cross = np.array([[99, 2, 2], [2, 99, 2], [2, 2, 99]], dtype=float)
    assert lineage.delta(within, cross) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        lineage.delta(within, np.triu(cross))


def test_resampled_delta_drops_self_pairs_and_uses_multiplicity() -> None:
    within = np.array([[1.0] * 3, [2.0] * 3, [3.0] * 3])
    cross = np.array([[0, 5, 7], [5, 0, 9], [7, 9, 0]], dtype=float)
    # runs (0, 0, 2): within mean = (1+1+3)/3; cross pairs with distinct runs: (0,2),(0,2)
    got = lineage.delta_for_resample(within, cross, np.array([0, 0, 2]))
    assert got == pytest.approx(7.0 - 5.0 / 3.0)


def test_resample_with_single_distinct_run_is_nan() -> None:
    within, cross = np.ones((3, 3)), np.ones((3, 3)) - np.eye(3)
    assert np.isnan(lineage.delta_for_resample(within, cross, np.array([1, 1, 1])))


def test_jackknife_matches_manual() -> None:
    rng = np.random.default_rng(0)
    within = rng.random((5, 3))
    c = rng.random((5, 5))
    cross = (c + c.T) / 2
    np.fill_diagonal(cross, 0)
    loo = [lineage.delta(np.delete(within, r, 0), np.delete(np.delete(cross, r, 0), r, 1))
           for r in range(5)]
    se = np.sqrt(4 / 5 * np.sum((np.array(loo) - np.mean(loo)) ** 2))
    assert lineage.jackknife(within, cross)["se"] == pytest.approx(se)


def test_two_way_bootstrap_shapes_and_decision() -> None:
    rng = np.random.default_rng(1)
    r, b = 6, 50
    within = rng.random((b, r, 3)) * 0.1
    c = 1 + rng.random((b, r, r))
    cross = (c + c.transpose(0, 2, 1)) / 2
    res = lineage.two_way_bootstrap(within, cross, point_within=within.mean(0),
                                    point_cross=cross.mean(0), seed=0)
    assert res["delta"] > 0 and res["lower_95"] > 0 and res["reject_h0"]
    assert res["n_valid"] <= b


def test_two_way_bootstrap_is_invariant_to_run_order() -> None:
    rng = np.random.default_rng(2)
    r, b = 5, 40
    within = rng.random((b, r, 3))
    c = rng.random((b, r, r))
    cross = (c + c.transpose(0, 2, 1)) / 2
    perm = rng.permutation(r)
    a = lineage.delta(within.mean(0), cross.mean(0))
    z = lineage.delta(within.mean(0)[perm], cross.mean(0)[np.ix_(perm, perm)])
    assert a == pytest.approx(z)
