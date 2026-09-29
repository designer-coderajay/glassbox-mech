"""V6 distances: unit, synthetic, null, positive-control and reproducibility tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from glassbox.v6 import distances as d


def _attr(values):
    return {(i // 4, i % 4): float(v) for i, v in enumerate(values)}


# ---- D_P: paired TOST ---------------------------------------------------------------

def test_tost_zero_discordance_uses_exact_bound_not_fake_p() -> None:
    # Regression (2026-09-28 audit): SD = 0 used to report p_tost = 0.0.
    r = d.paired_tost([0.0] * 200, margin=0.02)
    assert r["p_tost"] is None and r["method"] == "exact_discordance_bound"
    assert r["discordance_upper_bound"] == pytest.approx(1 - 0.05 ** (1 / 200))
    assert r["equivalent"]  # bound 1.49pp < 2pp


def test_tost_zero_discordance_small_n_is_not_equivalent() -> None:
    # 0/50 discordant only bounds the gap at ~5.8pp, which cannot establish +-2pp.
    r = d.paired_tost([0.0] * 50, margin=0.02)
    assert not r["equivalent"] and r["discordance_upper_bound"] > 0.05


def test_tost_all_discordant_is_not_equivalent() -> None:
    r = d.paired_tost([1.0] * 200, margin=0.02)
    assert not r["equivalent"] and r["discordance_upper_bound"] == 1.0


def test_tost_constant_continuous_diffs_are_deterministic() -> None:
    r = d.paired_tost([0.005] * 10, margin=0.02)
    assert r["method"] == "degenerate_constant" and r["equivalent"]
    assert not d.paired_tost([0.5] * 10, margin=0.02)["equivalent"]


def test_tost_regular_case_reports_t_method() -> None:
    assert d.paired_tost([0.01, -0.02, 0.0, 0.01], margin=0.05)["method"] == "paired_t_tost"


def _pair(n, a_only, b_only):
    a = [1] * n
    b = [1] * n
    for i in range(a_only):
        b[i] = 0
    for i in range(a_only, a_only + b_only):
        a[i] = 0
    return a, b


def test_t_tost_is_not_monotone_in_discordance_regression() -> None:
    # Regression (2026-09-29 seed audit), using the observed discordance patterns:
    # 2 discordant (2 A-only) is NOT matched, 3 discordant (1 A-only, 2 B-only) IS.
    two = d.performance_distance(*_pair(200, 2, 0), [1.0] * 200, [1.0] * 200, margin=0.02)
    three = d.performance_distance(*_pair(200, 1, 2), [1.0] * 200, [1.0] * 200, margin=0.02)
    assert not two["matched"] and three["matched"]


def test_exact_equivalence_is_monotone_in_discordance() -> None:
    decisions = [d.exact_paired_equivalence(*_pair(1000, k, 0), margin=0.02)["equivalent"]
                 for k in range(0, 30)]
    assert decisions == sorted(decisions, reverse=True)  # True...True, False...False
    assert decisions[12] and not decisions[13]  # n=1000: matched iff <= 12 discordant


def test_exact_equivalence_matches_clopper_pearson() -> None:
    r = d.exact_paired_equivalence(*_pair(200, 1, 0), margin=0.02)
    assert r["n_discordant"] == 1 and not r["equivalent"]
    assert r["discordance_upper_bound"] == pytest.approx(0.0235, abs=1e-4)


def test_performance_distance_reports_discordant_counts() -> None:
    r = d.performance_distance([1, 1, 0, 1], [1, 0, 1, 1], [1.0] * 4, [1.0] * 4, margin=0.02)
    assert r["n_a_only"] == 1 and r["n_b_only"] == 1


def test_tost_large_gap_is_not_equivalent() -> None:
    rng = np.random.default_rng(0)
    diffs = rng.choice([0, 1], size=200, p=[0.7, 0.3])  # A better by ~30pp
    assert not d.paired_tost(diffs, margin=0.02)["equivalent"]


def test_tost_small_gap_large_n_is_equivalent() -> None:
    rng = np.random.default_rng(1)
    diffs = rng.normal(0.0, 0.05, size=2000)
    assert d.paired_tost(diffs, margin=0.02)["equivalent"]


def test_tost_small_n_is_not_equivalent_even_if_mean_zero() -> None:
    # Absence of evidence is not equivalence: noisy + tiny n must fail TOST.
    assert not d.paired_tost([1, -1, 0, 1, -1, 0], margin=0.02)["equivalent"]


def test_tost_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        d.paired_tost([0.0, 0.0], margin=0.0)
    with pytest.raises(ValueError):
        d.paired_tost([0.0], margin=0.02)


def test_tost_matches_hand_computation() -> None:
    from scipy import stats

    x = np.array([0.01, -0.02, 0.03, 0.0, 0.01, -0.01, 0.02, 0.0])
    r = d.paired_tost(x, margin=0.05)
    se = x.std(ddof=1) / math.sqrt(len(x))
    assert r["p_lower"] == pytest.approx(stats.t.sf((x.mean() + 0.05) / se, 7))
    assert r["p_upper"] == pytest.approx(stats.t.cdf((x.mean() - 0.05) / se, 7))


def test_performance_distance_shapes_and_values() -> None:
    r = d.performance_distance([1, 1, 0, 1], [1, 0, 0, 1], [2.0, 1.0, -1, 3],
                               [1.0, -1, -2, 2], margin=0.02)
    assert r["accuracy_diff"] == pytest.approx(0.25)
    assert r["mean_ld_diff"] == pytest.approx(1.25)
    with pytest.raises(ValueError):
        d.performance_distance([1], [1, 0], [0.0], [0.0, 1.0], margin=0.02)


# ---- D_M -----------------------------------------------------------------------------

def test_dm_identical_is_zero_and_reversed_is_two() -> None:
    a = _attr(range(12))
    assert d.operational_mechanistic_distance(a, a)["value"] == pytest.approx(0.0)
    rev = _attr(range(11, -1, -1))
    assert d.operational_mechanistic_distance(a, rev)["value"] == pytest.approx(2.0)


def test_dm_is_rank_based() -> None:
    # Positive control: a monotone transform keeps the attribution ranking -> D_M = 0.
    a = _attr(np.linspace(-1, 1, 16))
    b = {k: v ** 3 * 10 for k, v in a.items()}
    assert d.operational_mechanistic_distance(a, b)["value"] == pytest.approx(0.0)


def test_dm_null_independent_vectors_near_one() -> None:
    rng = np.random.default_rng(0)
    vals = [d.operational_mechanistic_distance(_attr(rng.normal(size=144)),
                                               _attr(rng.normal(size=144)))["value"]
            for _ in range(200)]
    assert abs(np.mean(vals) - 1.0) < 0.03


def test_dm_constant_vector_is_nan_with_reason() -> None:
    r = d.operational_mechanistic_distance(_attr([1.0] * 8), _attr(range(8)))
    assert math.isnan(r["value"]) and r["reason"]


def test_dm_rejects_mismatched_heads_and_nonfinite() -> None:
    with pytest.raises(ValueError):
        d.operational_mechanistic_distance(_attr(range(8)), _attr(range(12)))
    with pytest.raises(ValueError):
        d.operational_mechanistic_distance(_attr([np.nan] + [1.0] * 7), _attr(range(8)))


def test_topk_jaccard_values_and_tie_break() -> None:
    a = _attr([5, 4, 3, 0, 0, 0, 0, 0])
    assert d.topk_jaccard_distance(a, a, 3)["value"] == 0.0
    b = _attr([0, 0, 0, 0, 0, 3, 4, 5])
    assert d.topk_jaccard_distance(a, b, 3)["value"] == 1.0
    ties = _attr([1.0] * 8)
    assert d.topk_jaccard_distance(ties, ties, 2)["top_a"] == d.topk_jaccard_distance(
        ties, ties, 2)["top_b"]
    with pytest.raises(ValueError):
        d.topk_jaccard_distance(a, a, 0)


def test_topk_uses_magnitude() -> None:
    a = _attr([-9, 1, 1, 1, 0, 0, 0, 0])
    b = _attr([9, 1, 1, 1, 0, 0, 0, 0])
    assert d.topk_jaccard_distance(a, b, 1)["value"] == 0.0


# ---- D_B -----------------------------------------------------------------------------

def test_jsd_bounds_and_symmetry() -> None:
    p, q = [1.0, 0.0], [0.0, 1.0]
    assert d.js_divergence(p, q) == pytest.approx(1.0)
    assert d.js_divergence(p, p) == 0.0
    rng = np.random.default_rng(3)
    x, y = rng.dirichlet(np.ones(50)), rng.dirichlet(np.ones(50))
    assert d.js_divergence(x, y) == pytest.approx(d.js_divergence(y, x))
    assert 0.0 <= d.js_divergence(x, y) <= 1.0


def test_jsd_matches_scipy() -> None:
    from scipy.spatial.distance import jensenshannon

    rng = np.random.default_rng(4)
    x, y = rng.dirichlet(np.ones(30)), rng.dirichlet(np.ones(30))
    assert d.js_divergence(x, y) == pytest.approx(jensenshannon(x, y, base=2) ** 2)


def test_jsd_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        d.js_divergence([0.5, 0.5], [1.0])
    with pytest.raises(ValueError):
        d.js_divergence([-0.1, 1.1], [0.5, 0.5])


def test_behavioral_distance_null_and_positive() -> None:
    rng = np.random.default_rng(5)
    a = rng.dirichlet(np.ones(100), size=12)
    assert d.behavioral_distance(a, a.copy())["value"] == 0.0
    b = rng.dirichlet(np.ones(100), size=12)
    assert d.behavioral_distance(a, b)["value"] > 0.05
    with pytest.raises(ValueError):
        d.behavioral_distance(a, b[:5])


def test_distances_are_deterministic() -> None:
    rng = np.random.default_rng(7)
    a, b = _attr(rng.normal(size=48)), _attr(rng.normal(size=48))
    assert d.operational_mechanistic_distance(a, b) == d.operational_mechanistic_distance(a, b)
    assert d.topk_jaccard_distance(a, b, 5) == d.topk_jaccard_distance(a, b, 5)
