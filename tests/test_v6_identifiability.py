"""Head identifiability: group action, weight-level symmetry, candidate distances."""
from __future__ import annotations

import itertools
import os

import numpy as np
import pytest
import torch
from scipy import stats

from glassbox.v6 import identifiability as idf

RNG = np.random.default_rng


# ── group action + weight-level symmetry ──────────────────────────────────────

def _tiny_model(seed: int = 0):
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    torch.manual_seed(seed)
    cfg = HookedTransformerConfig(
        n_layers=3, d_model=32, n_ctx=16, d_head=8, n_heads=4, d_vocab=50,
        act_fn="gelu", positional_embedding_type="rotary", rotary_dim=4,
        parallel_attn_mlp=True, normalization_type="LN", seed=seed)
    m = HookedTransformer(cfg)
    m.eval()
    return m


def _attr_matrix(model, clean, corr):
    from glassbox.core import GlassboxV2

    attr, _ = GlassboxV2(model).attribution_patching(clean, corr, 7, 11)
    n_l, n_h = model.cfg.n_layers, model.cfg.n_heads
    return np.array([[attr[(l, h)] for h in range(n_h)] for l in range(n_l)])


def test_weight_permutation_preserves_function_and_permutes_attribution() -> None:
    # The relabelling null is a real transformer symmetry for this measurement iff
    # (1) permuting head parameter slices leaves outputs unchanged and
    # (2) attribution patching on the permuted model returns pi . attribution.
    m = _tiny_model()
    clean = torch.tensor([[1, 5, 9, 12, 3, 8, 2]])
    corr = torch.tensor([[1, 6, 9, 12, 4, 8, 2]])
    with torch.no_grad():
        before = m(clean)
    a = _attr_matrix(m, clean, corr)
    pi = idf.random_group_element(3, 4, RNG(1))
    idf.permute_heads_(m, pi)
    with torch.no_grad():
        after = m(clean)
    assert torch.allclose(before, after, atol=1e-5)
    b = _attr_matrix(m, clean, corr)
    assert np.allclose(b, idf.act_on(pi, a), atol=1e-6)
    assert not np.allclose(b, a)  # a non-trivial relabelling


def test_act_on_is_a_group_action() -> None:
    rng = RNG(0)
    x = rng.normal(size=(5, 3, 4))
    p, q = idf.random_group_element(3, 4, rng), idf.random_group_element(3, 4, rng)
    ident = np.tile(np.arange(4), (3, 1))
    assert np.array_equal(idf.act_on(ident, x), x)
    # (p . (q . x))[l, h] = (q . x)[l, p(h)] = x[l, q(p(h))]
    composed = np.take_along_axis(q, p, axis=1)
    assert np.array_equal(idf.act_on(p, idf.act_on(q, x)), idf.act_on(composed, x))


# ── positional permutation null ───────────────────────────────────────────────

def test_positional_null_matches_brute_force() -> None:
    rng = RNG(3)
    a, b = rng.normal(size=(4, 5)), rng.normal(size=(4, 5))
    null = idf.positional_permutation_null(a, b, n_perm=4000, seed=0)
    brute = [idf.positional_dm(a, idf.act_on(idf.random_group_element(4, 5, rng), b))
             for _ in range(4000)]
    assert null["median"] == pytest.approx(np.median(brute), abs=0.03)
    assert null["observed"] == pytest.approx(idf.positional_dm(a, b))


def test_positional_null_flags_aligned_vs_relabelled() -> None:
    rng = RNG(4)
    a = rng.normal(size=(12, 16))
    aligned = a + 0.1 * rng.normal(size=a.shape)
    relabelled = idf.act_on(idf.random_group_element(12, 16, rng), aligned)
    assert idf.positional_permutation_null(a, aligned)["observed_percentile"] < 1.0
    assert 5.0 < idf.positional_permutation_null(a, relabelled)["observed_percentile"] < 95.0


# ── candidate distances ───────────────────────────────────────────────────────

def test_scalar_quotient_equals_min_over_group_brute_force() -> None:
    rng = RNG(5)
    a, b = rng.normal(size=(2, 3)), rng.normal(size=(2, 3))
    best = min(idf.positional_dm(a, idf.act_on(np.array([p, q]), b))
               for p in itertools.permutations(range(3))
               for q in itertools.permutations(range(3)))
    assert idf.scalar_quotient_dm(a, b) == pytest.approx(best)


def test_profile_orbit_is_zero_for_relabelled_rescaled_self_and_invariant() -> None:
    rng = RNG(6)
    x = rng.normal(size=(30, 4, 6))
    pi = idf.random_group_element(4, 6, rng)
    assert idf.profile_orbit_distance(x, 3.0 * idf.act_on(pi, x))["value"] == pytest.approx(
        0.0, abs=1e-12)
    y = rng.normal(size=x.shape)
    d1 = idf.profile_orbit_distance(x, y)["value"]
    assert d1 == pytest.approx(idf.profile_orbit_distance(x, idf.act_on(pi, y))["value"])
    assert d1 == pytest.approx(idf.profile_orbit_distance(y, x)["value"])
    assert 0.0 < d1 <= 4.0


def test_profile_orbit_sqrt_satisfies_triangle_inequality() -> None:
    rng = RNG(7)
    for _ in range(20):
        x, y, z = (rng.normal(size=(10, 3, 4)) for _ in range(3))
        dxy, dyz, dxz = (np.sqrt(idf.profile_orbit_distance(p, q)["value"])
                         for p, q in ((x, y), (y, z), (x, z)))
        assert dxz <= dxy + dyz + 1e-12


def test_crossfit_aligned_is_exactly_relabelling_invariant() -> None:
    rng = RNG(8)
    x, y = rng.normal(size=(40, 3, 5)), rng.normal(size=(40, 3, 5))
    pi = idf.random_group_element(3, 5, rng)
    d1 = idf.crossfit_aligned_dm(x, y, n_splits=5, seed=1)["value"]
    d2 = idf.crossfit_aligned_dm(x, idf.act_on(pi, y), n_splits=5, seed=1)["value"]
    assert d1 == pytest.approx(d2, abs=1e-12)


# ── synthetic scenarios A-E (small version of experiments/v6/audits/...) ─────

def _scenarios():
    import importlib.util
    import pathlib

    path = (pathlib.Path(__file__).resolve().parents[1]
            / "experiments/v6/audits/identifiability_synthetic.py")
    spec = importlib.util.spec_from_file_location("idsyn", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [mod.replicate(s, sparse) for s in range(4) for sparse in (False, True)]


@pytest.fixture(scope="module")
def reps():
    return _scenarios()


def _mean(reps, sc, k):
    return float(np.mean([r[sc][k] for r in reps]))


def test_positional_dm_fails_relabelling_scenario_a(reps) -> None:
    assert _mean(reps, "A", "positional") > 0.8  # regression: labels, not mechanism


@pytest.mark.parametrize("metric", ["crossfit_aligned", "profile_orbit"])
def test_invariant_profile_metrics_pass_a_c_d_e(reps, metric) -> None:
    for r in reps:
        assert r["A"][metric] == pytest.approx(r["F"][metric], abs=0.05)
        assert r["D"][metric] == pytest.approx(r["C"][metric], abs=1e-9)
    assert _mean(reps, "C", metric) > 2 * _mean(reps, "B", metric)
    assert _mean(reps, "E", metric) > 2 * _mean(reps, "B", metric)


def test_scalar_quotient_cannot_see_mechanism_with_same_magnitudes(reps) -> None:
    # Regression for item 7 of the identifiability audit: sorting scalar attributions
    # compares magnitude distributions only; scenario E (same magnitudes, different
    # per-input behaviour) looks almost like a perturbed copy.
    assert _mean(reps, "E", "scalar_quotient") < 0.5 * _mean(reps, "C", "scalar_quotient")


@pytest.mark.skipif(os.environ.get("GLASSBOX_V6_E2E") != "1",
                    reason="set GLASSBOX_V6_E2E=1 to run on pythia-70m")
def test_weight_permutation_symmetry_on_pythia() -> None:
    from glassbox.v6 import measure

    m = measure.load_model("pythia-70m", 143000)
    clean = m.to_tokens("When Mary and John went to the store, John gave a drink to")
    corr = m.to_tokens("When John and Mary went to the store, Mary gave a drink to")
    t, d = m.to_single_token(" Mary"), m.to_single_token(" John")
    from glassbox.core import GlassboxV2

    def attr(model):
        a, _ = GlassboxV2(model).attribution_patching(clean, corr, t, d)
        return np.array([[a[(l, h)] for h in range(8)] for l in range(6)])

    with torch.no_grad():
        before = m(clean)
    a = attr(m)
    pi = idf.random_group_element(6, 8, RNG(2))
    idf.permute_heads_(m, pi)
    # Tolerances: logits are O(800) in float32 and permuting heads changes the summation
    # order (observed max abs diff 3.7e-3, relative 4.5e-6); the backward pass amplifies
    # this to 0.12 % of max |attr| (observed 2026-09-29). Rank agreement is the test.
    with torch.no_grad():
        assert torch.allclose(before, m(clean), rtol=1e-5, atol=1e-2)
    b = attr(m)
    assert np.abs(b - idf.act_on(pi, a)).max() < 1e-2 * np.abs(a).max()
    assert stats.spearmanr(b.ravel(), idf.act_on(pi, a).ravel()).correlation > 0.999
    # ... while the pre-registered positional D_M calls the functionally identical
    # twin maximally different (observed Spearman -0.04, D_M 1.04).
    assert idf.positional_dm(a, b) > 0.8


# ── baseline tools ────────────────────────────────────────────────────────────

def _orbit_reference(xa, xb):
    """Direct (slow) definition: unit-normalise, explicit squared-distance assignment."""
    from scipy.optimize import linear_sum_assignment

    a, b = xa / np.linalg.norm(xa), xb / np.linalg.norm(xb)
    total = 0.0
    for layer in range(a.shape[1]):
        c = ((a[:, layer, :].T[:, None, :] - b[:, layer, :].T[None]) ** 2).sum(-1)
        r, k = linear_sum_assignment(c)
        total += c[r, k].sum()
    return total


def test_orbit_distance_matches_direct_definition() -> None:
    rng = RNG(12)
    for _ in range(10):
        x, y = rng.normal(size=(30, 4, 6)), rng.normal(size=(30, 4, 6))
        assert idf.profile_orbit_distance(x, y)["value"] == pytest.approx(
            _orbit_reference(x, y), abs=1e-10)


def test_weighted_orbit_equals_resampled_orbit() -> None:
    # A bootstrap replicate is the same statistic with prompts weighted by their counts.
    rng = RNG(13)
    x, y = rng.normal(size=(25, 3, 5)), rng.normal(size=(25, 3, 5))
    idx = rng.integers(0, 25, size=(6, 25))
    w = np.stack([np.bincount(i, minlength=25) for i in idx]).astype(float)
    fast = idf.orbit_distance_weighted(x, y, w)
    slow = [_orbit_reference(x[i], y[i]) for i in idx]
    assert fast == pytest.approx(slow, abs=1e-10)

def test_crossfit_orbit_brackets_and_is_invariant() -> None:
    rng = RNG(9)
    x, y = rng.normal(size=(60, 3, 5)), rng.normal(size=(60, 3, 5))
    r = idf.crossfit_orbit_distance(x, y, n_splits=5, seed=0)
    assert r["insample_half"] <= r["crossfit"] + 1e-12
    pi = idf.random_group_element(3, 5, rng)
    r2 = idf.crossfit_orbit_distance(x, idf.act_on(pi, y), n_splits=5, seed=0)
    assert r2["crossfit"] == pytest.approx(r["crossfit"], abs=1e-12)


def test_orbit_bootstrap_is_invariant_and_deterministic() -> None:
    rng = RNG(10)
    x, y = rng.normal(size=(40, 2, 4)), rng.normal(size=(40, 2, 4))
    pi = idf.random_group_element(2, 4, rng)
    b1 = idf.orbit_prompt_bootstrap(x, y, n_boot=50, seed=3)
    b2 = idf.orbit_prompt_bootstrap(x, idf.act_on(pi, y), n_boot=50, seed=3)
    for k in ("estimate", "boot_mean", "bias"):
        assert b1[k] == pytest.approx(b2[k], abs=1e-12)
    assert b1["ci_percentile"] == pytest.approx(b2["ci_percentile"], abs=1e-12)
    assert b1["ci_percentile"][0] <= b1["ci_percentile"][1]


def test_alignment_recovery_identity_for_near_copy() -> None:
    rng = RNG(11)
    x = rng.normal(size=(50, 3, 6))
    r = idf.alignment_recovery(x, x + 0.01 * rng.normal(size=x.shape))
    assert r["fraction_identity"] == 1.0
    pi = idf.random_group_element(3, 6, rng)
    assert idf.alignment_recovery(x, idf.act_on(pi, x))["fraction_identity"] < 1.0


def test_crossfit_bootstrap_is_invariant_and_deterministic() -> None:
    rng = RNG(14)
    x, y = rng.normal(size=(40, 3, 4)), rng.normal(size=(40, 3, 4))
    pi = idf.random_group_element(3, 4, rng)
    a = idf.crossfit_bootstrap(x, y, n_boot=20, n_splits=3, seed=5)
    b = idf.crossfit_bootstrap(x, idf.act_on(pi, y), n_boot=20, n_splits=3, seed=5)
    assert a == pytest.approx(b, abs=1e-12)
    assert a.shape == (20,)
    assert np.array_equal(a, idf.crossfit_bootstrap(x, y, n_boot=20, n_splits=3, seed=5))


def test_bracketed_v2_interval_orders_bounds() -> None:
    rng = RNG(15)
    x, y = rng.normal(size=(40, 3, 4)), rng.normal(size=(40, 3, 4))
    r = idf.bracketed_v2_interval(x, y, n_boot=30, n_splits=3, seed=0)
    assert r["lower"] <= r["estimate"] <= r["upper"]
