"""V6 Claim/Finding schema, provenance and IOI task generator."""
from __future__ import annotations

import json

import pytest

from glassbox.v6 import provenance, tasks
from glassbox.v6.claims import (
    Claim,
    EvidenceStatus,
    Finding,
    HypothesisResult,
    Measurement,
    Scope,
)

SCOPE = Scope(task="ioi", models=["a", "b"], dataset_hash="abc", metric_definitions="v")


def _finding(**kw) -> Finding:
    base = dict(run_id="r", scope=SCOPE,
                measurements=[Measurement("D_B", 0.1)],
                hypotheses=[HypothesisResult("H1", "e", "n", "t")],
                controls={}, record_hash="h")
    base.update(kw)
    return Finding(**base)


# ---- claims ---------------------------------------------------------------------------

def test_hypothesis_defaults_to_unresolved_with_reason() -> None:
    h = HypothesisResult("H1", "e", "n", "t")
    assert h.status is EvidenceStatus.UNRESOLVED and h.reason


def test_tested_hypothesis_needs_statistics() -> None:
    with pytest.raises(ValueError):
        HypothesisResult("H1", "e", "n", "t", status=EvidenceStatus.ASSOCIATED)
    HypothesisResult("H1", "e", "n", "t", status=EvidenceStatus.ASSOCIATED,
                     effect_size=0.3, ci=[0.1, 0.5], p_value=0.01)


def test_non_confirmatory_run_cannot_resolve_hypothesis() -> None:
    h = HypothesisResult("H1", "e", "n", "t", status=EvidenceStatus.ASSOCIATED,
                         effect_size=0.3, ci=[0.1, 0.5], p_value=0.01)
    with pytest.raises(ValueError):
        _finding(hypotheses=[h], label="pilot")
    _finding(hypotheses=[h], label="confirmatory")


def test_measurement_nan_becomes_unresolved_and_needs_reason() -> None:
    m = Measurement("D_M", float("nan"), reason="constant vector")
    assert m.value is None and m.status is EvidenceStatus.UNRESOLVED
    with pytest.raises(ValueError):
        Measurement("D_M", float("nan"))
    with pytest.raises(ValueError):
        Measurement("D_M", 0.3, status=EvidenceStatus.REPRODUCED)


def test_claim_ladder_rules() -> None:
    with pytest.raises(ValueError):
        Claim("x", EvidenceStatus.OBSERVED, SCOPE, evidence=[])
    with pytest.raises(ValueError):
        Claim("x", EvidenceStatus.CAUSALLY_SUPPORTED, SCOPE, evidence=["m1"])
    with pytest.raises(ValueError):
        Claim("x", EvidenceStatus.SURVIVED_FALSIFICATION, SCOPE, evidence=["m1"],
              gate_passed=2)
    Claim("x", EvidenceStatus.UNRESOLVED, SCOPE, evidence=[])


def test_scope_is_required() -> None:
    with pytest.raises(ValueError):
        Scope(task="", models=["a"], dataset_hash="h", metric_definitions="v")


def test_ladder_order() -> None:
    assert EvidenceStatus.UNRESOLVED.rank < EvidenceStatus.OBSERVED.rank
    assert EvidenceStatus.OBSERVED.rank < EvidenceStatus.REPRODUCED.rank


def test_finding_serialises_and_hash_is_stable() -> None:
    f1, f2 = _finding(), _finding()
    d = f1.to_dict()
    assert d["hypotheses"][0]["status"] == "UNRESOLVED"
    json.dumps(d)
    assert f1.content_hash() == f2.content_hash()
    assert _finding(record_hash="other").content_hash() != f1.content_hash()


def test_bad_label_rejected() -> None:
    with pytest.raises(ValueError):
        _finding(label="final")


# ---- provenance -----------------------------------------------------------------------

def test_sha256_json_is_order_independent() -> None:
    assert provenance.sha256_json({"a": 1, "b": 2}) == provenance.sha256_json({"b": 2, "a": 1})


def test_environment_record_has_required_fields(tmp_path) -> None:
    env = provenance.environment_record()
    assert {"git", "packages", "hardware"} <= set(env)
    assert "numpy" in env["packages"] and "python" in env["hardware"]
    f = tmp_path / "x.txt"
    f.write_bytes(b"abc")
    assert provenance.sha256_file(f) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


# ---- tasks ----------------------------------------------------------------------------

def test_ioi_is_deterministic_and_hashed() -> None:
    a, b = tasks.build_ioi(10, seed=3), tasks.build_ioi(10, seed=3)
    assert a.items == b.items and a.dataset_hash == b.dataset_hash
    assert tasks.build_ioi(10, seed=4).dataset_hash != a.dataset_hash
    assert tasks.build_ioi(10, seed=3, tokenizer_id="t").dataset_hash != a.dataset_hash


def test_ioi_item_structure_and_probes() -> None:
    ds = tasks.build_ioi(5, seed=0)
    assert len(ds.probes) == 15
    for i, it in enumerate(ds.items):
        assert it.prompt.endswith("to") and it.target == " " + it.name_a
        assert f", {it.name_b} gave" in it.prompt
        kinds = {p.kind: p for p in ds.probes if p.item_index == i}
        assert kinds["counterfactual"].expected == it.distractor
        assert f", {it.name_a} gave" in kinds["counterfactual"].prompt
        assert kinds["feature_swap"].prompt.startswith(f"When {it.name_b} and {it.name_a}")
        assert kinds["perturbation"].prompt != it.prompt


def test_ioi_single_token_filter() -> None:
    ds = tasks.build_ioi(20, seed=0, single_token=lambda s: len(s) <= 5)
    for it in ds.items:
        assert len(it.name_a) <= 4 and len(it.name_b) <= 4
    with pytest.raises(ValueError):
        tasks.build_ioi(2, single_token=lambda s: False)
    with pytest.raises(ValueError):
        tasks.build_ioi(0)
