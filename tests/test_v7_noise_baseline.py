"""Mechanics of V7 Experiment 3 (experiments/v7/noise_baseline) without running it.

Part A is not replicated with the protocol's seeds here, and Part B does not load
GPT-2: a fake sampler is used. These tests check the generators, the criteria logic
(including that failing worlds FAIL) and the never-overwrite rule.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

from glassbox.v7.trace import read_trace

EXP = Path(__file__).resolve().parent.parent / "experiments" / "v7" / "noise_baseline"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"exp3_{name}", EXP / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pa():
    return _load("part_a")


@pytest.fixture(scope="module")
def pb():
    return _load("part_b")


# --- Part A ---------------------------------------------------------------------


def test_part_a_constants_match_protocol(pa):
    assert (pa.N_PER_GROUP, pa.REPLICATIONS, pa.N_PERM, pa.ALPHA) == (
        30,
        200,
        999,
        0.05,
    )
    assert pa.MASTER_SEED == 20261005
    assert pa.BASELINE == pa.Params(0.90, 0.85, 0.20)
    assert pa.CANDIDATE["W0"] == pa.BASELINE
    assert pa.CANDIDATE["W1"] == pa.Params(0.90, 0.35, 0.20)
    assert pa.CANDIDATE["W2"] == pa.Params(0.40, 0.85, 0.20)
    assert pa.CANDIDATE["W3"] == pa.Params(0.40, 0.35, 0.20)


def test_simulate_run_structure(pa, tmp_path):
    rng = random.Random(1)
    always = pa.Params(1.0, 1.0, 0.0)
    never = pa.Params(1.0, 0.0, 0.0)
    spans_yes, _ = read_trace(pa.simulate_run(rng, always, tmp_path, "y"))
    spans_no, _ = read_trace(pa.simulate_run(rng, never, tmp_path, "n"))
    names = [s["attributes"].get("gen_ai.operation.name") for s in spans_yes[1:]]
    assert names == ["retrieval", "chat", "execute_tool"]
    assert len(spans_no) == 3  # root, retrieval, chat (no tool when distractor)
    chat = spans_yes[2]["attributes"]
    assert chat["glassbox.answer.label"] == "correct"
    assert "gen_ai.output.messages" not in chat  # hash_only: no raw content


def test_replicate_smoke_outside_protocol_range(pa):
    """Tiny, non-protocol settings (rep=-1, n=4, n_perm=9): mechanics only."""
    s = pa.replicate("W0", -1, n=4, n_perm=9)
    assert set(s) == {
        "any_significant",
        "first",
        "significant_steps",
        "output_hash_classification",
        "pairwise_divergence_rate",
    }
    assert s["output_hash_classification"] == "untestable"
    assert 0.0 <= s["pairwise_divergence_rate"] <= 1.0


def _summ(first=None, steps=(), pairwise=None, out_cls="untestable"):
    return {
        "any_significant": bool(steps),
        "first": list(first) if first else None,
        "significant_steps": sorted(steps),
        "output_hash_classification": out_cls,
        "pairwise_divergence_rate": pairwise,
    }


def _good_world(pa, n=10):
    chat = ("chat#0", "glassbox.answer.label")
    ret = ("retrieval#0", "glassbox.content.gen_ai.retrieval.documents.sha256")
    return {
        "W0": [_summ(pairwise=1.0) for _ in range(n)],
        "W1": [_summ(chat, ["chat#0"]) for _ in range(n)],
        "W2": [_summ(ret, ["retrieval#0", "chat#0"]) for _ in range(n)],
        "W3": [_summ(ret, ["retrieval#0", "chat#0"]) for _ in range(n)],
    }


def test_part_a_evaluate_pass(pa):
    crit = pa.evaluate(_good_world(pa))
    assert pa.overall(crit) == "PASS"
    assert crit["A5_one_vs_two_causes"]["chat_flagged_W2"] == 1.0
    assert crit["A6_pairwise_noise"]["mean_pairwise_divergence_rate_W0"] == 1.0


def test_part_a_noisy_null_fails(pa):
    world = _good_world(pa)
    world["W0"] = [_summ(("chat#0", "x"), ["chat#0"], pairwise=1.0)] * 10
    crit = pa.evaluate(world)
    assert not crit["A1_type_I"]["pass"]
    assert pa.overall(crit) == "FAIL"


def test_part_a_wrong_localisation_fails(pa):
    world = _good_world(pa)
    world["W1"] = [_summ(("retrieval#0", "x"), ["retrieval#0", "chat#0"])] * 10
    crit = pa.evaluate(world)
    assert not crit["A2_chat_localisation"]["pass"]
    assert not crit["A2b_no_false_upstream"]["pass"]


def test_part_a_tested_free_text_fails_a4(pa):
    world = _good_world(pa)
    world["W2"][0] = _summ(("retrieval#0", "x"), ["retrieval#0"], out_cls="tested")
    assert not pa.evaluate(world)["A4_free_text_untestable"]["pass"]


# --- Part B ---------------------------------------------------------------------


def _fake_sampler(prompt: str, seed: int) -> str:
    """Deterministic stand-in for GPT-2: varies with the seed, not with the condition."""
    words = ["Kestrel", "the", "a", "Halcyon", "and", "of", "to"]
    rng = random.Random(seed)
    return " " + " ".join(rng.choice(words) for _ in range(4))


def test_part_b_constants_match_protocol(pb):
    assert list(pb.SEEDS["baseline"]) == list(range(0, 20))
    assert list(pb.SEEDS["baseline_more"]) == list(range(100, 120))
    assert list(pb.SEEDS["fault"]) == list(range(200, 220))
    assert (pb.N_PERM, pb.ALPHA, pb.MAX_NEW_TOKENS, pb.TEMPERATURE) == (
        1999,
        0.05,
        8,
        1.0,
    )


def test_part_b_pipeline_with_fake_sampler(pb, tmp_path):
    out = tmp_path / "part_b"
    results = pb.run(out, _fake_sampler, model_name="fake")
    crit = results["criteria"]
    assert crit["B2_real_localisation"]["pass"]
    assert crit["B3_real_specificity"]["pass"]
    assert crit["B1_real_null"]["pass"]
    assert results["overall"] == "PASS"
    assert crit["B4_descriptive"]["pairwise_divergence_rate_baseline_vs_more"] == 1.0
    assert len(list(out.glob("*/*/*.trace.jsonl"))) == 3 * 8 * 20
    saved = json.loads((out / "results.json").read_text())
    assert saved["model"] == "fake"
    with pytest.raises(FileExistsError):
        pb.run(out, _fake_sampler, model_name="fake")


def test_part_b_traces_hold_no_raw_text(pb, tmp_path):
    out = tmp_path / "pb"
    pb.run(out, _fake_sampler, model_name="fake")
    blob = "".join(p.read_text() for p in out.glob("*/*/*.trace.jsonl"))
    assert "Kestrel" not in blob and "Question:" not in blob
