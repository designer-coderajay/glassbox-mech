"""V6 model loading: official TransformerLens names vs Pythia seed variants.

Seed variants (``pythia-410m-seed3`` etc.) are not in the TransformerLens model list, so
they are loaded with transformers at an explicit revision and handed to TransformerLens as
the base architecture. These tests mock both libraries (no downloads). A real-weights
equivalence test (TransformerLens logits == transformers logits) runs only when
``GLASSBOX_V6_E2E=1``.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from glassbox.v6 import measure


@pytest.fixture
def calls(monkeypatch):
    log = {}

    def fake_ht(name, **kw):
        log["ht"] = (name, kw)
        return SimpleNamespace(eval=lambda: None)

    def fake_hf(repo, **kw):
        log["hf"] = (repo, kw)
        return "HF_MODEL"

    monkeypatch.setattr(measure.HookedTransformer, "from_pretrained", fake_ht)
    monkeypatch.setattr(measure, "_hf_from_pretrained", fake_hf)
    return log


def test_seed_variant_parsing() -> None:
    assert measure.seed_variant("pythia-410m-seed3") == ("pythia-410m", 3)
    assert measure.seed_variant("pythia-1.4b-seed12") == ("pythia-1.4b", 12)
    assert measure.seed_variant("pythia-410m") is None
    assert measure.seed_variant("pythia-410m-deduped") is None


def test_official_model_uses_checkpoint_value(calls) -> None:
    measure.load_model("pythia-410m", 143000)
    name, kw = calls["ht"]
    assert name == "pythia-410m" and kw["checkpoint_value"] == 143000
    assert "hf" not in calls


def test_seed_variant_loads_revision_and_passes_hf_model(calls) -> None:
    measure.load_model("pythia-410m-seed3", 143000)
    repo, hf_kw = calls["hf"]
    assert repo == "EleutherAI/pythia-410m-seed3"
    assert hf_kw["revision"] == "step143000" and hf_kw["torch_dtype"] == torch.float32
    name, kw = calls["ht"]
    assert name == "pythia-410m" and kw["hf_model"] == "HF_MODEL"
    # checkpoint_value must NOT be passed: TransformerLens would then ignore hf_model
    # and silently load the official checkpoint instead of the seed variant.
    assert "checkpoint_value" not in kw


def test_seed_variant_requires_checkpoint(calls) -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        measure.load_model("pythia-410m-seed3", None)


@pytest.mark.skipif(os.environ.get("GLASSBOX_V6_E2E") != "1",
                    reason="set GLASSBOX_V6_E2E=1 to download pythia-70m-seed1")
def test_seed_variant_logits_match_transformers() -> None:
    from transformers import AutoModelForCausalLM

    tl = measure.load_model("pythia-70m-seed1", 143000)
    hf = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m-seed1", revision="step143000", torch_dtype=torch.float32)
    toks = tl.to_tokens("When Mary and John went to the store, John gave a drink to")
    with torch.no_grad():
        a = torch.softmax(tl(toks)[0, -1], -1)
        b = torch.softmax(hf(toks).logits[0, -1], -1)
    # TransformerLens weight processing (LN folding, centering) gives ~1e-4 probability
    # differences in fp32; measured 2026-09-29 as the same size for the official
    # pythia-70m path (1.0e-4) and this seed path (5.7e-5).
    assert (a - b).abs().max() < 1e-3 and a.argmax() == b.argmax()
    base = measure.load_model("pythia-70m", 143000)
    with torch.no_grad():
        c = torch.softmax(base(toks)[0, -1], -1)
    assert (a - c).abs().max() > 0.05  # really a different model (measured 0.195)
