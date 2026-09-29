"""Model-side measurements for V6 (torch + TransformerLens).

Everything that touches a model lives here; the distances are computed elsewhere from
the plain arrays returned by these functions.

Attribution instrument (D_M input): :meth:`glassbox.core.GlassboxV2.attribution_patching`
(Taylor, 3 passes) with the IOI name-swap corruption, averaged over items. The average
over items is a pre-registered analytic choice (PREREGISTRATION.md §3.2), not a property
of the models.
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformer_lens import HookedTransformer

from glassbox.core import GlassboxV2, _decision_value
from glassbox.v6.tasks import IOIItem, Probe

logger = logging.getLogger(__name__)

__all__ = [
    "load_model",
    "seed_variant",
    "single_token_predicate",
    "evaluate_items",
    "probe_distributions",
    "head_attribution_matrix",
    "mean_head_attribution",
]

Head = Tuple[int, int]


_SEED_RE = re.compile(r"^(pythia-[0-9.]+[mb])-seed(\d+)$")


def seed_variant(name: str) -> Optional[Tuple[str, int]]:
    """``"pythia-410m-seed3"`` -> ``("pythia-410m", 3)``; None for any other name."""
    m = _SEED_RE.match(name)
    return (m.group(1), int(m.group(2))) if m else None


def _hf_from_pretrained(repo: str, **kwargs: object) -> object:
    """transformers loader (seam for tests). For ``.bin`` files transformers uses
    ``torch.load(weights_only=True)`` and requires torch >= 2.6."""
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(repo, **kwargs)


def load_model(name: str, checkpoint: Optional[int] = None,
               device: str = "cpu") -> HookedTransformer:
    """Load a model in eval mode; ``checkpoint`` selects a Pythia training step.

    Pythia seed variants (``pythia-410m-seed1`` ... ``seed9``, EleutherAI, Apache-2.0)
    are not in the TransformerLens model list. They are loaded with transformers at
    revision ``step{checkpoint}`` and converted by TransformerLens as the base
    architecture (same config and vocabulary). ``checkpoint_value`` is deliberately not
    passed in that case: TransformerLens would ignore ``hf_model`` and load the official
    base checkpoint instead.
    """
    seed = seed_variant(name)
    if seed is None:
        kwargs = {"checkpoint_value": checkpoint} if checkpoint is not None else {}
        model = HookedTransformer.from_pretrained(name, device=device, **kwargs)
    else:
        if checkpoint is None:
            raise ValueError(f"{name}: seed variants need an explicit checkpoint step")
        hf = _hf_from_pretrained(f"EleutherAI/{name}", revision=f"step{checkpoint}",
                                 torch_dtype=torch.float32)
        model = HookedTransformer.from_pretrained(seed[0], hf_model=hf, device=device)
    model.eval()
    return model


def single_token_predicate(model: HookedTransformer) -> Callable[[str], bool]:
    """Return ``s -> True`` iff ``s`` encodes to exactly one token for this model."""

    def pred(s: str) -> bool:
        return len(model.to_str_tokens(s, prepend_bos=False)) == 1

    return pred


def _tok(model: HookedTransformer, s: str) -> int:
    return int(model.to_single_token(s))


@torch.no_grad()
def evaluate_items(model: HookedTransformer,
                   items: Sequence[IOIItem]) -> Tuple[List[bool], List[float]]:
    """Per item: correct (target logit > distractor logit) and logit difference."""
    correct: List[bool] = []
    lds: List[float] = []
    for it in items:
        logits = model(model.to_tokens(it.prompt))
        ld = float(_decision_value(logits, _tok(model, it.target),
                                   _tok(model, it.distractor), fp32_each=True))
        lds.append(ld)
        correct.append(ld > 0.0)
    return correct, lds


@torch.no_grad()
def probe_distributions(model: HookedTransformer, probes: Sequence[Probe]) -> np.ndarray:
    """``[n_probes, d_vocab]`` float64 next-token distributions at the last position."""
    rows = []
    for p in probes:
        logits = model(model.to_tokens(p.prompt))[0, -1].float()
        rows.append(torch.softmax(logits, dim=-1).double().cpu().numpy())
    return np.stack(rows)


def head_attribution_matrix(model: HookedTransformer,
                            items: Sequence[IOIItem]) -> Tuple[List[Head], np.ndarray]:
    """Per-item, per-head attribution: ``(heads, [n_items, n_heads])``.

    Taylor attribution patching with the IOI name-swap corruption. Heads are sorted.
    """
    gb = GlassboxV2(model)
    rows: List[Dict[Head, float]] = []
    for it in items:
        clean = model.to_tokens(it.prompt)
        corrupted = model.to_tokens(
            GlassboxV2._name_swap(it.prompt, it.name_a, it.name_b))
        if clean.shape != corrupted.shape:
            raise ValueError(f"name swap changed sequence length: {it.prompt!r}")
        attr, _ = gb.attribution_patching(
            clean, corrupted, _tok(model, it.target), _tok(model, it.distractor),
            method="taylor")
        rows.append({head: float(v) for head, v in attr.items()})
    heads = sorted(rows[0])
    return heads, np.array([[r[h] for h in heads] for r in rows], dtype=np.float64)


def mean_head_attribution(model: HookedTransformer,
                          items: Sequence[IOIItem]) -> Dict[Head, float]:
    """Mean per-head attribution over items (the D_M input vector)."""
    heads, matrix = head_attribution_matrix(model, items)
    return dict(zip(heads, matrix.mean(axis=0).tolist()))
