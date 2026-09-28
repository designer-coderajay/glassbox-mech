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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformer_lens import HookedTransformer

from glassbox.core import GlassboxV2, _decision_value
from glassbox.v6.tasks import IOIItem, Probe

logger = logging.getLogger(__name__)

__all__ = [
    "load_model",
    "single_token_predicate",
    "evaluate_items",
    "probe_distributions",
    "mean_head_attribution",
]

Head = Tuple[int, int]


def load_model(name: str, checkpoint: Optional[int] = None,
               device: str = "cpu") -> HookedTransformer:
    """Load a model in eval mode; ``checkpoint`` selects a Pythia training step."""
    kwargs = {"checkpoint_value": checkpoint} if checkpoint is not None else {}
    model = HookedTransformer.from_pretrained(name, device=device, **kwargs)
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


def mean_head_attribution(model: HookedTransformer,
                          items: Sequence[IOIItem]) -> Dict[Head, float]:
    """Mean per-head attribution over items (Taylor attribution patching, name swap)."""
    gb = GlassboxV2(model)
    total: Dict[Head, float] = {}
    for it in items:
        clean = model.to_tokens(it.prompt)
        corrupted = model.to_tokens(
            GlassboxV2._name_swap(it.prompt, it.name_a, it.name_b))
        if clean.shape != corrupted.shape:
            raise ValueError(f"name swap changed sequence length: {it.prompt!r}")
        attr, _ = gb.attribution_patching(
            clean, corrupted, _tok(model, it.target), _tok(model, it.distractor),
            method="taylor")
        for head, v in attr.items():
            total[head] = total.get(head, 0.0) + float(v)
    return {h: v / len(items) for h, v in total.items()}
