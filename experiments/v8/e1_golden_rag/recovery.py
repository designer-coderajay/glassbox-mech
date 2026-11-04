"""Fix verification for E1 (PROTOCOL.md §6).

Verified recovery on a question requires both:
1. fixed vs candidate: correct rate higher, one-sided permutation p < 0.05 (999
   permutations, seeded);
2. fixed vs baseline: 95% percentile bootstrap CI of the accuracy difference inside
   [−0.15, +0.15].
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Sequence

N_PERM = 999
N_BOOT = 2000
ALPHA = 0.05
MARGIN = 0.15


def _rng(seed: str) -> random.Random:
    return random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))


def _mean(x: Sequence[int]) -> float:
    return sum(x) / len(x)


def p_greater(a: Sequence[int], b: Sequence[int], seed: str) -> float:
    """One-sided permutation p-value for mean(a) > mean(b)."""
    observed = _mean(a) - _mean(b)
    pooled, n_a, rng = list(a) + list(b), len(a), _rng(seed)
    hits = 0
    for _ in range(N_PERM):
        rng.shuffle(pooled)
        if _mean(pooled[:n_a]) - _mean(pooled[n_a:]) >= observed:
            hits += 1
    return (hits + 1) / (N_PERM + 1)


def diff_ci(a: Sequence[int], b: Sequence[int], seed: str) -> List[float]:
    """95% percentile bootstrap CI of mean(a) − mean(b)."""
    rng = _rng(seed)
    stats = sorted(
        _mean([a[rng.randrange(len(a))] for _ in a])
        - _mean([b[rng.randrange(len(b))] for _ in b])
        for _ in range(N_BOOT)
    )
    return [stats[int(0.025 * N_BOOT)], stats[int(0.975 * N_BOOT) - 1]]


def verified(
    fixed: Sequence[int], cand: Sequence[int], base: Sequence[int], seed: str
) -> bool:
    """Both recovery conditions hold for one question."""
    if p_greater(fixed, cand, f"{seed}|perm") >= ALPHA:
        return False
    lo, hi = diff_ci(fixed, base, f"{seed}|ci")
    return -MARGIN <= lo and hi <= MARGIN


def fix_config(candidate: Dict[str, Any], attributed: Sequence[str]) -> Dict[str, Any]:
    """Candidate with every attributed factor reverted to its baseline value."""
    from_base = {
        "index": "fresh",
        "chunk_words": 120,
        "template": "T1",
        "log_level": "info",
    }
    return {**candidate, **{f: from_base[f] for f in attributed}}


def wrong_fix_config(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """The control: revert only the visible suspect, ``template``."""
    return {**candidate, "template": "T1"}


def recovery_question(
    system: Any,
    q: Dict[str, Any],
    attributed: Sequence[str],
    base_cfg: Dict[str, Any],
    cand_cfg: Dict[str, Any],
    n: int,
) -> Dict[str, Any]:
    """Run baseline, candidate, fixed and wrong-fix ``n`` times; apply the rule."""

    def sample(config: Dict[str, Any]) -> List[int]:
        return [int(system.answer(config, q, i)[0] == "correct") for i in range(n)]

    base, cand = sample(base_cfg), sample(cand_cfg)
    wrong = sample(wrong_fix_config(cand_cfg))
    out: Dict[str, Any] = {
        "n": n,
        "baseline_correct": sum(base),
        "candidate_correct": sum(cand),
        "wrong_fix_correct": sum(wrong),
        "wrong_fix": verified(wrong, cand, base, f"{q['id']}|wrong"),
        "attributed": list(attributed),
    }
    if attributed:
        fixed = sample(fix_config(cand_cfg, attributed))
        out["fixed_correct"] = sum(fixed)
        out["fixed"] = verified(fixed, cand, base, f"{q['id']}|fixed")
    else:
        out["fixed_correct"], out["fixed"] = None, None
    return out
