"""Fix verification for E1b (PROTOCOL.md §6).

Arms per test question, ``n`` runs each (seed ``e1b-recovery``): baseline, candidate,
``gb_fix`` (revert GB's harmful factors), ``wrong_fix`` (revert ``template`` only) and
``b2_fix`` (revert B2's blamed factors). An arm with nothing to revert is not run. Arms
with the same configuration share one set of runs.

Verified recovery of a fix (one-sided):
1. better than the candidate: one-sided permutation p < 0.05 (999 permutations);
2. not worse than the baseline: the 95% bootstrap CI lower bound of (fix − baseline)
   accuracy is ≥ −0.15. A fix may exceed the baseline.

E1's two-sided rule (the CI inside [−0.15, +0.15]) is also reported, descriptively.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List, Sequence

N_PERM = 999
N_BOOT = 2000
ALPHA = 0.05
MARGIN = 0.15
FROM_BASE = {
    "index": "fresh",
    "chunk_words": 120,
    "template": "T1",
    "log_level": "info",
}


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
    """E1b rule: better than the candidate and not more than 0.15 below baseline."""
    if p_greater(fixed, cand, f"{seed}|perm") >= ALPHA:
        return False
    return diff_ci(fixed, base, f"{seed}|ci")[0] >= -MARGIN


def verified_two_sided(
    fixed: Sequence[int], cand: Sequence[int], base: Sequence[int], seed: str
) -> bool:
    """E1's rule (descriptive only): the CI of fix − baseline inside ±0.15."""
    if p_greater(fixed, cand, f"{seed}|perm") >= ALPHA:
        return False
    lo, hi = diff_ci(fixed, base, f"{seed}|ci")
    return -MARGIN <= lo and hi <= MARGIN


def fix_config(candidate: Dict[str, Any], revert: Sequence[str]) -> Dict[str, Any]:
    """Candidate with every factor in ``revert`` set back to its baseline value."""
    return {**candidate, **{f: FROM_BASE[f] for f in revert}}


def wrong_fix_config(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """The control: revert only the visible suspect, ``template``."""
    return {**candidate, "template": "T1"}


def recovery_question(
    system: Any,
    q: Dict[str, Any],
    gb_revert: Sequence[str],
    b2_revert: Sequence[str],
    base_cfg: Dict[str, Any],
    cand_cfg: Dict[str, Any],
    n: int,
) -> Dict[str, Any]:
    """Run the arms ``n`` times each and apply both recovery rules."""
    runs: Dict[str, List[int]] = {}

    def sample(config: Dict[str, Any]) -> List[int]:
        key = json.dumps(config, sort_keys=True)
        if key not in runs:
            runs[key] = [
                int(system.answer(config, q, i)[0] == "correct") for i in range(n)
            ]
        return runs[key]

    base, cand = sample(dict(base_cfg)), sample(dict(cand_cfg))
    arms_spec = {
        "gb_fix": list(gb_revert),
        "wrong_fix": ["template"],
        "b2_fix": list(b2_revert),
    }
    arms: Dict[str, Any] = {}
    for name, revert in arms_spec.items():
        if not revert:
            arms[name] = {"reverted": [], "run": False}
            continue
        cfg = fix_config(dict(cand_cfg), revert)
        fixed = sample(cfg)
        seed = f"{q['id']}|{name}"
        lo, hi = diff_ci(fixed, base, f"{seed}|ci")
        arms[name] = {
            "reverted": list(revert),
            "run": True,
            "correct": sum(fixed),
            "verified": verified(fixed, cand, base, seed),
            "verified_two_sided": verified_two_sided(fixed, cand, base, seed),
            "diff_vs_baseline_ci": [round(lo, 4), round(hi, 4)],
        }
    return {
        "n": n,
        "baseline_correct": sum(base),
        "candidate_correct": sum(cand),
        "arms": arms,
    }
