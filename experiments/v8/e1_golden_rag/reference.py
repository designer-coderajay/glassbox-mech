"""Reference (ground-truth) labels for E1 (PROTOCOL.md §4).

The true effect of factor X on a question is TVD(only_X, baseline) on binary outcomes,
which equals the absolute difference of the correct rates. A 95% percentile bootstrap
CI (2,000 resamples, seeded) gives the label: ``large`` if the lower bound ≥ 0.20,
``null`` if the upper bound ≤ 0.10, otherwise ``small``. ``log_level`` is ``null`` by
construction.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Sequence, Tuple

N_BOOT = 2000
LARGE_LO = 0.20
NULL_HI = 0.10
INERT = ("log_level",)


def _rng(seed: str) -> random.Random:
    return random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))


def tvd(a: Sequence[int], b: Sequence[int]) -> float:
    """TVD of two binary samples (1 = correct)."""
    return abs(sum(a) / len(a) - sum(b) / len(b))


def tvd_ci(a: Sequence[int], b: Sequence[int], seed: str) -> Tuple[float, float]:
    """95% percentile bootstrap CI of the TVD, resampling each sample independently."""
    rng = _rng(seed)
    stats = []
    for _ in range(N_BOOT):
        ra = [a[rng.randrange(len(a))] for _ in a]
        rb = [b[rng.randrange(len(b))] for _ in b]
        stats.append(tvd(ra, rb))
    stats.sort()
    return stats[int(0.025 * N_BOOT)], stats[int(0.975 * N_BOOT) - 1]


def label(
    only_x: Sequence[int], baseline: Sequence[int], factor: str, seed: str
) -> str:
    """``large`` / ``null`` / ``small`` for one (question, factor) pair."""
    if factor in INERT:
        return "null"
    lo, hi = tvd_ci(only_x, baseline, seed)
    if lo >= LARGE_LO:
        return "large"
    if hi <= NULL_HI:
        return "null"
    return "small"


def reference_question(
    system: Any, q: Dict[str, Any], factors: Sequence[Any], n: int
) -> Dict[str, Any]:
    """Sample baseline and each only_X cell ``n`` times and label every factor."""
    base_cfg = {f.name: f.base for f in factors}

    def sample(config: Dict[str, Any]) -> List[int]:
        return [int(system.answer(config, q, i)[0] == "correct") for i in range(n)]

    base = sample(base_cfg)
    out: Dict[str, Any] = {"baseline_correct": sum(base), "n": n, "factors": {}}
    for f in factors:
        only = sample({**base_cfg, f.name: f.candidate})
        seed = f"ref-ci|{q['id']}|{f.name}"
        lo, hi = tvd_ci(only, base, seed)
        out["factors"][f.name] = {
            "only_correct": sum(only),
            "tvd": round(tvd(only, base), 4),
            "ci": [round(lo, 4), round(hi, 4)],
            "label": label(only, base, f.name, seed),
        }
    return out
