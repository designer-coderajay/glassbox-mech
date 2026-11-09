"""Reference (ground-truth) labels for E2 (PROTOCOL.md §4). The labelling rules are
E1b's, unchanged (E1b PROTOCOL §4); only the system and the stored extras differ.

Cells per test question: ``baseline``, ``candidate`` and each ``only_<f>``, ``n``
episodes each. Outcomes are binary, so TVD equals the absolute accuracy difference.
Labels per (question, factor), from a 95% percentile bootstrap CI (2,000 resamples):

* ``harmful``: CI lower bound ≥ 0.20 and only_f accuracy < baseline accuracy;
* ``helpful``: CI lower bound ≥ 0.20 and only_f accuracy > baseline accuracy;
* ``null``: CI upper bound ≤ 0.10 (``log_level`` is null by construction);
* ``small`` otherwise (reported, not scored).

A question is *regressed* iff the CI lower bound of baseline − candidate accuracy is
≥ 0.20. Per cell, the fault labels of failing episodes, the forced-final count and
the parse failures are stored (trajectory truth; reported, not scored).
"""

from __future__ import annotations

import hashlib
import importlib.util
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

N_BOOT = 2000
LARGE_LO = 0.20
NULL_HI = 0.10
REGRESSED_LO = 0.20
INERT = ("log_level",)
HERE = Path(__file__).resolve().parent


def _system() -> Any:
    key = "e2_system"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / "system.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def _rng(seed: str) -> random.Random:
    return random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))


def _mean(x: Sequence[int]) -> float:
    return sum(x) / len(x)


def tvd(a: Sequence[int], b: Sequence[int]) -> float:
    """TVD of two binary samples (1 = correct)."""
    return abs(_mean(a) - _mean(b))


def _boot(a: Sequence[int], b: Sequence[int], seed: str, signed: bool) -> List[float]:
    rng = _rng(seed)
    stats = []
    for _ in range(N_BOOT):
        d = _mean([a[rng.randrange(len(a))] for _ in a]) - _mean(
            [b[rng.randrange(len(b))] for _ in b]
        )
        stats.append(d if signed else abs(d))
    return sorted(stats)


def tvd_ci(a: Sequence[int], b: Sequence[int], seed: str) -> Tuple[float, float]:
    """95% percentile bootstrap CI of the TVD, resampling each sample independently."""
    s = _boot(a, b, seed, signed=False)
    return s[int(0.025 * N_BOOT)], s[int(0.975 * N_BOOT) - 1]


def diff_ci(a: Sequence[int], b: Sequence[int], seed: str) -> Tuple[float, float]:
    """95% percentile bootstrap CI of mean(a) − mean(b)."""
    s = _boot(a, b, seed, signed=True)
    return s[int(0.025 * N_BOOT)], s[int(0.975 * N_BOOT) - 1]


def label(
    only_x: Sequence[int], baseline: Sequence[int], factor: str, seed: str
) -> str:
    """``harmful`` / ``helpful`` / ``null`` / ``small`` for one (question, factor)."""
    if factor in INERT:
        return "null"
    lo, hi = tvd_ci(only_x, baseline, seed)
    if lo >= LARGE_LO:
        return "harmful" if _mean(only_x) < _mean(baseline) else "helpful"
    if hi <= NULL_HI:
        return "null"
    return "small"


def regressed(baseline: Sequence[int], candidate: Sequence[int], seed: str) -> bool:
    """Baseline − candidate accuracy has a 95% CI lower bound ≥ 0.20."""
    return diff_ci(baseline, candidate, seed)[0] >= REGRESSED_LO


def _trajectories(eps: Sequence[Dict[str, Any]], required: Sequence[str]):
    from glassbox.v7.taxonomy import FAULTS, label_episode

    faults = {f: 0 for f in FAULTS}
    for e in eps:
        lab = label_episode(e["calls"], required, e["hit_limit"],
                            e["label"] == "correct")
        if lab is not None:
            faults[lab] += 1
    return {
        "faults": faults,
        "forced_final": sum(bool(e["hit_limit"]) for e in eps),
        "parse_failures": sum(int(e["parse_failures"]) for e in eps),
    }


def reference_question(
    system: Any, q: Dict[str, Any], factors: Sequence[Any], n: int
) -> Dict[str, Any]:
    """Sample baseline, candidate and each only_f ``n`` times; label everything."""
    required = _system().REQUIRED[q["kind"]]
    base_cfg = {f.name: f.base for f in factors}
    cand_cfg = {f.name: f.candidate for f in factors}
    traj: Dict[str, Any] = {}

    def sample(name: str, config: Dict[str, Any]) -> List[int]:
        eps = [system.episode(config, q, i) for i in range(n)]
        traj[name] = _trajectories(eps, required)
        return [int(e["label"] == "correct") for e in eps]

    base, cand = sample("baseline", base_cfg), sample("candidate", cand_cfg)
    reg_seed = f"ref-reg|{q['id']}"
    out: Dict[str, Any] = {
        "n": n,
        "baseline_correct": sum(base),
        "candidate_correct": sum(cand),
        "regressed_ci": [round(v, 4) for v in diff_ci(base, cand, reg_seed)],
        "regressed": regressed(base, cand, reg_seed),
        "factors": {},
    }
    for f in factors:
        only = sample(f"only_{f.name}", {**base_cfg, f.name: f.candidate})
        seed = f"ref-ci|{q['id']}|{f.name}"
        lo, hi = tvd_ci(only, base, seed)
        out["factors"][f.name] = {
            "only_correct": sum(only),
            "tvd": round(tvd(only, base), 4),
            "ci": [round(lo, 4), round(hi, 4)],
            "label": label(only, base, f.name, seed),
        }
    out["trajectories"] = traj
    system.clear()
    return out
