"""Reference (ground-truth) labels for E2b (PROTOCOL.md §4): two-sided.

Cells per test question: ``baseline``, ``candidate``, each ``only_<f>`` and each
``all_but_<f>``, ``n`` episodes each. For every (question, factor) two contrasts are
measured, each with a 95% percentile bootstrap CI (2,000 resamples) of the accuracy
difference:

* **alone**: ``only_f`` − ``baseline``;
* **in the release**: ``candidate`` − ``all_but_f``.

A contrast is ``neg`` if its CI upper bound is ≤ −0.20, ``pos`` if its lower bound is
≥ +0.20, ``flat`` if the whole CI lies within ±0.10, and ``unclear`` otherwise.

Pair labels: ``harmful`` (a ``neg`` and no ``pos``), ``helpful`` (a ``pos`` and no
``neg``), ``mixed`` (both), ``null`` (both flat; ``log_level`` by construction),
otherwise ``small``. The one-sided label (E2's rule) uses the "alone" contrast only.

A question is *regressed* iff the CI lower bound of baseline − candidate accuracy is
≥ 0.20. It is *key-suspect* (excluded from every criterion) iff it is not C-def, its
baseline accuracy is below 0.50, and at least 80% of its failing baseline episodes are
labelled ``other`` (right tools in order, answer not matched).
"""

from __future__ import annotations

import hashlib
import importlib.util
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

N_BOOT = 2000
LARGE = 0.20
FLAT = 0.10
REGRESSED_LO = 0.20
KEY_SUSPECT_ACC = 0.50
KEY_SUSPECT_OTHER = 0.80
N_ANSWER_SAMPLES = 3
INERT = ("log_level",)
HERE = Path(__file__).resolve().parent


def _system() -> Any:
    key = "e2b_system"
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


def diff_ci(a: Sequence[int], b: Sequence[int], seed: str) -> Tuple[float, float]:
    """95% percentile bootstrap CI of mean(a) − mean(b), resampling each sample."""
    rng = _rng(seed)
    stats = sorted(
        _mean([a[rng.randrange(len(a))] for _ in a])
        - _mean([b[rng.randrange(len(b))] for _ in b])
        for _ in range(N_BOOT)
    )
    return stats[int(0.025 * N_BOOT)], stats[int(0.975 * N_BOOT) - 1]


def contrast_kind(lo: float, hi: float) -> str:
    """``neg`` / ``pos`` / ``flat`` / ``unclear`` for a CI of an accuracy difference."""
    if hi <= -LARGE:
        return "neg"
    if lo >= LARGE:
        return "pos"
    if -FLAT <= lo and hi <= FLAT:
        return "flat"
    return "unclear"


def pair_label(alone: str, in_release: str, factor: str) -> str:
    """Two-sided label of one (question, factor) from its two contrast kinds."""
    if factor in INERT:
        return "null"
    kinds = {alone, in_release}
    if "neg" in kinds and "pos" in kinds:
        return "mixed"
    if "neg" in kinds:
        return "harmful"
    if "pos" in kinds:
        return "helpful"
    if kinds == {"flat"}:
        return "null"
    return "small"


def one_sided_label(alone: str, factor: str) -> str:
    """E2's label: the "alone" contrast only."""
    if factor in INERT:
        return "null"
    return {"neg": "harmful", "pos": "helpful", "flat": "null"}.get(alone, "small")


def regressed(baseline: Sequence[int], candidate: Sequence[int], seed: str) -> bool:
    """Baseline − candidate accuracy has a 95% CI lower bound ≥ 0.20."""
    return diff_ci(baseline, candidate, seed)[0] >= REGRESSED_LO


def key_suspect(kind: str, baseline_acc: float, baseline_faults: Dict[str, int]) -> bool:
    """Right tools, wrong match, on a question the baseline should get right."""
    failing = sum(baseline_faults.values())
    if kind == "C-def" or baseline_acc >= KEY_SUSPECT_ACC or failing == 0:
        return False
    return baseline_faults.get("other", 0) / failing >= KEY_SUSPECT_OTHER


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
        "answers": [e["final"][:300] for e in eps[:N_ANSWER_SAMPLES]],
    }


def reference_question(
    system: Any, q: Dict[str, Any], factors: Sequence[Any], n: int
) -> Dict[str, Any]:
    """Sample the 2 + 2k cells ``n`` times each; label every factor both ways."""
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
        "key_suspect": key_suspect(
            q["kind"], _mean(base), traj["baseline"]["faults"]
        ),
        "factors": {},
    }
    for f in factors:
        only = sample(f"only_{f.name}", {**base_cfg, f.name: f.candidate})
        but = sample(f"all_but_{f.name}", {**cand_cfg, f.name: f.base})
        a_lo, a_hi = diff_ci(only, base, f"ref-alone|{q['id']}|{f.name}")
        r_lo, r_hi = diff_ci(cand, but, f"ref-release|{q['id']}|{f.name}")
        alone, in_release = contrast_kind(a_lo, a_hi), contrast_kind(r_lo, r_hi)
        out["factors"][f.name] = {
            "only_correct": sum(only),
            "all_but_correct": sum(but),
            "alone_ci": [round(a_lo, 4), round(a_hi, 4)],
            "in_release_ci": [round(r_lo, 4), round(r_hi, 4)],
            "alone": alone,
            "in_release": in_release,
            "label": pair_label(alone, in_release, f.name),
            "label_one_sided": one_sided_label(alone, f.name),
        }
    out["trajectories"] = traj
    system.clear()
    return out
