"""Noise-aware group diff: which steps differ beyond run-to-run variation?

Specified in ``experiments/v7/noise_baseline/PROTOCOL.md`` §2 (committed before this
code). The pairwise diff (:mod:`glassbox.v7.diff`) compares one run with one run; in a
system that samples, two runs with the same configuration already differ. This module
compares a *group* A of baseline runs with a group B of candidate runs and reports a
step only when its behaviour differs beyond the variation between runs.

Method:
    1. Each non-root span gets the key ``<signature>#<k>``: the pairwise diff's step
       signature, plus its occurrence count earlier in the same run.
    2. Fields per step key: ``present``, ``status``, every comparable attribute (same
       exclusions as the pairwise diff) and every content hash. A step absent from a
       run gives ``present=False`` and ``None`` for its other fields.
    3. Each field, pooled over A ∪ B (n runs), is *constant* (one value), *untestable*
       (more than n/2 distinct values, e.g. hashes of sampled free text), or *tested*.
    4. Tested fields: total-variation distance between the A and B value distributions,
       with a seeded permutation p-value ``(1 + #{TVD_perm >= TVD_obs}) / (1 + n_perm)``.
    5. Holm–Bonferroni across the tested fields of one comparison at ``alpha``.

Scope of the claim: a significant step says **where** the groups' behaviour differs
beyond noise, never **why** and never **how many causes** there are. A change upstream
propagates to downstream steps, so one cause and two causes can look the same.
Numeric attributes are treated as categorical values in this version.

Usage:
    python -m glassbox.v7.groupdiff --a base1.trace.jsonl base2... --b cand1... [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple, Union

from glassbox.v7.diff import _comparable, step_signature
from glassbox.v7.trace import canonical_json, read_trace

PathLike = Union[str, "os.PathLike[str]"]
FieldKey = Tuple[str, str]
Features = Dict[FieldKey, Hashable]

_HASH_PREFIX_LEN = 12


@dataclass
class FieldResult:
    """Outcome for one (step key, field) pair.

    ``classification`` is ``constant``, ``untestable`` or ``tested``. ``dist_a`` and
    ``dist_b`` give value counts (values as strings; content hashes are shortened to
    12-character prefixes), and are left empty for untestable fields.
    """

    step: str
    field: str
    position: float
    classification: str
    varies_in_baseline: bool
    tvd: Optional[float] = None
    p_value: Optional[float] = None
    significant: bool = False
    n_distinct: int = 0
    dist_a: Dict[str, int] = dc_field(default_factory=dict)
    dist_b: Dict[str, int] = dc_field(default_factory=dict)


@dataclass
class GroupDiffReport:
    """Result of :func:`group_diff`.

    ``fields`` are in run order. ``environment`` maps root-span attributes whose value
    sets differ between the groups to ``(values_a, values_b)``. These are reported
    separately and never tested.
    """

    fields: List[FieldResult]
    environment: Dict[str, Tuple[List[str], List[str]]]
    n_a: int
    n_b: int
    alpha: float
    n_perm: int

    @property
    def significant(self) -> List[FieldResult]:
        """All fields that differ beyond noise, in run order."""
        return [f for f in self.fields if f.significant]

    @property
    def first_beyond_noise(self) -> Optional[FieldResult]:
        """Earliest significant field, or ``None``."""
        sig = self.significant
        return sig[0] if sig else None

    @property
    def untestable(self) -> List[FieldKey]:
        """Fields with too many distinct values to separate from noise."""
        return [
            (f.step, f.field) for f in self.fields if f.classification == "untestable"
        ]

    @property
    def noisy_in_baseline(self) -> List[FieldKey]:
        """Fields that already vary between the baseline runs."""
        return [(f.step, f.field) for f in self.fields if f.varies_in_baseline]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        first = self.first_beyond_noise
        return {
            "n_a": self.n_a,
            "n_b": self.n_b,
            "alpha": self.alpha,
            "n_perm": self.n_perm,
            "first_beyond_noise": asdict(first) if first else None,
            "significant": [asdict(f) for f in self.significant],
            "untestable": [list(k) for k in self.untestable],
            "noisy_in_baseline": [list(k) for k in self.noisy_in_baseline],
            "fields": [asdict(f) for f in self.fields],
            "environment": {k: [a, b] for k, (a, b) in self.environment.items()},
        }


# --- statistics -----------------------------------------------------------------


def tvd(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Total-variation distance between the empirical distributions of ``a`` and ``b``."""
    ca, cb = Counter(a), Counter(b)
    na, nb = len(a), len(b)
    return 0.5 * sum(abs(ca[k] / na - cb[k] / nb) for k in set(ca) | set(cb))


def permutation_p(
    a: Sequence[Hashable], b: Sequence[Hashable], n_perm: int, seed: str
) -> float:
    """Permutation p-value for ``tvd(a, b)`` with group labels reshuffled ``n_perm`` times."""
    observed = tvd(a, b)
    pooled = list(a) + list(b)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if tvd(pooled[: len(a)], pooled[len(a) :]) >= observed - 1e-12:
            hits += 1
    return (1 + hits) / (1 + n_perm)


def holm(p_values: Sequence[float], alpha: float) -> List[bool]:
    """Holm–Bonferroni step-down: which hypotheses are rejected at family-wise ``alpha``."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    reject = [False] * m
    for rank, i in enumerate(order):
        if p_values[i] > alpha / (m - rank):
            break
        reject[i] = True
    return reject


# --- feature extraction ---------------------------------------------------------


def _freeze(value: Any) -> Hashable:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return canonical_json(value)


def _features(path: PathLike) -> Tuple[Features, Dict[str, int], Dict[str, Any]]:
    """``(fields, step index per key, comparable root attributes)`` for one trace."""
    spans, _ = read_trace(path)
    root, steps = spans[0], spans[1:]
    seen: Counter = Counter()  # type: ignore[type-arg]
    feats: Features = {}
    index: Dict[str, int] = {}
    for span in steps:
        sig = step_signature(span)
        key = f"{sig}#{seen[sig]}"
        seen[sig] += 1
        index[key] = span["attributes"]["glassbox.step.index"]
        feats[(key, "present")] = True
        feats[(key, "status")] = span["status"]["code"]
        for attr, value in _comparable(span["attributes"]).items():
            if attr != "gen_ai.operation.name":
                feats[(key, attr)] = _freeze(value)
    env = {
        k: _freeze(v)
        for k, v in _comparable(root["attributes"]).items()
        if k != "gen_ai.operation.name"
    }
    return feats, index, env


def _value(feats: Features, key: FieldKey) -> Hashable:
    if key in feats:
        return feats[key]
    return False if key[1] == "present" else None


def _label(field_name: str, value: Hashable) -> str:
    text = str(value)
    if field_name.endswith(".sha256") and value is not None:
        return text[:_HASH_PREFIX_LEN]
    return text


# --- comparison -----------------------------------------------------------------


def _classify(values_a: List[Hashable], values_b: List[Hashable]) -> Tuple[str, int]:
    distinct = len(set(values_a) | set(values_b))
    if distinct == 1:
        return "constant", distinct
    if distinct > (len(values_a) + len(values_b)) / 2:
        return "untestable", distinct
    return "tested", distinct


def _field_result(
    key: FieldKey, va: List[Hashable], vb: List[Hashable], position: float
) -> FieldResult:
    classification, distinct = _classify(va, vb)
    result = FieldResult(
        step=key[0],
        field=key[1],
        position=position,
        classification=classification,
        varies_in_baseline=len(set(va)) > 1,
        n_distinct=distinct,
    )
    if classification != "untestable":
        result.dist_a = dict(sorted(Counter(_label(key[1], v) for v in va).items()))
        result.dist_b = dict(sorted(Counter(_label(key[1], v) for v in vb).items()))
    return result


def _environment(
    envs_a: List[Dict[str, Any]], envs_b: List[Dict[str, Any]]
) -> Dict[str, Tuple[List[str], List[str]]]:
    keys = sorted({k for e in envs_a + envs_b for k in e})
    out: Dict[str, Tuple[List[str], List[str]]] = {}
    for k in keys:
        sa = sorted({str(e.get(k)) for e in envs_a})
        sb = sorted({str(e.get(k)) for e in envs_b})
        if sa != sb:
            out[k] = (sa, sb)
    return out


def _positions(indexes: List[Dict[str, int]]) -> Dict[str, float]:
    sums: Dict[str, List[int]] = {}
    for idx in indexes:
        for step, i in idx.items():
            sums.setdefault(step, []).append(i)
    return {s: sum(v) / len(v) for s, v in sums.items()}


def group_diff(
    paths_a: Sequence[PathLike],
    paths_b: Sequence[PathLike],
    n_perm: int = 999,
    alpha: float = 0.05,
    seed: int = 0,
) -> GroupDiffReport:
    """Compare a baseline group of traces with a candidate group, beyond noise.

    Raises:
        ValueError: If either group has fewer than two runs.
        TraceIntegrityError: If any trace fails its integrity check.
    """
    if len(paths_a) < 2 or len(paths_b) < 2:
        raise ValueError("each group needs at least two runs")
    parsed_a = [_features(p) for p in paths_a]
    parsed_b = [_features(p) for p in paths_b]
    positions = _positions([x[1] for x in parsed_a + parsed_b])
    keys = sorted(
        {k for f, _, _ in parsed_a + parsed_b for k in f},
        key=lambda k: (positions[k[0]], k[0], k[1] != "present", k[1]),
    )
    results: List[FieldResult] = []
    for key in keys:
        va = [_value(f, key) for f, _, _ in parsed_a]
        vb = [_value(f, key) for f, _, _ in parsed_b]
        res = _field_result(key, va, vb, positions[key[0]])
        if res.classification == "tested":
            res.tvd = tvd(va, vb)
            res.p_value = permutation_p(va, vb, n_perm, f"{seed}:{key[0]}:{key[1]}")
        results.append(res)
    tested = [r for r in results if r.classification == "tested"]
    for r, rejected in zip(tested, holm([r.p_value or 1.0 for r in tested], alpha)):
        r.significant = rejected
    env = _environment([x[2] for x in parsed_a], [x[2] for x in parsed_b])
    return GroupDiffReport(results, env, len(paths_a), len(paths_b), alpha, n_perm)


# --- CLI ------------------------------------------------------------------------


def _describe(f: FieldResult) -> str:
    return (
        f"{f.step} {f.field}: TVD {f.tvd:.2f}, p {f.p_value:.4f} "
        f"(A {f.dist_a} vs B {f.dist_b})"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: exit 0 if no step differs beyond noise, 1 otherwise."""
    parser = argparse.ArgumentParser(description="Noise-aware group diff of traces.")
    parser.add_argument("--a", nargs="+", required=True, help="baseline traces")
    parser.add_argument("--b", nargs="+", required=True, help="candidate traces")
    parser.add_argument("--n-perm", type=int, default=999)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="print the full report")
    args = parser.parse_args(argv)
    report = group_diff(args.a, args.b, args.n_perm, args.alpha, args.seed)
    code = 1 if report.significant else 0
    if args.json:
        sys.stdout.write(json.dumps(report.to_dict(), indent=2) + "\n")
        return code
    out = sys.stdout
    out.write(f"Groups: A={report.n_a} runs, B={report.n_b} runs\n")
    if not report.significant:
        out.write("No step differs beyond run-to-run noise.\n")
    for i, f in enumerate(report.significant):
        out.write(
            ("First beyond noise: " if i == 0 else "  also: ") + _describe(f) + "\n"
        )
    for step, name in report.untestable:
        out.write(f"Untestable (too many distinct values): {step} {name}\n")
    for k, (va, vb) in report.environment.items():
        out.write(f"Environment differs: {k}: {va} → {vb}\n")
    out.write(
        "Note: beyond-noise says where groups differ, not why or how many causes.\n"
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
