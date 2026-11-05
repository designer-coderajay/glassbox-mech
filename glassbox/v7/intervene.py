# SPDX-License-Identifier: BUSL-1.1
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
# Licensed under the Business Source License 1.1 (see LICENSE-COMMERCIAL).
# Free for non-production use and for production use inside your own organisation.
# Offering it as a commercial service or product to third parties needs a license.
"""Controlled one-factor-at-a-time interventions (V7 plan §12–13, Hard Gate 4).

Specified in ``experiments/v7/intervention/PROTOCOL.md`` §2, committed before this
code. A *diff* says where two groups of runs differ; it cannot say which of several
changes caused the difference, because effects propagate downstream (Experiment 3b,
D3). This module re-runs the system with one change at a time instead.

Given factors ``(name, base, candidate)``, a user-supplied ``runner`` that runs the
system once under a configuration and records a Glassbox trace, and a user-supplied
``outcome`` that reads the behaviour of interest from a trace, it runs these cells,
running each distinct configuration once:

* ``baseline``: every factor at its base value;
* ``candidate``: every factor at its candidate value;
* ``only_<f>``: baseline with ``f`` switched to its candidate value;
* ``all_but_<f>``: candidate with ``f`` reverted to its base value.

Each cell gets ``n_per_cell`` runs with independent seeds derived from the settings
seed, the cell configuration and the run index. Outcome distributions are compared by
total-variation distance with a permutation test (as in :mod:`glassbox.v7.groupdiff`):

* total change: ``baseline`` vs ``candidate`` (unadjusted, reported);
* sufficiency (confirmatory): ``baseline`` vs ``only_<f>``, Holm across factors. A
  factor is *attributed* iff this test is significant;
* necessity (descriptive): ``candidate`` vs ``all_but_<f>``, separate Holm. With
  interactions this can be weak even for a real cause, so it does not decide
  attribution.

Scope of the claim: an attribution is intervention evidence for the tested factors,
values, inputs and outcome measure. It is not a general causal claim.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Union

from glassbox.v7.groupdiff import holm, permutation_p, tvd
from glassbox.v7.trace import canonical_json

PathLike = Union[str, "os.PathLike[str]"]
Config = Dict[str, Any]
#: ``runner(config, run_seed, out_dir, run_id) -> trace path``.
Runner = Callable[[Config, int, Path, str], PathLike]
#: ``outcome(trace_path) -> hashable outcome value``.
Outcome = Callable[[Path], Hashable]


@dataclass(frozen=True)
class Factor:
    """One configuration difference between baseline and candidate."""

    name: str
    base: Any
    candidate: Any


@dataclass(frozen=True)
class InterventionSettings:
    """Runs per cell, permutations, family-wise alpha and the master seed."""

    n_per_cell: int = 50
    n_perm: int = 999
    alpha: float = 0.05
    seed: Union[int, str] = 0


@dataclass
class CellResult:
    """A configuration that was run, with its outcome counts."""

    name: str
    config: Config
    counts: Dict[str, int]


@dataclass
class InterventionTest:
    """Comparison of the outcome distributions of two cells."""

    family: str
    factor: Optional[str]
    cell_a: str
    cell_b: str
    tvd: float
    p_value: float
    significant: bool = False


@dataclass
class InterventionReport:
    """Result of :func:`run_interventions`."""

    factors: List[Factor]
    settings: InterventionSettings
    cells: Dict[str, CellResult]
    total: InterventionTest
    sufficiency: List[InterventionTest]
    necessity: List[InterventionTest]
    paths: Dict[str, List[str]] = dc_field(default_factory=dict)

    @property
    def attributed(self) -> List[str]:
        """Factors whose sufficiency test is significant, in factor order."""
        return [t.factor for t in self.sufficiency if t.significant and t.factor]

    def rate(self, cell: str, target: str) -> float:
        """Share of runs in ``cell`` whose outcome is ``target`` (0.0 if empty)."""
        counts = self.cells[cell].counts
        total = sum(counts.values())
        return counts.get(str(target), 0) / total if total else 0.0

    def direction(self, target: str) -> Dict[str, Dict[str, Any]]:
        """Per factor: signed target-rate changes and the harmful/helpful decision.

        ``sufficiency_delta`` is ``rate(only_f) - rate(baseline)`` and
        ``necessity_delta`` is ``rate(all_but_f) - rate(candidate)``. A factor is
        *harmful* iff its sufficiency test is significant with a negative delta, or
        its necessity test is significant with a positive delta (reverting it alone
        helps). It is *helpful* iff its sufficiency test is significant with a
        positive delta and it is not harmful. The tests are the existing ones (Holm
        within each family); only the sign is added.

        Raises:
            ValueError: If ``target`` is empty.
        """
        if not target:
            raise ValueError("target outcome must be a non-empty value")
        suff = {t.factor: t for t in self.sufficiency}
        nec = {t.factor: t for t in self.necessity}
        out: Dict[str, Dict[str, Any]] = {}
        for f in self.factors:
            d_suff = self.rate(f"only_{f.name}", target) - self.rate("baseline", target)
            d_nec = self.rate(f"all_but_{f.name}", target) - self.rate(
                "candidate", target
            )
            harmful = (suff[f.name].significant and d_suff < 0) or (
                nec[f.name].significant and d_nec > 0
            )
            out[f.name] = {
                "sufficiency_delta": round(d_suff, 6),
                "necessity_delta": round(d_nec, 6),
                "harmful": bool(harmful),
                "helpful": bool(
                    not harmful and suff[f.name].significant and d_suff > 0
                ),
            }
        return out

    def harmful(self, target: str) -> List[str]:
        """Factors that lower the ``target`` rate (see :meth:`direction`)."""
        return [n for n, d in self.direction(target).items() if d["harmful"]]

    def helpful(self, target: str) -> List[str]:
        """Factors that raise the ``target`` rate and are not harmful."""
        return [n for n, d in self.direction(target).items() if d["helpful"]]

    @property
    def evidence(self) -> List[str]:
        """Plain statements of what the interventions established (and did not)."""
        out = []
        for t in self.sufficiency + self.necessity:
            verb = "changed" if t.significant else "did not detectably change"
            action = (
                f"switching {t.factor} alone from the baseline"
                if t.family == "sufficiency"
                else f"reverting {t.factor} alone in the candidate"
            )
            out.append(
                f"{action} {verb} the outcome distribution "
                f"(TVD {t.tvd:.2f}, p {t.p_value:.4f}, Holm within {t.family})"
            )
        return out

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "factors": [asdict(f) for f in self.factors],
            "settings": asdict(self.settings),
            "cells": {k: asdict(v) for k, v in self.cells.items()},
            "total": asdict(self.total),
            "sufficiency": [asdict(t) for t in self.sufficiency],
            "necessity": [asdict(t) for t in self.necessity],
            "attributed": self.attributed,
            "evidence": self.evidence,
            "paths": self.paths,
        }


def _validate(factors: Sequence[Factor], settings: InterventionSettings) -> None:
    if not factors:
        raise ValueError("need at least one factor")
    names = [f.name for f in factors]
    if len(set(names)) != len(names):
        raise ValueError("factor names must be unique")
    for f in factors:
        if canonical_json(f.base) == canonical_json(f.candidate):
            raise ValueError(f"factor {f.name}: base and candidate must differ")
    if settings.n_per_cell < 2:
        raise ValueError("n_per_cell must be at least 2")


def _cell_configs(factors: Sequence[Factor]) -> Dict[str, Config]:
    base = {f.name: f.base for f in factors}
    cand = {f.name: f.candidate for f in factors}
    cells: Dict[str, Config] = {"baseline": base, "candidate": cand}
    for f in factors:
        cells[f"only_{f.name}"] = {**base, f.name: f.candidate}
        cells[f"all_but_{f.name}"] = {**cand, f.name: f.base}
    return cells


def run_seed(master: Union[int, str], config: Config, index: int) -> int:
    """Deterministic 63-bit run seed from master seed, configuration and run index."""
    digest = hashlib.sha256(f"{master}|{canonical_json(config)}|{index}".encode())
    return int.from_bytes(digest.digest()[:8], "big") >> 1


def _run_cells(
    cells: Dict[str, Config],
    runner: Runner,
    out_dir: Path,
    settings: InterventionSettings,
) -> Dict[str, List[Path]]:
    """Run each distinct configuration once; map every cell name to its traces."""
    by_config: Dict[str, List[Path]] = {}
    paths: Dict[str, List[Path]] = {}
    for name, config in cells.items():
        key = canonical_json(config)
        if key not in by_config:
            cell_dir = out_dir / hashlib.sha256(key.encode()).hexdigest()[:12]
            by_config[key] = [
                Path(
                    runner(
                        dict(config),
                        run_seed(settings.seed, config, i),
                        cell_dir,
                        f"r{i:03d}",
                    )
                )
                for i in range(settings.n_per_cell)
            ]
        paths[name] = by_config[key]
    return paths


def _compare(
    family: str,
    factor: Optional[str],
    pair: Sequence[str],
    outcomes: Dict[str, List[Hashable]],
    settings: InterventionSettings,
) -> InterventionTest:
    a, b = outcomes[pair[0]], outcomes[pair[1]]
    seed = f"{settings.seed}:{family}:{factor}"
    return InterventionTest(
        family,
        factor,
        pair[0],
        pair[1],
        tvd(a, b),
        permutation_p(a, b, settings.n_perm, seed),
    )


def _family(
    family: str,
    factors: Sequence[Factor],
    outcomes: Dict[str, List[Hashable]],
    settings: InterventionSettings,
) -> List[InterventionTest]:
    """One test per factor for a family, Holm-corrected within the family."""
    ref, prefix = (
        ("baseline", "only_") if family == "sufficiency" else ("candidate", "all_but_")
    )
    tests = [
        _compare(family, f.name, (ref, f"{prefix}{f.name}"), outcomes, settings)
        for f in factors
    ]
    for t, rejected in zip(tests, holm([t.p_value for t in tests], settings.alpha)):
        t.significant = rejected
    return tests


def run_interventions(
    factors: Sequence[Factor],
    runner: Runner,
    outcome: Outcome,
    out_dir: PathLike,
    settings: InterventionSettings,
) -> InterventionReport:
    """Run the one-factor-at-a-time design and test which factors change the outcome.

    Raises:
        ValueError: For no factors, duplicate names, a factor whose base equals its
            candidate, or ``n_per_cell < 2``.
    """
    _validate(factors, settings)
    cells = _cell_configs(factors)
    paths = _run_cells(cells, runner, Path(out_dir), settings)
    outcomes = _read_outcomes(paths, outcome)
    total = _compare("total", None, ("baseline", "candidate"), outcomes, settings)
    total.significant = total.p_value <= settings.alpha
    suff = _family("sufficiency", factors, outcomes, settings)
    nec = _family("necessity", factors, outcomes, settings)
    counts = {
        name: CellResult(
            name,
            dict(cells[name]),
            dict(sorted((str(k), v) for k, v in _count(outcomes[name]).items())),
        )
        for name in cells
    }
    return InterventionReport(
        list(factors),
        settings,
        counts,
        total,
        suff,
        nec,
        {name: [str(p) for p in ps] for name, ps in paths.items()},
    )


def _read_outcomes(
    paths: Dict[str, List[Path]], outcome: Outcome
) -> Dict[str, List[Hashable]]:
    """Outcome per run, reading each trace once even if cells share it."""
    cache: Dict[Path, Hashable] = {}
    for ps in paths.values():
        for p in ps:
            if p not in cache:
                cache[p] = outcome(p)
    return {name: [cache[p] for p in ps] for name, ps in paths.items()}


def _count(values: Sequence[Hashable]) -> Dict[Hashable, int]:
    out: Dict[Hashable, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out
