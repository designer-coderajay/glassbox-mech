"""V6 Claim / Finding schema with an explicit evidence ladder.

A :class:`Finding` is what an experiment run produces: a set of :class:`Measurement`
objects (numbers with provenance) and one :class:`HypothesisResult` per pre-registered
hypothesis. A :class:`Claim` is a single statement promoted out of a Finding; it can
never carry a stronger status than the evidence recorded for it.

``UNRESOLVED`` is a first-class state and the default for every hypothesis: a result
that has not been tested against its pre-registered null is not "negative", it is
unresolved.

This ladder is a separate concept from :class:`glassbox.evidence_tier.EvidenceTier`
(tiers A-D for single-model circuit audits) and does not replace it.
"""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "v6.claims/1.0.0"

__all__ = [
    "SCHEMA_VERSION",
    "EvidenceStatus",
    "Scope",
    "Measurement",
    "HypothesisResult",
    "Claim",
    "Finding",
    "canonical_json",
]


class EvidenceStatus(str, enum.Enum):
    """Evidence ladder, weakest to strongest, plus the two terminal side states."""

    UNRESOLVED = "UNRESOLVED"
    REFUTED = "REFUTED"
    OBSERVED = "OBSERVED"
    ASSOCIATED = "ASSOCIATED"
    INTERVENED = "INTERVENED"
    CAUSALLY_SUPPORTED = "CAUSALLY_SUPPORTED"
    SURVIVED_FALSIFICATION = "SURVIVED_FALSIFICATION"
    REPRODUCED = "REPRODUCED"
    EXTERNALLY_REPRODUCED = "EXTERNALLY_REPRODUCED"

    @property
    def rank(self) -> int:
        """Position on the ladder; UNRESOLVED and REFUTED rank below OBSERVED."""
        return _RANK[self]


_RANK = {s: i for i, s in enumerate(EvidenceStatus)}
_UNRESOLVED_OR_REFUTED = {EvidenceStatus.UNRESOLVED, EvidenceStatus.REFUTED}


def canonical_json(obj: Any) -> str:
    """Deterministic JSON (sorted keys, no whitespace, NaN encoded as null)."""
    return json.dumps(_nan_to_none(obj), sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=str)


def _nan_to_none(obj: Any) -> Any:
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    if isinstance(obj, dict):
        return {str(k): _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_none(v) for v in obj]
    return obj


@dataclasses.dataclass(frozen=True)
class Scope:
    """Where a result is claimed to hold. Every field is required; no global claims."""

    task: str
    models: List[str]
    dataset_hash: str
    metric_definitions: str

    def __post_init__(self) -> None:
        if not self.task or not self.models or not self.dataset_hash:
            raise ValueError("Scope requires task, models and dataset_hash")


@dataclasses.dataclass
class Measurement:
    """One measured quantity. Status is at most OBSERVED; a measurement is not a claim."""

    name: str
    value: Optional[float]
    details: Dict[str, Any] = dataclasses.field(default_factory=dict)
    status: EvidenceStatus = EvidenceStatus.OBSERVED
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status.rank > EvidenceStatus.OBSERVED.rank:
            raise ValueError("a Measurement cannot be stronger than OBSERVED")
        if self.value is None or self.value != self.value:  # None or NaN
            self.value = None
            if self.status is EvidenceStatus.OBSERVED:
                self.status = EvidenceStatus.UNRESOLVED
            if not self.reason:
                raise ValueError(f"measurement {self.name!r} has no value and no reason")


@dataclasses.dataclass
class HypothesisResult:
    """Result for one pre-registered hypothesis (H1-H4)."""

    hypothesis: str
    estimand: str
    null: str
    test: str
    status: EvidenceStatus = EvidenceStatus.UNRESOLVED
    reason: str = "not tested in this run"
    effect_size: Optional[float] = None
    ci: Optional[List[float]] = None
    p_value: Optional[float] = None
    control: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status in _UNRESOLVED_OR_REFUTED and not self.reason:
            raise ValueError("UNRESOLVED/REFUTED results must state a reason")
        if self.status not in _UNRESOLVED_OR_REFUTED and (
            self.p_value is None or self.effect_size is None or self.ci is None
        ):
            raise ValueError(
                f"{self.hypothesis}: a tested result needs effect_size, ci and p_value"
            )


@dataclasses.dataclass
class Claim:
    """A statement promoted from a Finding, bounded by its evidence and scope."""

    statement: str
    status: EvidenceStatus
    scope: Scope
    evidence: List[str]
    assumptions: List[str] = dataclasses.field(default_factory=list)
    gate_passed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.status not in _UNRESOLVED_OR_REFUTED and not self.evidence:
            raise ValueError("a Claim above UNRESOLVED needs evidence references")
        if self.status is EvidenceStatus.CAUSALLY_SUPPORTED and not self.assumptions:
            raise ValueError("CAUSALLY_SUPPORTED requires stated assumptions")
        if self.status.rank >= EvidenceStatus.SURVIVED_FALSIFICATION.rank and (
            self.gate_passed is None or self.gate_passed < 3
        ):
            raise ValueError("SURVIVED_FALSIFICATION or above requires Gate 3 passed")


@dataclasses.dataclass
class Finding:
    """Output of one experiment run: measurements, hypotheses, provenance link."""

    run_id: str
    scope: Scope
    measurements: List[Measurement]
    hypotheses: List[HypothesisResult]
    controls: Dict[str, Any]
    record_hash: str
    label: str = "smoke"  # smoke | pilot | confirmatory
    claims: List[Claim] = dataclasses.field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.label not in {"smoke", "pilot", "confirmatory"}:
            raise ValueError("label must be smoke, pilot or confirmatory")
        if self.label != "confirmatory":
            for h in self.hypotheses:
                if h.status not in _UNRESOLVED_OR_REFUTED:
                    raise ValueError(
                        f"{h.hypothesis}: only a confirmatory run may resolve a hypothesis"
                    )

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict form with enums as strings."""
        return json.loads(canonical_json(dataclasses.asdict(self)))

    def content_hash(self) -> str:
        """sha256 of the canonical JSON form."""
        return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()
