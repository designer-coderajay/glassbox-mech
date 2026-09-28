"""Deterministic, versioned IOI task data and D_B probe set (Arm A).

No torch here. Tokenizer-dependent filtering is injected as a callable, so the same
generator is used for every model and the dataset hash records which tokenizer filter
was applied.

IOI item: ``"When {A} and {B} went to the {place}, {B} gave {obj} to"`` with the correct
answer `` A`` (the indirect object) and distractor `` B`` (the subject).

Probe set for D_B (outputs only; one probe of each kind per item):
  * ``counterfactual`` - the second-clause subject becomes A, so the correct answer flips to B;
  * ``feature_swap``   - the first-clause name order is swapped; the answer is unchanged;
  * ``perturbation``   - place and object are replaced (seeded); the answer is unchanged.
"""
from __future__ import annotations

import dataclasses
import random
from typing import Callable, Dict, List, Optional

from glassbox.v6.provenance import sha256_json

TASK_VERSION = "v6.tasks.ioi/1.0.0"

__all__ = ["TASK_VERSION", "IOIItem", "Probe", "IOIDataset", "build_ioi"]

NAMES = [
    "Mary", "John", "Sarah", "Tom", "Alice", "Bob", "Emma", "James", "Lisa", "Mark",
    "Anna", "David", "Laura", "Paul", "Kate", "Peter", "Julia", "Michael", "Rachel",
    "Daniel", "Jessica", "Robert", "Linda", "George", "Susan", "Frank", "Karen", "Steve",
    "Helen", "Kevin", "Amy", "Brian", "Nancy", "Jack", "Emily", "Adam", "Grace", "Ryan",
]
PLACES = ["store", "park", "office", "school", "station", "garden", "restaurant",
          "hospital", "library", "market", "beach", "house"]
OBJECTS = ["a bottle", "a book", "the ball", "a drink", "the key", "a letter",
           "a ring", "the bag", "a gift", "the phone", "a ticket", "the box"]
TEMPLATE = "When {a} and {b} went to the {place}, {c} gave {obj} to"


@dataclasses.dataclass(frozen=True)
class IOIItem:
    """One IOI prompt. ``target``/``distractor`` carry the leading space."""

    prompt: str
    target: str
    distractor: str
    name_a: str
    name_b: str
    place: str
    obj: str


@dataclasses.dataclass(frozen=True)
class Probe:
    """One D_B probe prompt (the answer is recorded for reference only)."""

    kind: str
    prompt: str
    expected: str
    item_index: int


@dataclasses.dataclass(frozen=True)
class IOIDataset:
    """Items, probes and the hash that identifies them."""

    items: List[IOIItem]
    probes: List[Probe]
    seed: int
    tokenizer_id: Optional[str]
    dataset_hash: str
    version: str = TASK_VERSION


def _prompt(a: str, b: str, c: str, place: str, obj: str) -> str:
    return TEMPLATE.format(a=a, b=b, c=c, place=place, obj=obj)


def build_ioi(
    n_items: int,
    seed: int = 0,
    single_token: Optional[Callable[[str], bool]] = None,
    tokenizer_id: Optional[str] = None,
) -> IOIDataset:
    """Generate ``n_items`` IOI items and 3 probes per item, deterministically.

    Args:
        n_items: Number of IOI items (> 0).
        seed: RNG seed; same seed + same filter -> identical dataset and hash.
        single_token: Predicate on ``" Name"`` strings; names failing it are dropped so
            that name swaps never change sequence length. ``None`` keeps all names.
        tokenizer_id: Recorded in the hash (e.g. ``"EleutherAI/pythia-70m"``).

    Raises:
        ValueError: if fewer than 2 names survive the filter or ``n_items`` < 1.
    """
    if n_items < 1:
        raise ValueError("n_items must be >= 1")
    names = [n for n in NAMES if single_token is None or single_token(" " + n)]
    if len(names) < 2:
        raise ValueError("fewer than 2 single-token names for this tokenizer")
    rng = random.Random(seed)
    items: List[IOIItem] = []
    probes: List[Probe] = []
    for i in range(n_items):
        a, b = rng.sample(names, 2)
        place, obj = rng.choice(PLACES), rng.choice(OBJECTS)
        item = IOIItem(_prompt(a, b, b, place, obj), " " + a, " " + b, a, b, place, obj)
        items.append(item)
        p2 = rng.choice([p for p in PLACES if p != place])
        o2 = rng.choice([o for o in OBJECTS if o != obj])
        probes.extend([
            Probe("counterfactual", _prompt(a, b, a, place, obj), " " + b, i),
            Probe("feature_swap", _prompt(b, a, b, place, obj), " " + a, i),
            Probe("perturbation", _prompt(a, b, b, p2, o2), " " + a, i),
        ])
    payload: Dict[str, object] = {
        "version": TASK_VERSION, "seed": seed, "tokenizer_id": tokenizer_id,
        "names": names,
        "items": [dataclasses.asdict(x) for x in items],
        "probes": [dataclasses.asdict(p) for p in probes],
    }
    return IOIDataset(items, probes, seed, tokenizer_id, sha256_json(payload))
