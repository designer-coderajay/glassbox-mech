# SPDX-License-Identifier: BUSL-1.1
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
# Licensed under the Business Source License 1.1 (see LICENSE-COMMERCIAL).
# Free for non-production use and for production use inside your own organisation.
# Offering it as a commercial service or product to third parties needs a license.
"""Agent fault labeller: which kind of mistake did a failing agent episode make?

Given the tools an agent called (in order), whether it hit its step limit, and the
tools the task requires (in order), a **failing** episode gets exactly one label. The
first matching rule wins (V8 E2 PROTOCOL §5.1):

1. ``no_action``: no tool was called;
2. ``premature_finish``: the step limit was reached and a required tool was never
   called;
3. ``wrong_tool``: a tool outside the required set was called and a required tool was
   never called;
4. ``skipped_tool``: the episode ended on its own, every call was to a required tool,
   and a required tool was never called;
5. ``wrong_order``: every required tool was called, but their first calls are out of
   the required order;
6. ``other``: anything else (right tools in the right order with a wrong answer, tool
   errors, unparsable output, ...).

A correct episode gets no label. The label describes the trajectory, not its cause,
and it depends on the required sequence the task author wrote.

Traces: tool calls are spans with ``gen_ai.operation.name = execute_tool`` and
``gen_ai.tool.name``; reaching the step limit is marked by any span with
``glassbox.agent.step_limit_reached = true``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple, Union

from glassbox.v7.trace import read_trace

FAULTS = (
    "no_action",
    "premature_finish",
    "wrong_tool",
    "skipped_tool",
    "wrong_order",
    "other",
)
PathLike = Union[str, Path]


def label_episode(
    calls: Sequence[str], required: Sequence[str], hit_limit: bool, correct: bool
) -> Optional[str]:
    """Fault label of one episode, or ``None`` if it was correct.

    Raises:
        ValueError: If ``required`` is empty.
    """
    if not required:
        raise ValueError("required must name at least one tool")
    if correct:
        return None
    if not calls:
        return "no_action"
    req = set(required)
    missing = any(t not in calls for t in required)
    if hit_limit and missing:
        return "premature_finish"
    if missing and any(c not in req for c in calls):
        return "wrong_tool"
    if missing:
        return "skipped_tool"
    first = sorted(required, key=calls.index)
    if first != list(required):
        return "wrong_order"
    return "other"


def episode_from_trace(path: PathLike) -> Tuple[List[str], bool]:
    """(tool names in call order, step limit reached) from one agent trace."""
    spans, _ = read_trace(path)
    calls = [
        str(s["attributes"]["gen_ai.tool.name"])
        for s in spans
        if s["attributes"].get("gen_ai.operation.name") == "execute_tool"
    ]
    limit = any(
        bool(s["attributes"].get("glassbox.agent.step_limit_reached")) for s in spans
    )
    return calls, limit


def label_trace(
    path: PathLike,
    required: Sequence[str],
    outcome: Callable[[PathLike], Hashable],
    target: Hashable,
) -> Optional[str]:
    """Label of the episode in ``path``; correct means ``outcome(path) == target``."""
    calls, limit = episode_from_trace(path)
    return label_episode(calls, required, limit, outcome(path) == target)


def count_faults(
    paths: Sequence[PathLike],
    required: Sequence[str],
    outcome: Callable[[PathLike], Hashable],
    target: Hashable,
) -> Dict[str, int]:
    """Counts of each fault label over ``paths`` (correct episodes not counted)."""
    counts = {f: 0 for f in FAULTS}
    for p in paths:
        lab = label_trace(p, required, outcome, target)
        if lab is not None:
            counts[lab] += 1
    return counts


def dominant(counts: Dict[str, int], min_failing: int = 5) -> Optional[str]:
    """Most common label, or ``None`` below ``min_failing`` failing episodes.

    Ties go to the label listed first in :data:`FAULTS`.
    """
    if sum(counts.values()) < min_failing:
        return None
    return max(FAULTS, key=lambda f: (counts.get(f, 0), -FAULTS.index(f)))
