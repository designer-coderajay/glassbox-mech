"""Lifecycle of an async audit job, in one place.

A job moves ``queued → running → completed`` or to ``failed`` (from ``queued`` or
``running``). ``completed`` and ``failed`` are terminal: nothing moves out of them.
Every status change goes through :func:`advance`, which refuses any other move, and
every response about a job is built by :func:`job_view`, so each endpoint shows the
same fields.

This module has no web-framework imports; ``api/main.py`` uses it.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, FrozenSet, Optional

__all__ = [
    "JobStatus",
    "TERMINAL",
    "TRANSITIONS",
    "new_job",
    "advance",
    "set_progress",
    "job_view",
]


class JobStatus(str, Enum):
    """The four states of an async job. Values are the strings the API returns."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL: FrozenSet[JobStatus] = frozenset({JobStatus.COMPLETED, JobStatus.FAILED})

#: Allowed moves. Anything else raises in :func:`advance`.
TRANSITIONS: Dict[JobStatus, FrozenSet[JobStatus]] = {
    JobStatus.QUEUED: frozenset({JobStatus.RUNNING, JobStatus.FAILED}),
    JobStatus.RUNNING: frozenset({JobStatus.COMPLETED, JobStatus.FAILED}),
    JobStatus.COMPLETED: frozenset(),
    JobStatus.FAILED: frozenset(),
}

_NEXT = {
    JobStatus.QUEUED: "poll",
    JobStatus.RUNNING: "poll",
    JobStatus.COMPLETED: "fetch_report",
    JobStatus.FAILED: "none",
}


def new_job(now: float) -> Dict[str, Any]:
    """A job record in ``queued``, with every field present from the start."""
    return {
        "status": JobStatus.QUEUED.value,
        "report_id": None,
        "error": None,
        "created_at": now,
        "percent": 0,
        "message": "Queued",
    }


def advance(job: Dict[str, Any], status: JobStatus, **fields: Any) -> Dict[str, Any]:
    """Move ``job`` to ``status`` and update ``fields``.

    Raises:
        ValueError: If the move is not in :data:`TRANSITIONS` (for example out of a
            terminal state), or ``completed`` is requested without a ``report_id``.
    """
    current = JobStatus(job["status"])
    target = JobStatus(status)
    if target not in TRANSITIONS[current]:
        raise ValueError(f"job cannot move from {current.value} to {target.value}")
    if target is JobStatus.COMPLETED and not (
        fields.get("report_id") or job.get("report_id")
    ):
        raise ValueError("a completed job needs a report_id")
    job.update(fields)
    job["status"] = target.value
    if target is JobStatus.COMPLETED:
        job["percent"] = 100
    return job


def set_progress(job: Dict[str, Any], percent: int, message: str) -> Dict[str, Any]:
    """Update progress of a running job. Ignored once the job is terminal."""
    if JobStatus(job["status"]) is JobStatus.RUNNING:
        job["percent"] = max(0, min(99, int(percent)))
        job["message"] = message
    return job


def job_view(
    job_id: str, job: Dict[str, Any], now: float, has_pdf: bool = False
) -> Dict[str, Any]:
    """The response for one job.

    The fields returned before this module existed are unchanged (``job_id``,
    ``status``, ``created_at``, ``elapsed``; ``report_id``, ``report_url`` and
    ``pdf_url`` when completed; ``error`` when failed). Added: ``percent``,
    ``message``, ``terminal`` and ``next`` (``poll``, ``fetch_report`` or ``none``).
    """
    status = JobStatus(job["status"])
    view: Dict[str, Any] = {
        "job_id": job_id,
        "status": status.value,
        "created_at": job["created_at"],
        "elapsed": round(now - job["created_at"], 1),
        "percent": int(job.get("percent", 0)),
        "message": job.get("message", ""),
        "terminal": status in TERMINAL,
        "next": _NEXT[status],
    }
    if status is JobStatus.COMPLETED:
        report_id: Optional[str] = job["report_id"]
        view["report_id"] = report_id
        view["report_url"] = f"/v1/audit/report/{report_id}"
        view["pdf_url"] = f"/v1/audit/pdf/{report_id}" if has_pdf else None
    if status is JobStatus.FAILED:
        view["error"] = job["error"]
    return view
