"""Async-job lifecycle (api/jobs.py) and its use in api/main.py."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def J():
    spec = importlib.util.spec_from_file_location("api_jobs_t", ROOT / "api" / "jobs.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["api_jobs_t"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_states_and_transition_table(J):
    S = J.JobStatus
    assert [s.value for s in S] == ["queued", "running", "completed", "failed"]
    assert J.TERMINAL == {S.COMPLETED, S.FAILED}
    assert J.TRANSITIONS[S.QUEUED] == {S.RUNNING, S.FAILED}
    assert J.TRANSITIONS[S.RUNNING] == {S.COMPLETED, S.FAILED}
    assert all(not J.TRANSITIONS[s] for s in J.TERMINAL)
    assert set(J.TRANSITIONS) == set(S)


def test_new_job_has_every_field(J):
    job = J.new_job(100.0)
    assert job == {"status": "queued", "report_id": None, "error": None,
                   "created_at": 100.0, "percent": 0, "message": "Queued"}


def test_happy_path(J):
    S = J.JobStatus
    job = J.new_job(0.0)
    J.advance(job, S.RUNNING, message="Initializing")
    J.set_progress(job, 35, "Running")
    assert (job["status"], job["percent"], job["message"]) == ("running", 35, "Running")
    J.advance(job, S.COMPLETED, report_id="R1", message="Complete!")
    assert job["status"] == "completed" and job["percent"] == 100
    assert job["report_id"] == "R1"


@pytest.mark.parametrize(
    "path",
    [("completed", "running"), ("completed", "failed"), ("failed", "running"),
     ("failed", "completed"), ("queued", "completed"), ("running", "queued"),
     ("queued", "queued")],
)
def test_illegal_moves_raise_and_change_nothing(J, path):
    src, dst = path
    job = {**J.new_job(0.0), "status": src, "report_id": "R1"}
    before = dict(job)
    with pytest.raises(ValueError, match="cannot move"):
        J.advance(job, J.JobStatus(dst), message="x")
    assert job == before


def test_completed_needs_a_report_id(J):
    job = J.new_job(0.0)
    J.advance(job, J.JobStatus.RUNNING)
    with pytest.raises(ValueError, match="report_id"):
        J.advance(job, J.JobStatus.COMPLETED)
    assert job["status"] == "running"


def test_failure_from_queued_and_running(J):
    for start in ("queued", "running"):
        job = {**J.new_job(0.0), "status": start}
        J.advance(job, J.JobStatus.FAILED, error="boom", percent=0)
        assert job["status"] == "failed" and job["error"] == "boom"


def test_progress_is_clamped_and_ignored_when_terminal(J):
    job = J.new_job(0.0)
    J.set_progress(job, 50, "early")  # still queued: ignored
    assert job["percent"] == 0 and job["message"] == "Queued"
    J.advance(job, J.JobStatus.RUNNING)
    J.set_progress(job, 250, "x")
    assert job["percent"] == 99  # 100 is reserved for completed
    J.advance(job, J.JobStatus.FAILED, error="e")
    J.set_progress(job, 10, "late")
    assert job["message"] != "late"


def test_view_keeps_old_fields_and_adds_new_ones(J):
    S = J.JobStatus
    job = J.new_job(10.0)
    v = J.job_view("J1", job, now=12.34)
    assert v == {"job_id": "J1", "status": "queued", "created_at": 10.0,
                 "elapsed": 2.3, "percent": 0, "message": "Queued",
                 "terminal": False, "next": "poll"}
    J.advance(job, S.RUNNING)
    J.set_progress(job, 65, "Generating")
    v = J.job_view("J1", job, now=13.0)
    assert (v["percent"], v["message"], v["terminal"], v["next"]) == (
        65, "Generating", False, "poll")
    J.advance(job, S.COMPLETED, report_id="R9")
    v = J.job_view("J1", job, now=14.0, has_pdf=True)
    assert v["report_id"] == "R9" and v["report_url"] == "/v1/audit/report/R9"
    assert v["pdf_url"] == "/v1/audit/pdf/R9"
    assert (v["terminal"], v["next"], v["percent"]) == (True, "fetch_report", 100)
    assert "error" not in v
    assert J.job_view("J1", job, now=14.0)["pdf_url"] is None


def test_view_of_a_failed_job(J):
    job = J.new_job(0.0)
    J.advance(job, J.JobStatus.FAILED, error="boom")
    v = J.job_view("J2", job, now=1.0)
    assert v["error"] == "boom" and v["terminal"] is True and v["next"] == "none"
    assert "report_id" not in v


# --- wiring in api/main.py (checked from source: FastAPI may not be installed) -------


@pytest.fixture(scope="module")
def main_src():
    return (ROOT / "api" / "main.py").read_text(encoding="utf-8")


def test_main_parses_and_uses_the_lifecycle(main_src):
    ast.parse(main_src)
    assert "from api.jobs import" in main_src
    assert "_JOB_STORE[job_id] = new_job(" in main_src
    assert "advance(job, JobStatus.RUNNING" in main_src
    assert "advance(job, JobStatus.COMPLETED" in main_src
    assert "advance(job, JobStatus.FAILED" in main_src
    assert "return job_view(" in main_src


def test_main_sets_job_status_only_through_advance(main_src):
    assert '.update({"status"' not in main_src
    assert '"status": "running"' not in main_src


def test_webhook_error_cannot_fail_a_completed_job(main_src):
    done = main_src.index("advance(job, JobStatus.COMPLETED")
    fail = main_src.index("advance(job, JobStatus.FAILED")
    hook = main_src.index('_fire_webhooks("job.completed"')
    assert done < fail < hook  # the completed webhook fires after the try/except


def test_job_endpoints_with_fastapi():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    sys.path.insert(0, str(ROOT))
    from api import main as api_main

    client = TestClient(api_main.create_app())
    assert client.get("/v1/jobs/NOPE").status_code == 404
    api_main._JOB_STORE["T1"] = api_main.new_job(0.0)
    body = client.get("/v1/jobs/T1").json()
    assert body["status"] == "queued" and body["terminal"] is False
    assert body["next"] == "poll" and body["percent"] == 0
    api_main.advance(api_main._JOB_STORE["T1"], api_main.JobStatus.RUNNING)
    api_main.advance(
        api_main._JOB_STORE["T1"], api_main.JobStatus.COMPLETED, report_id="R1"
    )
    body = client.get("/v1/jobs/T1").json()
    assert body["terminal"] is True and body["report_url"] == "/v1/audit/report/R1"
    api_main._JOB_STORE.pop("T1")
