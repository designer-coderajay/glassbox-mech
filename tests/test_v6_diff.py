"""V6 diff pipeline: CLI wiring, guard rails, and a gated end-to-end run.

The end-to-end test downloads two small Pythia checkpoints and is skipped unless
``GLASSBOX_V6_E2E=1`` is set.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

from glassbox.v6.diff import DiffConfig, run_diff


def test_credit_task_not_available_yet(tmp_path) -> None:
    with pytest.raises(NotImplementedError, match="Arm B"):
        run_diff(DiffConfig("a", "b", task="credit"), tmp_path)


def test_single_pair_cannot_be_confirmatory(tmp_path) -> None:
    with pytest.raises(ValueError):
        run_diff(DiffConfig("a", "b", label="confirmatory"), tmp_path)


def test_cli_credit_exits_2(monkeypatch, capsys, tmp_path) -> None:
    from glassbox import cli

    monkeypatch.setattr(sys, "argv", ["glassbox-ai", "diff", "a", "b", "--task", "credit",
                                      "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "Arm B" in capsys.readouterr().err


@pytest.mark.skipif(os.environ.get("GLASSBOX_V6_E2E") != "1",
                    reason="set GLASSBOX_V6_E2E=1 to download Pythia and run end to end")
def test_end_to_end_pythia_checkpoints(tmp_path) -> None:
    cfg = DiffConfig("pythia-70m", "pythia-70m", checkpoint_a=143000,
                     checkpoint_b=71000, n_prompts=4, seed=0, k=5)
    finding = run_diff(cfg, tmp_path)
    rec = json.loads((tmp_path / "record.json").read_text())
    fnd = json.loads((tmp_path / "finding.json").read_text())
    assert fnd["label"] == "smoke"
    assert all(h["status"] == "UNRESOLVED" for h in fnd["hypotheses"])
    assert rec["controls"]["A"]["status"] in {"PASS_BITWISE", "RECORDED_TOLERANCE_PENDING"}
    assert rec["controls"]["A"]["D_B_self"] < 1e-6
    assert 0.0 <= rec["distances"]["D_B"]["value"] <= 1.0
    assert rec["dataset"]["hash"] == finding.scope.dataset_hash
    assert rec["environment"]["packages"]["torch"]
