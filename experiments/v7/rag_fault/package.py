#!/usr/bin/env python3
"""Package V7 Experiment 1 as a verifiable evidence bundle.

Usage:
    python3 experiments/v7/rag_fault/package.py --out evidence
    python -m glassbox.v7.evidence verify evidence/v7-exp1-rag-fault
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from glassbox.v7.evidence import PackageSpec, build_package

HERE = Path(__file__).resolve().parent
PACKAGE_ID = "v7-exp1-rag-fault"
CONDITIONS = ("baseline", "baseline_repeat", "fault", "restored")

CLAIM = (
    "In a controlled RAG pipeline with one injected cause (a stale index missing four "
    "documents), the Glassbox trace diff placed the first divergence at the retrieval "
    "step for 4/4 affected questions, reported no divergence for 4/4 unaffected "
    "questions, confirmed identical behaviour after the fault was undone (8/8) and was "
    "deterministic on re-run (8/8)."
)
SCOPE = (
    "Pre-registered (protocol commit 0e70405, before the run). GPT-2 small, "
    "teacher-forced two-option scoring, 8 fictional questions, bag-of-words retriever "
    "(top_k=1), no sampling, single machine (CPU). The cause was known by construction; "
    "this does not show Glassbox localises unknown causes in real systems (V7 Hard Gate "
    "3, open). Criteria recomputed from the traces by the author; not independently "
    "reproduced."
)


def spec(root: Path = HERE) -> PackageSpec:
    """Build the package specification from the committed experiment files."""
    artifacts: Dict[str, List[Union[str, Path]]] = {
        "protocol": [root / "PROTOCOL.md", root / "corpus.json"],
        "code": [root / "run.py"],
        "measurements": [root / "results" / "results.json", root / "RESULTS.md"],
    }
    for cond in CONDITIONS:
        artifacts[f"traces_{cond}"] = sorted((root / "results" / cond).glob("*.jsonl"))
    return PackageSpec(
        claim=CLAIM,
        status="EXPERIMENTALLY_SUPPORTED",
        scope=SCOPE,
        artifacts=artifacts,
        provenance={
            "protocol_commit": "0e70405",
            "results_commit": "aedfebf",
            "model": "gpt2 (TransformerLens)",
            "assurance_row": "V2.3",
        },
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Package V7 Experiment 1")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    pkg = build_package(args.out, PACKAGE_ID, spec())
    sys.stdout.write(f"wrote {pkg}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
