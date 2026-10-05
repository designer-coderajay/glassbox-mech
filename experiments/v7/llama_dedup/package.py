#!/usr/bin/env python3
"""Package V7 Experiment 2 (llama_index #21033) as a verifiable evidence bundle.

Usage:
    python3 experiments/v7/llama_dedup/package.py --out evidence
    python -m glassbox.v7.evidence verify evidence/v7-exp2-llama-dedup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from glassbox.v7.evidence import PackageSpec, build_package

HERE = Path(__file__).resolve().parent
PACKAGE_ID = "v7-exp2-llama-dedup"

CLAIM = (
    "On a real pre-existing bug in llama-index-core (issue #21033), Glassbox traces and "
    "diff placed the sync-vs-async behavioural difference at the retrieval step under "
    "0.14.17, showed no difference under 0.14.19, showed the version change altered the "
    "sync path's retrieval but not the async path's, and were deterministic on re-run."
)
SCOPE = (
    "Pre-registered (protocol commit d866eea) with a disclosed sandbox dry run "
    "(Addendum 1). Not blind: the cause was known from the issue. Minimal repro from the "
    "issue, no LLM, two pinned versions. Retrieved-document hashes agreed across macOS "
    "and Linux runs by the same author; not independently reproduced."
)


def spec(root: Path = HERE) -> PackageSpec:
    """Build the package specification from the committed experiment files."""
    res = root / "results"
    artifacts: Dict[str, List[Union[str, Path]]] = {
        "protocol": [root / "PROTOCOL.md", root / "RUN.md"],
        "code": [root / "subject.py", root / "evaluate.py"],
        "measurements": [res / "results.json", root / "RESULTS.md"],
    }
    for version in ("0.14.17", "0.14.19"):
        for path in ("sync", "async"):
            key = f"traces_{version.replace('.', '_')}_{path}"
            artifacts[key] = sorted((res / version / path).glob("*.trace.jsonl"))
    return PackageSpec(
        claim=CLAIM,
        status="EXPERIMENTALLY_SUPPORTED",
        scope=SCOPE,
        artifacts=artifacts,
        provenance={
            "protocol_commit": "d866eea",
            "issue": "https://github.com/run-llama/llama_index/issues/21033",
            "versions": "llama-index-core 0.14.17 (bug), 0.14.19 (fixed)",
            "assurance_row": "V2.5",
        },
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Package V7 Experiment 2")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    pkg = build_package(args.out, PACKAGE_ID, spec())
    sys.stdout.write(f"wrote {pkg}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
