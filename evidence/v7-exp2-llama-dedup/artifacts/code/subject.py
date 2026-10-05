#!/usr/bin/env python3
"""Subject process for V7 Experiment 2 (see PROTOCOL.md): runs in a venv that has
only a pinned ``llama-index-core``. It reproduces llama_index issue #21033 (repro
adapted verbatim in substance from the issue body; llama_index is MIT-licensed) and
records one Glassbox trace per call.

``glassbox/v7/trace.py`` is loaded by file path (it is stdlib-only), so this venv needs
neither Glassbox nor torch.

Usage:
    python subject.py --path sync  --run r1 --out experiments/v7/llama_dedup/results
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Sequence

REPO = Path(__file__).resolve().parents[3]
QUERY = "test"


def load_trace_module() -> ModuleType:
    """Import glassbox/v7/trace.py without importing the glassbox package."""
    spec = importlib.util.spec_from_file_location(
        "gb_trace", REPO / "glassbox" / "v7" / "trace.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load glassbox/v7/trace.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["gb_trace"] = module
    spec.loader.exec_module(module)
    return module


def make_retriever() -> Any:
    """The issue's repro retriever: two nodes, same text, different source docs."""
    from llama_index.core.base.base_retriever import BaseRetriever
    from llama_index.core.schema import (
        NodeRelationship,
        NodeWithScore,
        RelatedNodeInfo,
        TextNode,
    )

    def make_node(text: str, source_doc_id: str) -> Any:
        node = TextNode(text=text, metadata={})
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id=source_doc_id
        )
        return node

    class DuplicateContentRetriever(BaseRetriever):  # type: ignore[misc]
        def _retrieve(self, query_bundle: Any) -> List[Any]:
            return [
                NodeWithScore(node=make_node("shared content", "doc-1"), score=0.9),
                NodeWithScore(node=make_node("shared content", "doc-2"), score=0.8),
            ]

    return DuplicateContentRetriever()


def describe(nodes: Sequence[Any]) -> List[Dict[str, Any]]:
    """Stable per-node record: no random node_id, only source, score and text hash."""
    return [
        {
            "ref_doc_id": n.node.ref_doc_id,
            "score": n.score,
            "text_sha256": hashlib.sha256(n.node.get_content().encode()).hexdigest(),
        }
        for n in nodes
    ]


def run(path: str, run_id: str, out: Path) -> Path:
    """Call the retriever via ``path`` and record the trace; return its file."""
    trace = load_trace_module()
    version = importlib.metadata.version("llama-index-core")
    retriever = make_retriever()
    rec = trace.Recorder(
        out_dir=out / version / path, run_id=run_id, label=f"{version}/{path}"
    )
    rec.environment["glassbox.env.packages"] = [f"llama-index-core=={version}"]
    with rec:
        with rec.span(
            "retrieval",
            "repro-21033",
            kind="INTERNAL",
            attributes={"gen_ai.data_source.id": "repro-21033"},
        ) as span:
            if path == "sync":
                nodes = retriever.retrieve(QUERY)
            else:
                nodes = asyncio.run(retriever.aretrieve(QUERY))
            rec.add_content(
                span,
                {
                    "gen_ai.retrieval.query.text": QUERY,
                    "gen_ai.retrieval.documents": describe(nodes),
                },
            )
    sys.stdout.write(f"{version} {path} {run_id}: {len(nodes)} node(s)\n")
    return Path(rec.path)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="V7 Exp 2 subject (llama_index #21033)"
    )
    parser.add_argument("--path", choices=("sync", "async"), required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    run(args.path, args.run, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
