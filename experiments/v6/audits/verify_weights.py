"""Verify cached Hugging Face weights against the Hub's published sha256, and repair.

A resumed download can produce a file of the right size with the wrong content
(seen 2026-09-29: a plain-HTTP resume on top of a failed hf-xet partial left an empty
safetensors header). This script hashes each cached ``model.safetensors`` and compares
it with the LFS sha256 the Hub publishes for that exact revision.

Usage (model specs as for ``glassbox-ai pilot``):
    python experiments/v6/audits/verify_weights.py pythia-410m@123000 pythia-410m-deduped@123000
    python experiments/v6/audits/verify_weights.py --repair pythia-410m-deduped@123000

``--repair`` deletes a corrupt or missing file and re-downloads it (set
HF_HUB_DISABLE_XET=1 first for a plain, resumable HTTP download), then re-verifies.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

FILE = "model.safetensors"


def repo_and_revision(spec: str) -> Tuple[str, Optional[str]]:
    """``pythia-410m@123000`` -> (``EleutherAI/pythia-410m``, ``step123000``)."""
    name, _, step = spec.partition("@")
    repo = name if "/" in name else f"EleutherAI/{name}"
    return repo, (f"step{step}" if step else None)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_sha(repo: str, revision: Optional[str]) -> Tuple[str, str, int]:
    """(weights filename, sha256, size). Prefers safetensors; seed repos ship .bin only."""
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    by_name = {x.rfilename: x for x in info.siblings}
    fname = FILE if FILE in by_name else "pytorch_model.bin"
    s = by_name[fname]
    return fname, s.lfs.sha256, s.size


def check(spec: str) -> Tuple[str, Optional[Path]]:
    """Return (status, cached_path). Status: OK, MISSING, SIZE_MISMATCH, HASH_MISMATCH."""
    repo, rev = repo_and_revision(spec)
    fname, want_sha, want_size = expected_sha(repo, rev)
    try:
        path = Path(hf_hub_download(repo, fname, revision=rev, local_files_only=True))
    except LocalEntryNotFoundError:
        return "MISSING", None
    if path.stat().st_size != want_size:
        return "SIZE_MISMATCH", path
    return ("OK" if sha256_of(path) == want_sha else "HASH_MISMATCH"), path


def repair(spec: str, path: Optional[Path]) -> None:
    """Delete the bad blob and its snapshot link, then download again."""
    repo, rev = repo_and_revision(spec)
    fname = expected_sha(repo, rev)[0]
    if path is not None:
        blob = path.resolve()
        path.unlink(missing_ok=True)
        blob.unlink(missing_ok=True)
    hf_hub_download(repo, fname, revision=rev, force_download=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--repair", action="store_true")
    args = ap.parse_args()
    if args.repair and os.environ.get("HF_HUB_DISABLE_XET") != "1":
        print("note: HF_HUB_DISABLE_XET is not 1; the re-download will use hf-xet")
    bad = 0
    for spec in args.specs:
        status, path = check(spec)
        if status != "OK" and args.repair:
            print(f"{spec}: {status} -> re-downloading", flush=True)
            repair(spec, path)
            status, path = check(spec)
        print(f"{spec}: {status}", flush=True)
        bad += status != "OK"
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
