"""Experiment-record provenance: git state, package versions, hardware, hashes.

Git is queried with ``--no-optional-locks`` so that reading provenance never creates
``.git/index.lock`` (a lock left behind by a crashed process blocks the user's commits).
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from glassbox.v6.claims import canonical_json

__all__ = ["git_state", "package_versions", "hardware", "sha256_json", "sha256_file",
           "environment_record"]

_PACKAGES = ("glassbox-mech-interp", "torch", "transformer_lens", "numpy", "scipy",
             "transformers")


def _git(args: Iterable[str], cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "--no-optional-locks", *args], cwd=cwd, capture_output=True,
            text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state(repo: Optional[Path] = None) -> Dict[str, Any]:
    """Commit SHA and dirty flag. ``sha`` is None outside a git checkout."""
    root = repo or Path(__file__).resolve().parents[2]
    sha = _git(["rev-parse", "HEAD"], root)
    status = _git(["status", "--porcelain", "--untracked-files=no"], root)
    return {"sha": sha, "dirty": bool(status) if status is not None else None}


def package_versions(names: Iterable[str] = _PACKAGES) -> Dict[str, Optional[str]]:
    """Installed versions; None for packages that are not installed."""
    out: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def hardware() -> Dict[str, Any]:
    """Platform, Python and CPU/accelerator description."""
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch_threads"] = torch.get_num_threads()
        info["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["cuda"] = None
    return info


def sha256_json(obj: Any) -> str:
    """sha256 of the canonical JSON encoding of ``obj``."""
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    """sha256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_record() -> Dict[str, Any]:
    """Everything about the execution environment that goes into a record."""
    return {"git": git_state(), "packages": package_versions(), "hardware": hardware()}
