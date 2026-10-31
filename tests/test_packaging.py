"""The distribution ships only the ``glassbox`` package.

Release 4.5.1 also installed top-level ``tests``, ``scripts``, ``benchmarks``,
``examples``, ``api``, ``dashboard``, ``experiments`` and ``mcp`` packages into
site-packages. The last one shadows the official ``mcp`` SDK. These tests pin the
package list so that cannot happen again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on 3.9/3.10 CI only
    tomllib = pytest.importorskip("tomli")


def _find_config() -> dict:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["tool"]["setuptools"]["packages"]["find"]


def _packages() -> list:
    from setuptools import find_packages

    cfg = _find_config()
    return find_packages(
        where=cfg.get("where", ["."])[0],
        include=cfg.get("include", ("*",)),
        exclude=cfg.get("exclude", ()),
    )


def test_only_glassbox_is_packaged():
    pkgs = _packages()
    assert pkgs, "no packages found"
    stray = [p for p in pkgs if p != "glassbox" and not p.startswith("glassbox.")]
    assert stray == []


def test_every_glassbox_subpackage_is_packaged():
    pkgs = set(_packages())
    for init in (ROOT / "glassbox").rglob("__init__.py"):
        rel = init.parent.relative_to(ROOT)
        if "__pycache__" in rel.parts:
            continue
        assert ".".join(rel.parts) in pkgs
    assert "glassbox.v7" in pkgs


def test_every_glassbox_module_dir_has_init():
    missing = [
        str(d.relative_to(ROOT))
        for d in {p.parent for p in (ROOT / "glassbox").rglob("*.py")}
        if "__pycache__" not in d.parts and not (d / "__init__.py").exists()
    ]
    assert missing == []


def test_python_versions_match_what_is_tested():
    """Declare only versions the suite has run on (CI 3.11; local 3.10)."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    assert project["requires-python"] == ">=3.10"
    declared = {
        c.rsplit("::", 1)[-1].strip()
        for c in project["classifiers"]
        if c.startswith("Programming Language :: Python :: 3.")
    }
    assert declared == {"3.10", "3.11"}
