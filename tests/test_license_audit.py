"""Dependency licence audit (V8 B2 supply chain): reads installed package metadata,
follows runtime requirements from the roots, and flags copyleft or unknown licences."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "license_audit.py"


@pytest.fixture(scope="module")
def la():
    spec = importlib.util.spec_from_file_location("license_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dist(site: Path, name: str, version: str, headers: str) -> None:
    d = site / f"{name.replace('-', '_')}-{version}.dist-info"
    d.mkdir(parents=True)
    (d / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{headers}\n",
        encoding="utf-8",
    )


@pytest.fixture
def site(tmp_path):
    _dist(
        tmp_path,
        "rootpkg",
        "1.0",
        "License: MIT\nRequires-Dist: child-a\n"
        "Requires-Dist: extra-only; extra == 'dev'",
    )
    _dist(
        tmp_path,
        "child-a",
        "2.0",
        "License-Expression: Apache-2.0\n" "Requires-Dist: grandchild>=1",
    )
    _dist(
        tmp_path,
        "grandchild",
        "3.0",
        "Classifier: License :: OSI Approved :: BSD License",
    )
    _dist(tmp_path, "extra-only", "1.0", "License: GPL-3.0")
    _dist(tmp_path, "unrelated", "1.0", "License: AGPL-3.0")
    return tmp_path


@pytest.mark.parametrize(
    "lic,verdict",
    [
        ("MIT", "ALLOW"),
        ("BSD-3-Clause", "ALLOW"),
        ("Apache-2.0", "ALLOW"),
        ("Apache Software License", "ALLOW"),
        ("PSF-2.0", "ALLOW"),
        ("GPL-3.0-or-later", "DENY"),
        ("GNU Affero General Public License v3", "DENY"),
        ("LGPL-2.1", "REVIEW"),
        ("MPL-2.0", "REVIEW"),
        ("", "REVIEW"),
        ("UNKNOWN", "REVIEW"),
    ],
)
def test_verdict(la, lic, verdict):
    assert la.verdict(lic) == verdict


def test_closure_follows_runtime_requirements_only(la, site):
    dists = la.load(site)
    assert la.closure(dists, ["rootpkg"]) == ["child-a", "grandchild", "rootpkg"]


def test_licence_sources(la, site):
    dists = la.load(site)
    assert dists["child-a"]["licence"] == "Apache-2.0"
    assert dists["grandchild"]["licence"] == "BSD License"


def test_report_and_exit_code(la, site, capsys):
    assert la.main(["--site", str(site), "--roots", "rootpkg"]) == 0
    out = capsys.readouterr().out
    assert "| child-a | 2.0 | Apache-2.0 | ALLOW |" in out
    assert "unrelated" not in out
    assert la.main(["--site", str(site), "--roots", "unrelated"]) == 1


def test_missing_root_is_reported(la, site, capsys):
    assert la.main(["--site", str(site), "--roots", "nothere"]) == 1
    assert "not installed: nothere" in capsys.readouterr().out


def test_accept_list_covers_reviewed_packages_but_never_deny(la, tmp_path, capsys):
    _dist(
        tmp_path, "root", "1.0", "License: MIT\nRequires-Dist: mpl\nRequires-Dist: gpl"
    )
    _dist(tmp_path, "mpl", "1.0", "License: MPL-2.0")
    _dist(tmp_path, "gpl", "1.0", "License: GPL-3.0")
    args = ["--site", str(tmp_path), "--roots", "root", "--accept", "mpl", "gpl"]
    assert la.main(args) == 1  # GPL stays DENY even if listed
    out = capsys.readouterr().out
    assert "| mpl | 1.0 | MPL-2.0 | ACCEPTED |" in out
    assert "| gpl | 1.0 | GPL-3.0 | DENY |" in out
    _dist(tmp_path / "ok", "root", "1.0", "License: MIT\nRequires-Dist: mpl")
    _dist(tmp_path / "ok", "mpl", "1.0", "License: MPL-2.0")
    assert (
        la.main(["--site", str(tmp_path / "ok"), "--roots", "root", "--accept", "mpl"])
        == 0
    )
