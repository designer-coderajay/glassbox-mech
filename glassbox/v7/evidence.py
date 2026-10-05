"""Evidence packages: a self-contained, verifiable bundle for one claim (V7 plan §15–§17).

Layout of ``<out_dir>/<package_id>/``:

- ``claim.json``: the claim, its ASSURANCE status, its scope and limits;
- ``provenance.json``: commits, environment, tool version, and the reproducer if any;
- ``artifacts/<role>/<file>``: copies of protocol, traces, measurements, etc.;
- ``README.md``: a human-readable report, including how to verify the package;
- ``manifest.json``: the sha256 of every file above.

Mapping to the V7 plan §16 names: ``experiment.json`` → ``artifacts/protocol``,
``trace.json`` → ``artifacts/traces``, ``measurements.json`` → ``artifacts/measurements``,
``integrity.json`` → ``manifest.json``, ``report.html`` → ``README.md``.

Integrity is not truth (plan §17): a package that verifies proves only that its files
are unchanged since packaging. It says nothing about whether the experiment was
correct.

Usage:
    python -m glassbox.v7.evidence verify evidence/v7-exp1-rag-fault
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

PathLike = Union[str, "os.PathLike[str]"]
Violation = Tuple[str, str]

SCHEMA = "glassbox.evidence/0.1"
MANIFEST = "manifest.json"
STATUSES = frozenset(
    {
        "VERIFIED",
        "EXPERIMENTALLY_SUPPORTED",
        "REPRODUCED",
        "PRELIMINARY",
        "IMPLEMENTED",
        "HYPOTHESIS",
        "UNRESOLVED",
        "REFUTED",
        "PLANNED",
        "NOT_SUPPORTED",
    }
)
INTEGRITY_NOTE = (
    "Integrity is not truth: a verifying manifest shows these files are unchanged since "
    "packaging; it does not show the experiment or the claim is correct."
)


class EvidenceError(ValueError):
    """Raised when a package specification is invalid."""


@dataclass
class PackageSpec:
    """What goes into a package.

    Attributes:
        claim: One-sentence claim, worded at the strength the evidence supports.
        status: A status from the ASSURANCE.md taxonomy.
        scope: Conditions and limits under which the claim holds.
        artifacts: Role name → files (e.g. ``{"protocol": [...], "traces": [...]}``).
        provenance: Commits, dates, and ``reproduced_by`` for ``REPRODUCED``.
    """

    claim: str
    status: str
    scope: str
    artifacts: Dict[str, List[PathLike]]
    provenance: Dict[str, str] = field(default_factory=dict)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(spec: PackageSpec) -> None:
    if spec.status not in STATUSES:
        raise EvidenceError(f"status {spec.status!r} is not in the ASSURANCE taxonomy")
    if spec.status == "REPRODUCED" and not spec.provenance.get("reproduced_by"):
        raise EvidenceError(
            "REPRODUCED requires provenance['reproduced_by'] naming an independent "
            "reproducer"
        )
    for role, files in spec.artifacts.items():
        names = [Path(f).name for f in files]
        if len(names) != len(set(names)):
            raise EvidenceError(f"duplicate file names in artifact role {role!r}")
        for f in files:
            if not Path(f).is_file():
                raise EvidenceError(f"artifact not found: {f}")


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _readme(package_id: str, spec: PackageSpec, files: Sequence[str]) -> str:
    lines = [
        f"# Evidence package: {package_id}",
        "",
        f"**Claim.** {spec.claim}",
        "",
        f"**Status:** `{spec.status}` (ASSURANCE.md taxonomy)",
        "",
        f"**Scope and limits.** {spec.scope}",
        "",
        "## Provenance",
        "",
        *[f"- {k}: `{v}`" for k, v in sorted(spec.provenance.items())],
        "",
        "## Contents",
        "",
        *[f"- `{f}`" for f in files],
        "",
        "## Verify",
        "",
        f"`python -m glassbox.v7.evidence verify <path-to>/{package_id}`",
        "",
        f"**{INTEGRITY_NOTE.split(':')[0]}.**{INTEGRITY_NOTE.split(':', 1)[1]}",
        "",
    ]
    return "\n".join(lines)


def build_package(out_dir: PathLike, package_id: str, spec: PackageSpec) -> Path:
    """Create ``<out_dir>/<package_id>`` and return its path.

    Raises:
        EvidenceError: If the spec is invalid (status, reproducer, missing/duplicate files).
        FileExistsError: If the package directory already exists.
    """
    _validate(spec)
    pkg = Path(out_dir) / package_id
    if pkg.exists():
        raise FileExistsError(f"{pkg} exists; evidence packages are never overwritten")
    copied: List[str] = []
    for role, paths in sorted(spec.artifacts.items()):
        for f in paths:
            dest = pkg / "artifacts" / role / Path(f).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dest)
            copied.append(dest.relative_to(pkg).as_posix())
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_json(
        pkg / "claim.json",
        {
            "schema": SCHEMA,
            "package_id": package_id,
            "claim": spec.claim,
            "status": spec.status,
            "scope": spec.scope,
            "integrity_note": INTEGRITY_NOTE,
            "created_utc": created,
        },
    )
    provenance = dict(spec.provenance)
    provenance.update(
        {"python": sys.version.split()[0], "platform": platform.platform()}
    )
    _write_json(pkg / "provenance.json", provenance)
    files = sorted(copied + ["claim.json", "provenance.json", "README.md"])
    (pkg / "README.md").write_text(_readme(package_id, spec, files), encoding="utf-8")
    manifest = {"schema": SCHEMA, "files": {f: _sha256(pkg / f) for f in files}}
    _write_json(pkg / MANIFEST, manifest)
    return pkg


def verify_package(pkg: PathLike) -> List[Violation]:
    """Return ``(kind, relpath)`` violations (added/modified/removed); empty if intact."""
    root = Path(pkg)
    recorded: Dict[str, str] = json.loads(
        (root / MANIFEST).read_text(encoding="utf-8")
    )["files"]
    present = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.name != MANIFEST
    }
    out: List[Violation] = []
    for rel, digest in recorded.items():
        if rel not in present:
            out.append(("removed", rel))
        elif _sha256(root / rel) != digest:
            out.append(("modified", rel))
    out.extend(("added", rel) for rel in present - set(recorded))
    return sorted(out, key=lambda v: (v[1], v[0]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: ``verify <package>`` → exit 0 if intact, 1 otherwise."""
    parser = argparse.ArgumentParser(description="Glassbox V7 evidence packages")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify").add_argument("package")
    args = parser.parse_args(argv)
    violations = verify_package(args.package)
    if violations:
        for kind, rel in violations:
            sys.stdout.write(f"{kind:8s} {rel}\n")
        sys.stdout.write("INTEGRITY FAILED\n")
        return 1
    sys.stdout.write(f"Package intact. {INTEGRITY_NOTE}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
