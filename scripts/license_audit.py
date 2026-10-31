#!/usr/bin/env python3
"""Audit the licences of Glassbox's runtime dependencies (V8 B2, supply chain).

Reads the metadata of the packages installed in a ``site-packages`` directory, follows
runtime ``Requires-Dist`` entries (extras excluded) from the root packages, and prints a
Markdown table with a verdict per package:

- ALLOW: permissive (MIT, BSD, Apache, PSF, ISC, ...);
- REVIEW: weak copyleft (LGPL, MPL, EPL) or no licence information; read it by hand;
- DENY: strong copyleft (GPL, AGPL, SSPL); not allowed by the project rules;
- ACCEPTED: a REVIEW package that was read by hand and listed with ``--accept``
  (see ``docs/SUPPLY_CHAIN.md``). DENY can never be accepted.

Exit code 1 if anything is DENY or REVIEW, or a root is not installed. The verdict comes
from package metadata, which can be wrong or incomplete; it is a screening step, not
legal advice.

Usage:
    python3 scripts/license_audit.py --site .venv/lib/python3.11/site-packages \\
        --roots torch transformer-lens einops numpy scipy
"""

from __future__ import annotations

import argparse
import re
import sys
from email.parser import Parser
from pathlib import Path
from typing import Dict, List, Optional, Sequence

BASE_ROOTS = ("torch", "transformer-lens", "einops", "numpy", "scipy")
_DENY = re.compile(r"\bA?GPL\b|General Public License|\bSSPL\b", re.I)
_WEAK = re.compile(r"\bLGPL\b|Lesser General|\bMPL\b|Mozilla|\bEPL\b|Eclipse", re.I)
_ALLOW = re.compile(
    r"\bMIT\b|\bBSD\b|Apache|\bPSF\b|Python Software Foundation|\bISC\b|Unlicense|"
    r"Zlib|0BSD|HPND|CC0|Public Domain",
    re.I,
)

Dist = Dict[str, object]


def norm(name: str) -> str:
    """PEP 503 normalised package name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def verdict(licence: str) -> str:
    """ALLOW, REVIEW or DENY for a licence string (see module docstring)."""
    if not licence or licence.strip().upper() == "UNKNOWN":
        return "REVIEW"
    if _WEAK.search(licence):
        return "REVIEW"
    if _DENY.search(licence):
        return "DENY"
    return "ALLOW" if _ALLOW.search(licence) else "REVIEW"


def _licence(meta) -> str:  # type: ignore[no-untyped-def]
    expr = meta.get("License-Expression")
    if expr:
        return str(expr).strip()
    lic = (meta.get("License") or "").strip()
    if lic and len(lic) <= 80 and "\n" not in lic:
        return lic
    classifiers = [
        c.split("::")[-1].strip()
        for c in meta.get_all("Classifier") or []
        if c.startswith("License ::")
    ]
    if classifiers:
        return "; ".join(classifiers)
    return lic.splitlines()[0][:80] if lic else ""


def load(site: Path) -> Dict[str, Dist]:
    """Name → {name, version, licence, requires} for every ``*.dist-info`` in ``site``."""
    dists: Dict[str, Dist] = {}
    for info in sorted(Path(site).glob("*.dist-info")):
        meta_file = info / "METADATA"
        if not meta_file.exists():
            continue
        meta = Parser().parsestr(meta_file.read_text(encoding="utf-8", errors="ignore"))
        requires = [
            norm(re.split(r"[ ;<>=!~\[(]", r.strip(), maxsplit=1)[0])
            for r in meta.get_all("Requires-Dist") or []
            if "extra ==" not in r and "extra==" not in r
        ]
        dists[norm(meta["Name"])] = {
            "name": norm(meta["Name"]),
            "version": meta["Version"],
            "licence": _licence(meta),
            "requires": requires,
        }
    return dists


def closure(dists: Dict[str, Dist], roots: Sequence[str]) -> List[str]:
    """Sorted names reachable from ``roots`` through runtime requirements."""
    seen, todo = set(), [norm(r) for r in roots]
    while todo:
        name = todo.pop()
        if name in seen or name not in dists:
            continue
        seen.add(name)
        todo.extend(dists[name]["requires"])  # type: ignore[arg-type]
    return sorted(seen)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print the audit table; exit 1 on DENY, REVIEW or a missing root."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", required=True, type=Path)
    parser.add_argument("--roots", nargs="+", default=list(BASE_ROOTS))
    parser.add_argument(
        "--accept",
        nargs="*",
        default=[],
        help="packages reviewed by hand: REVIEW becomes ACCEPTED (DENY never does)",
    )
    args = parser.parse_args(argv)
    dists = load(args.site)
    missing = [r for r in args.roots if norm(r) not in dists]
    names = closure(dists, args.roots)
    lines = [
        "| Package | Version | Licence (from metadata) | Verdict |",
        "|---|---|---|---|",
    ]
    flagged = 0
    for n in names:
        d = dists[n]
        v = verdict(str(d["licence"]))
        if v == "REVIEW" and n in {norm(a) for a in args.accept}:
            v = "ACCEPTED"
        flagged += v not in ("ALLOW", "ACCEPTED")
        lines.append(f"| {n} | {d['version']} | {d['licence'] or '(none)'} | {v} |")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.write(f"\n{len(names)} packages, {flagged} need attention.\n")
    if missing:
        sys.stdout.write(f"not installed: {', '.join(missing)}\n")
    return 1 if flagged or missing else 0


if __name__ == "__main__":
    sys.exit(main())
