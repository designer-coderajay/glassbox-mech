#!/usr/bin/env python3
"""Guard the frozen V6 artefacts (V7 Step 1).

V6 is a completed, preregistered study. Its protocol, runner, prompts, record,
cache, audits and the ``glassbox.v6`` library are frozen: V7 may read them but
must never change them. This script keeps a sha256 manifest of every
git-tracked file under the frozen roots and reports any file that was
modified, removed or added since the freeze.

Usage:
    python3 scripts/check_v6_frozen.py            # check (exit 1 on violation)
    python3 scripts/check_v6_frozen.py --write    # create manifest (once only)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "glassbox.v6.frozen/1"
FROZEN_ROOTS: Tuple[str, ...] = ("experiments/v6", "glassbox/v6")
MANIFEST_RELPATH = Path("experiments") / "V6_FROZEN_MANIFEST.json"
LOCK_TAG = "v6-amendment3-lock"

Violation = Tuple[str, str]


def sha256_file(path: Path) -> str:
    """Return the hex sha256 digest of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "--no-optional-locks", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout


def list_frozen_files(root: Path) -> List[str]:
    """List git-tracked files under the frozen roots, sorted, POSIX paths."""
    raw = _git(root, "ls-files", "-z", "--", *FROZEN_ROOTS)
    return sorted(p for p in raw.split("\0") if p)


def build_manifest(root: Path, files: Sequence[str], frozen_at: str) -> Dict:
    """Build the manifest dict for ``files`` (paths relative to ``root``)."""
    return {
        "schema": SCHEMA,
        "frozen_at_commit": frozen_at,
        "lock_tag": LOCK_TAG,
        "roots": list(FROZEN_ROOTS),
        "files": {rel: sha256_file(root / rel) for rel in sorted(files)},
    }


def compare(
    manifest: Dict, root: Path, current_files: Sequence[str]
) -> List[Violation]:
    """Return sorted (kind, path) violations; kind is added/modified/removed."""
    recorded: Dict[str, str] = manifest["files"]
    current = set(current_files)
    violations: List[Violation] = []
    for rel, expected in recorded.items():
        path = root / rel
        if rel not in current or not path.exists():
            violations.append(("removed", rel))
        elif sha256_file(path) != expected:
            violations.append(("modified", rel))
    for rel in current - set(recorded):
        violations.append(("added", rel))
    return sorted(violations, key=lambda v: (v[1], v[0]))


def write_manifest(manifest: Dict, target: Path) -> None:
    """Write the manifest; refuse to overwrite an existing freeze."""
    if target.exists():
        raise FileExistsError(f"{target} exists; a freeze is never rewritten")
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write", action="store_true", help="create the manifest (refuses overwrite)"
    )
    args = parser.parse_args(argv)
    target = ROOT / MANIFEST_RELPATH
    files = list_frozen_files(ROOT)

    if args.write:
        head = _git(ROOT, "rev-parse", "HEAD").strip()
        manifest = build_manifest(ROOT, files, frozen_at=head)
        try:
            write_manifest(manifest, target)
        except FileExistsError as err:
            print(f"refused: {err}", file=sys.stderr)
            return 1
        print(f"wrote {MANIFEST_RELPATH} ({len(files)} files, HEAD {head[:7]})")
        return 0

    if not target.exists():
        print(f"missing {MANIFEST_RELPATH}; run with --write once", file=sys.stderr)
        return 2
    manifest = json.loads(target.read_text(encoding="utf-8"))
    violations = compare(manifest, ROOT, files)
    if violations:
        print(f"V6 FREEZE VIOLATED ({len(violations)}):", file=sys.stderr)
        for kind, rel in violations:
            print(f"  {kind:8s} {rel}", file=sys.stderr)
        return 1
    print(f"V6 frozen artefacts intact ({len(manifest['files'])} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
