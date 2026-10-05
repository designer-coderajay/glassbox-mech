#!/usr/bin/env python3
"""Package the toolkit skills as zip files for upload to claude.ai (Cowork, Claude apps).

Each zip holds one skill folder (<name>/SKILL.md plus its scripts), the layout the
claude.ai skill upload expects. Third-party skills (Agent-Reach, google-maps-scraper)
are included with their MIT licenses when their clones exist in .agent-tools/.

    python3 tools/agent-toolkit/package-skills.py            # -> dist/cowork-skills/*.zip
    python3 tools/agent-toolkit/package-skills.py --no-third-party
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TOOLS = REPO / ".agent-tools"
SKIP = {"__pycache__", ".DS_Store"}


def check_skill(skill_md: Path) -> str:
    text = skill_md.read_text()
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        raise SystemExit(f"{skill_md}: missing YAML front matter")
    name = re.search(r"^name:\s*(\S+)", m.group(1), re.M)
    desc = re.search(r"^description:\s*(.+)", m.group(1), re.M)
    if not name or not desc:
        raise SystemExit(f"{skill_md}: front matter needs name and description")
    if not re.fullmatch(r"[a-z0-9-]{1,64}", name.group(1)):
        raise SystemExit(f"{skill_md}: name must be lowercase letters, digits and hyphens (max 64)")
    if len(desc.group(1).strip().strip('"')) > 1024:
        raise SystemExit(f"{skill_md}: description longer than 1024 characters")
    return name.group(1)


def add_tree(zf: zipfile.ZipFile, src: Path, arc_root: str) -> None:
    for path in sorted(src.rglob("*")):
        if any(part in SKIP for part in path.parts) or path.suffix == ".pyc" or not path.is_file():
            continue
        zf.write(path, f"{arc_root}/{path.relative_to(src)}")


def package(src: Path, out_dir: Path, extras: dict[str, Path] | None = None) -> Path:
    name = check_skill(src / "SKILL.md")
    target = out_dir / f"{name}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        add_tree(zf, src, name)
        for arc, path in (extras or {}).items():
            if path.is_dir():
                add_tree(zf, path, f"{name}/{arc}")
            else:
                zf.write(path, f"{name}/{arc}")
    return target


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=str(REPO / "dist" / "cowork-skills"))
    ap.add_argument("--no-third-party", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    built = [package(d, out) for d in sorted((HERE / "skills").iterdir()) if (d / "SKILL.md").exists()]

    if not args.no_third_party:
        reach = TOOLS / "Agent-Reach"
        if reach.exists():
            # Agent-Reach ships SKILL.md (Chinese) and SKILL_en.md; package the English one.
            staging = out / ".agent-reach"
            staging.mkdir(exist_ok=True)
            (staging / "SKILL.md").write_text((reach / "agent_reach/skill/SKILL_en.md").read_text())
            built.append(package(staging, out, {"references": reach / "agent_reach/skill/references",
                                                "LICENSE": reach / "LICENSE"}))
            for p in sorted(staging.rglob("*"), reverse=True):
                p.unlink() if p.is_file() else p.rmdir()
            staging.rmdir()
        kit = TOOLS / "google-maps-scraper-kit"
        if kit.exists():
            built.append(package(kit / ".claude/skills/google-maps-scraper", out, {
                "scripts": kit / "scripts",
                "docker-compose.yml": kit / "docker-compose.yml",
                "LICENSE": kit / "LICENSE",
            }))

    for z in built:
        with zipfile.ZipFile(z) as zf:
            print(f"{z.relative_to(REPO) if z.is_relative_to(REPO) else z}  ({len(zf.namelist())} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
