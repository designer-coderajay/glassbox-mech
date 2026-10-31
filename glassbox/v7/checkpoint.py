# SPDX-License-Identifier: BUSL-1.1
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
# Licensed under the Business Source License 1.1 (see LICENSE-COMMERCIAL).
# Free for non-production use and for production use inside your own organisation.
# Offering it as a commercial service or product to third parties needs a license.
"""Checkpoint and resume for long V7 runs (e.g. on a Colab GPU that may disconnect).

Each finished unit of work (a case, a model, a configuration) is saved as one JSON file,
written atomically and never overwritten. A fingerprint of the settings is stored once;
reopening the directory with different settings is refused, so a resumed run can never
mix results from two set-ups.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping

_UNIT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FINGERPRINT = "_fingerprint.json"


class Checkpoint:
    """Per-unit JSON results in ``directory``, bound to a settings fingerprint."""

    def __init__(self, directory: Path, fingerprint: Mapping[str, Any]) -> None:
        """Open or create the checkpoint directory.

        Raises:
            ValueError: If the directory holds results made with different settings.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        wanted = json.loads(json.dumps(dict(fingerprint), sort_keys=True))
        path = self.directory / _FINGERPRINT
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored != wanted:
                raise ValueError(
                    f"{self.directory} holds results made with different settings: "
                    f"{stored} != {wanted}; use a new directory"
                )
        else:
            self._write(path, wanted)

    def _path(self, unit: str) -> Path:
        if not _UNIT.match(unit):
            raise ValueError(f"invalid unit name {unit!r}")
        return self.directory / f"{unit}.json"

    @staticmethod
    def _write(path: Path, data: Any) -> None:
        text = json.dumps(data, indent=1, sort_keys=True) + "\n"  # fails before writing
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def done(self, unit: str) -> bool:
        """Whether ``unit`` has a saved result."""
        return self._path(unit).exists()

    def load(self, unit: str) -> Dict[str, Any]:
        """The saved result of ``unit``."""
        data: Dict[str, Any] = json.loads(self._path(unit).read_text(encoding="utf-8"))
        return data

    def save(self, unit: str, data: Mapping[str, Any]) -> None:
        """Save ``unit`` atomically.

        Raises:
            FileExistsError: If ``unit`` was already saved (results are never replaced).
            TypeError: If ``data`` is not JSON-serialisable (nothing is written).
        """
        path = self._path(unit)
        if path.exists():
            raise FileExistsError(f"{path} exists; results are never overwritten")
        self._write(path, dict(data))

    def units(self) -> List[str]:
        """Names of all saved units, sorted."""
        return sorted(
            p.stem for p in self.directory.glob("*.json") if p.name != _FINGERPRINT
        )

    def run(self, unit: str, work: Callable[[], Mapping[str, Any]]) -> Dict[str, Any]:
        """Return the saved result of ``unit``, or compute it with ``work`` and save."""
        if not self.done(unit):
            self.save(unit, work())
        return self.load(unit)
