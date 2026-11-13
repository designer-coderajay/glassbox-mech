# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
"""Package metadata helpers that are safe to import from any module."""

from __future__ import annotations


def package_version() -> str:
    """The installed Glassbox version (``glassbox.__version__``).

    Imported lazily so modules can call this without an import cycle. Use it wherever
    an output names the Glassbox version; never hardcode a version string.
    """
    from glassbox import __version__

    return str(__version__)
