# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
metadata.adapters

Registry of supported simulation codes. `get_adapter(root, name)` returns
the adapter named by --software, or auto-detects one from the files under
`root`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from metadata.adapters.base import SoftwareAdapter
from metadata.adapters.dumux import DumuxAdapter

ADAPTERS: list[type[SoftwareAdapter]] = [DumuxAdapter]

try:  # registered once implemented
    from metadata.adapters.openfoam import OpenFOAMAdapter
    ADAPTERS.append(OpenFOAMAdapter)
except ImportError:  # pragma: no cover
    pass

SOFTWARE_CHOICES = ["auto"] + [a.slug for a in ADAPTERS]


def get_adapter(root: Path, name: str | None = "auto") -> SoftwareAdapter:
    if name and name != "auto":
        for cls in ADAPTERS:
            if cls.slug == name.lower():
                return cls()
        sys.exit(f"Error: unknown --software {name!r}; choose from {', '.join(SOFTWARE_CHOICES)}")
    matches = [cls for cls in ADAPTERS if cls.detect(root)]
    if not matches:
        supported = ", ".join(a.name for a in ADAPTERS)
        sys.exit(f"Error: could not recognise the simulation software under {root} (supported: {supported}).")
    if len(matches) > 1:
        names = ", ".join(m.slug for m in matches)
        sys.exit(f"Error: {root} contains implementations for several codes ({names}); "
                 "pass --software to pick one, or point module_dir at one implementation.")
    return matches[0]()
