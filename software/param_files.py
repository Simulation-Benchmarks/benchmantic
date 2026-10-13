# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
software.param_files

Finds a benchmark's parameter file when it isn't where software.yaml says
(e.g. a DuMux module whose inputs are in `rotating.input` instead of
`params.input`). Turned on in software.yaml with

  inputs:
    search:
      patterns: ["*.input", "*.ini", "*.prm"]   file names to consider
      max_depth: 3                              how far below the module

Every file matching a pattern is read with the format's sniff(), which
says how many entries it holds (0: not this format). The candidates are
ranked by

  1. being named in the source code (e.g. main.cc reads "rotating.input"),
  2. having the configured name (inputs.files),
  3. holding more entries,

and the best one is used. All candidates are kept on the adapter
(`parameter_files`) so a run can say what it found and why.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

#: Never searched: version control, build trees, benchmantic's own files.
SKIP_PARTS = {".git", ".benchmantic", "build", "build-cmake", "__pycache__", "node_modules", "outputs"}
DEFAULT_PATTERNS = ["*.input", "*.ini", "*.cfg", "*.conf", "*.prm", "*.par", "*.params", "*.inp"]


@dataclass
class FoundFile:
    path: Path
    #: Number of entries the format read from it.
    entries: int
    #: Why it ranks where it does, for messages.
    reasons: list[str]
    score: tuple[int, int, int]


def find_parameter_files(root: Path, fmt: ModuleType, search: dict[str, Any] | bool,
                         code_texts: list[str], configured: list[str] | None = None) -> list[FoundFile]:
    """Parameter files under `root`, best first. Empty if none reads as
    the format."""
    settings = search if isinstance(search, dict) else {}
    patterns = list(settings.get("patterns") or DEFAULT_PATTERNS)
    max_depth = int(settings.get("max_depth", 3))
    configured = configured or []
    code = "\n".join(code_texts)

    found = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if (not path.is_file() or len(rel.parts) > max_depth + 1
                or SKIP_PARTS & set(rel.parts) or any(part.startswith(".") for part in rel.parts)
                or not any(fnmatch.fnmatch(path.name, p) for p in patterns)):
            continue
        entries = fmt.sniff(path)
        if not entries:
            continue
        named_in_code = f'"{path.name}"' in code or f"'{path.name}'" in code
        reasons = [f"{entries} entries"]
        if named_in_code:
            reasons.insert(0, "named in the source code")
        if path.name in configured:
            reasons.insert(0, "has the configured name")
        found.append(FoundFile(path=path, entries=entries, reasons=reasons,
                               score=(int(named_in_code), int(path.name in configured), entries)))
    return sorted(found, key=lambda f: f.score, reverse=True)
