# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
snakefile.styles

How a software runs one benchmark case, as named in its software.yaml
(`snakefile: {style: ...}`). Each style module provides:

  add_arguments(group)                      command-line options of the style
  build_hints(adapter, module_dir, repo)    facts its Snakefile needs, found
                                            when the description is made
  render(adapter, ctx) -> SnakefileResult   the Snakefile
"""

from __future__ import annotations

from types import ModuleType

from snakefile.styles import case_files, command_line

STYLES: dict[str, ModuleType] = {
    "command_line": command_line,
    "case_files": case_files,
}


def get_style(name: str) -> ModuleType:
    try:
        return STYLES[name]
    except KeyError:
        raise SystemExit(f"Error: unknown snakefile style {name!r} in software.yaml; "
                         f"choose from {', '.join(STYLES)}") from None
