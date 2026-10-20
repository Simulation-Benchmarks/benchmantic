# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
snakefile.styles

How a software runs one benchmark case, chosen by `run: {type: ...}` in
its software description (program -> command_line, case_folder ->
case_files; see software.generic.RUN_TYPES). Each style module provides:

  add_arguments(group)                      command-line options of the style
  build_hints(adapter, module_dir, repo)    facts its Snakefile needs, found
                                            when the description is made
  render(adapter, ctx) -> SnakefileResult   the Snakefile
"""

from __future__ import annotations

from types import ModuleType

from snakefile.styles import case_files, command_line, python_script

STYLES: dict[str, ModuleType] = {
    "command_line": command_line,
    "case_files": case_files,
    "python_script": python_script,
}


def get_style(name: str) -> ModuleType:
    try:
        return STYLES[name]
    except KeyError:
        raise SystemExit("Error: unknown `run: type` in the software description; "
                         "choose program or case_folder") from None


def add_arguments(parser) -> None:
    """Every style's command-line options, once each."""
    for name, style in STYLES.items():
        style.add_arguments(parser.add_argument_group(f"Snakefile options ({name} style)"))


def argument_dests() -> list[str]:
    """Namespace attributes of the styles' options."""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    add_arguments(parser)
    return [action.dest for action in parser._actions]  # noqa: SLF001 -- no public accessor
