# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
software

One folder per supported simulation code. Every sub-package whose
__init__.py exports `Adapter` (a software.base.SoftwareAdapter subclass) is
registered automatically -- adding a code means adding a folder, nothing
else. Folders starting with "_" (e.g. _template) are skipped.

`get_adapter(root, name)` returns the adapter named by --software, or
auto-detects one from the files under `root`. `adapter_for_label(label)`
finds the adapter for a description's software label ("DuMux").
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from pathlib import Path

from software.base import SnakefileContext, SnakefileResult, SoftwareAdapter

__all__ = [
    "ADAPTERS", "SOFTWARE_CHOICES", "SnakefileContext", "SnakefileResult", "SoftwareAdapter",
    "adapter_for_label", "add_arguments", "add_snakefile_arguments", "argument_dests", "get_adapter",
    "snakefile_argument_dests",
]


def _discover() -> list[type[SoftwareAdapter]]:
    found: list[type[SoftwareAdapter]] = []
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        if not info.ispkg or info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        cls = getattr(module, "Adapter", None)
        if not (isinstance(cls, type) and issubclass(cls, SoftwareAdapter)):
            raise ImportError(f"software/{info.name}/__init__.py must export `Adapter`, "
                              "a subclass of software.base.SoftwareAdapter")
        cls.slug = cls.slug or info.name
        found.append(cls)
    return found


ADAPTERS: list[type[SoftwareAdapter]] = _discover()
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


def adapter_for_label(label: str | None) -> SoftwareAdapter | None:
    """The adapter whose name or slug matches `label` (case-insensitive)."""
    wanted = (label or "").strip().lower()
    for cls in ADAPTERS:
        if wanted in (cls.name.lower(), cls.slug):
            return cls()
    return None


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Each software's own discovery options, one group per software."""
    for cls in ADAPTERS:
        cls.add_arguments(parser.add_argument_group(f"{cls.name} options"))


def add_snakefile_arguments(parser: argparse.ArgumentParser) -> None:
    """Each software's own Snakefile options, one group per software."""
    for cls in ADAPTERS:
        cls.add_snakefile_arguments(parser.add_argument_group(f"{cls.name} Snakefile options"))


def _dests(add) -> list[str]:
    parser = argparse.ArgumentParser(add_help=False)
    add(parser)
    return [action.dest for action in parser._actions]  # noqa: SLF001 -- no public accessor


def argument_dests() -> list[str]:
    """Namespace attributes of every software's discovery options -- what a
    caller forwards to metadata.builder without listing them itself."""
    return _dests(add_arguments)


def snakefile_argument_dests() -> list[str]:
    """Namespace attributes of every software's Snakefile options."""
    return _dests(add_snakefile_arguments)
