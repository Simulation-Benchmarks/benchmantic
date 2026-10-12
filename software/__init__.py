# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
software

One folder per supported simulation code, registered automatically --
adding a code means adding a folder, nothing else:

  software/<name>/software.yaml   the usual case: the basic details of the
                                  code, read by software.generic
  software/<name>/__init__.py     only for a code no config can describe:
                                  exports `Adapter`, a SoftwareAdapter
                                  subclass written in Python

Folders starting with "_" (e.g. _template) are skipped, and so are the
shared packages here (formats/).

`get_adapter(root, name)` returns the adapter named by --software, or
auto-detects one from the files under `root`. `adapter_for_label(label)`
finds the adapter for a description's software label ("DuMux").
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from software.base import SnakefileContext, SnakefileResult, SoftwareAdapter

__all__ = [
    "ADAPTERS", "SOFTWARE_CHOICES", "SnakefileContext", "SnakefileResult", "SoftwareAdapter",
    "adapter_for_label", "add_arguments", "add_snakefile_arguments", "argument_dests", "get_adapter",
    "snakefile_argument_dests",
]


def _discover() -> list[type[SoftwareAdapter]]:
    from software.generic import GenericAdapter

    found: list[type[SoftwareAdapter]] = []
    folders = sorted({d for base in __path__ for d in Path(base).iterdir() if d.is_dir()}, key=lambda d: d.name)
    for folder in folders:
        if folder.name.startswith(("_", ".")):
            continue
        if (folder / "software.yaml").is_file():
            found.append(GenericAdapter.for_folder(folder))
            continue
        if not (folder / "__init__.py").is_file():
            continue
        module = importlib.import_module(f"{__name__}.{folder.name}")
        cls = getattr(module, "Adapter", None)
        if cls is None:
            continue  # a shared package such as formats/, not a software
        if not (isinstance(cls, type) and issubclass(cls, SoftwareAdapter)):
            raise ImportError(f"software/{folder.name}/__init__.py: `Adapter` must be a subclass of "
                              "software.base.SoftwareAdapter")
        cls.slug = cls.slug or folder.name
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
    """Snakefile options: once per Snakefile style in use (styles are
    shared between codes), then each software's own extras."""
    styles: dict[str, list[str]] = {}
    for cls in ADAPTERS:
        style = getattr(cls, "style", None)
        if style is not None:
            styles.setdefault(style().__name__.rsplit(".", 1)[-1], []).append(cls.name)
    for style_name, users in styles.items():
        from snakefile.styles import get_style
        get_style(style_name).add_arguments(
            parser.add_argument_group(f"Snakefile options, {style_name} style ({', '.join(users)})"))
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
