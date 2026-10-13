# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
software

Which simulation software a benchmark uses, and how benchmantic reads it.
benchmantic holds no software-specific code: a *software description*
(software/template.yaml lists every key) says how a software organises an
implementation. The run config only names the software; resolve() then

  1. uses software_descriptions/<name>.yaml when there is one,
  2. otherwise lets the LLM write the description (software.autoconfig).

With no software named, the first of its own descriptions that fits the
implementation is used, else the LLM is asked.

`software-description` in the run config points at a description file of
your own instead -- it may be partial; the LLM fills in the rest.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from software.base import SnakefileContext, SnakefileResult, SoftwareAdapter

__all__ = ["DESCRIPTIONS_DIR", "SnakefileContext", "SnakefileResult", "SoftwareAdapter", "find_description",
           "get_adapter", "load_description", "resolve"]

#: benchmantic's own folder.
ROOT = Path(__file__).resolve().parents[1]
#: Software descriptions benchmantic knows (one YAML file per software).
DESCRIPTIONS_DIR = ROOT / "software_descriptions"
#: `software-description` value that asks the LLM for a new description.
ASK_LLM = "llm"


def _slug(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "-", name.lower()).strip("-")


def load_description(path_text: str) -> dict[str, Any]:
    """A description file: as given (absolute, or relative to the current
    folder), else relative to benchmantic's own folder."""
    from software.generic import load_spec

    for path in (Path(path_text).expanduser(), ROOT / path_text):
        if path.is_file():
            return load_spec(path, require_name=False)
    sys.exit(f"Error: software description {path_text} not found (looked in the current folder and in {ROOT}).")


def find_description(name: str) -> Path | None:
    """software_descriptions/<name>.yaml, or the file whose `name:` is
    `name` (case-insensitive)."""
    wanted = name.strip().lower()
    for path in sorted(DESCRIPTIONS_DIR.glob("*.yaml")):
        if path.stem.lower() == _slug(name):
            return path
        m = re.search(r"^name:\s*(.+?)\s*$", path.read_text(encoding="utf-8"), re.MULTILINE)
        if m and m.group(1).strip("'\"").lower() == wanted:
            return path
    return None


def _fitting_description(root: Path, args: Any) -> Path | None:
    """No software named: the first of benchmantic's own descriptions that
    fits the implementation, if any."""
    from software import autoconfig

    for path in sorted(DESCRIPTIONS_DIR.glob("*.yaml")):
        if not autoconfig.check(load_description(str(path)), root, args):
            return path
    return None


def resolve(root: Path, name: str | None, args: Any = None) -> SoftwareAdapter:
    """The adapter for the software named in the run config (`software:`),
    or given as a description file (`software-description:`)."""
    from software import autoconfig
    from software.generic import GenericAdapter

    name = None if name in (None, "", "auto") else str(name)
    description = getattr(args, "software_description", None)
    if name and name.endswith((".yaml", ".yml")):     # a description file given as `software:`
        name, description = None, description or name
    verbose = getattr(args, "verbose", False)

    if description and description != ASK_LLM:
        settings = load_description(str(description))
        if name and not settings.get("name"):
            settings["name"] = name
        source = str(description)
    else:
        known = find_description(name) if name and description != ASK_LLM else None
        if not name and not description:
            known = _fitting_description(root, args)
        settings = load_description(str(known)) if known else ({"name": name} if name else {})
        source = str(known.relative_to(ROOT)) if known else None

    if source and settings.get("name"):
        problems = autoconfig.check(settings, root, args)
        if not problems:
            if verbose:
                print(f"Software: {settings['name']} (description: {source})")
            return GenericAdapter.from_spec(settings)()
        if description:
            print(f"The software description {source} is incomplete or doesn't fit this implementation "
                  "-- the LLM fills in the rest:\n  - " + "\n  - ".join(problems))
        else:
            print(f"The software description {source} doesn't fit this implementation:\n  - "
                  + "\n  - ".join(problems))
            # benchmantic's own description doesn't fit: the LLM writes one,
            # with that description as the first example
            print(f"Asking the LLM for a description that fits (with {source} as an example).")
            settings = {"name": settings["name"]}
    return GenericAdapter.from_spec(autoconfig.complete(root, settings, args))()


def get_adapter(root: Path, name: str | None = "auto") -> SoftwareAdapter:
    """resolve() without run settings, for tools that only name the software."""
    return resolve(root, name)


def __getattr__(attr: str) -> Any:
    """`software.ADAPTERS`: one adapter class per description in
    software_descriptions/ (slug = file name), for tools and tests that
    pick one directly."""
    if attr == "ADAPTERS":
        from software.generic import GenericAdapter

        return [GenericAdapter.from_spec(load_description(str(p)), slug=p.stem)
                for p in sorted(DESCRIPTIONS_DIR.glob("*.yaml"))]
    raise AttributeError(f"module 'software' has no attribute {attr!r}")
