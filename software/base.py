# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
software.base

The interface every supported simulation code implements. The shared
pipeline (metadata.builder, snakefile.generator, ai.*) only talks to a
SoftwareAdapter; everything that depends on how a code stores its inputs,
writes its metrics, lays out its cases, is built and is run lives in the
code's own folder under software/.

An input parameter is identified by a (section, key) pair throughout the
pipeline (caches, LLM inference, review, graph, mapping). What the pair
means is up to the adapter -- e.g. an INI [Section] + key for DuMux, or a
dictionary file + entry path for OpenFOAM.

software.generic.GenericAdapter implements this for any software config
(software/template.yaml); software.resolve() builds it from the run config.
"""

from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from metadata.metrics import MetricCandidate
from metadata.parameters import ParameterCandidate


@dataclass
class SnakefileContext:
    """Everything snakefile.generator has gathered for one benchmark
    description, handed to SoftwareAdapter.render_snakefile()."""
    #: Parsed command-line options (shared ones plus every adapter's own).
    args: argparse.Namespace
    #: The description file the Snakefile is generated from.
    metadata_jsonld: Path
    #: The description's @graph, by @id.
    by_id: dict[str, dict[str, Any]]
    #: The build_hints sidecar (see SoftwareAdapter.build_hints()).
    build_hints: dict[str, Any]
    #: {case id: {parameter label: value}}
    cases: dict[str, dict[str, Any]]
    #: {parameter label: QUDT unit}
    units: dict[str, str]
    #: QUDT unit -> symbol used in parameters.json keys.
    unit_symbols: dict[str, str]
    #: The input mapping (*_mapping.json), or None.
    mapping: dict[str, Any] | None = None
    #: Where the mapping was read from, or None.
    mapping_file: Path | None = None


@dataclass
class SnakefileResult:
    text: str
    #: Lines printed after the files are written (what was generated).
    summary: list[str] = field(default_factory=list)
    #: Appended to the "Wrote <Snakefile>" line.
    note: str = ""


class SoftwareAdapter:
    #: Software label written to the description ("DuMux", "OpenFOAM", ...).
    name: str = "unknown"
    #: CLI value for --software. Defaults to the folder name.
    slug: str = ""
    #: How the input files are called in messages and LLM prompts.
    input_label: str = "input files"
    #: Human-readable description of the default parameter selection.
    default_selection_label: str = "all parameters"
    #: Default for --container-shared-dir when the standalone Snakefile
    #: generator isn't given one (None: no default).
    default_container_shared_dir: str | None = None

    # --- discovery -----------------------------------------------------
    @classmethod
    def detect(cls, root: Path) -> bool:
        """True if `root` (a repo checkout or module folder) looks like a
        benchmark implemented in this software."""
        raise NotImplementedError

    def find_module_dir(self, root: Path, args: Any) -> Path:
        """The directory holding the benchmark implementation."""
        raise NotImplementedError

    def source_texts(self, module_dir: Path) -> dict[str, str]:
        """Source files used for license/label/description extraction and
        as fallback LLM context: {file name: text}."""
        return {}

    def discover_parameters(self, module_dir: Path) -> list[ParameterCandidate]:
        """All input parameters with their template values, and code hints
        attached where available."""
        raise NotImplementedError

    def default_scenario_candidates(self, candidates: list[ParameterCandidate]) -> list[ParameterCandidate]:
        """The parameters pre-selected as case-varying."""
        return list(candidates)

    def discover_metrics(self, module_dir: Path, repo_root: Path) -> list[MetricCandidate]:
        """Output metrics (keys of the metrics file). Exits with a helpful
        message if none are found."""
        raise NotImplementedError

    def discover_cases(self, module_dir: Path) -> list[tuple[Path, str]]:
        """[(case_dir, case_id), ...]"""
        raise NotImplementedError

    def resolve_case_params(self, case_dir: Path, parameter_fields: dict[str, Any]) -> dict[str, Any]:
        """{field key: value} for one case, for the selected parameters."""
        raise NotImplementedError

    def input_display(self, section: str, key: str) -> str:
        """How one input is written for people (review screens, mapping
        file, Snakefile comments)."""
        return f"{section}:{key}"

    # --- descriptive metadata -----------------------------------------
    def benchmark_description(self, module_dir: Path, texts: dict[str, str]) -> str:
        """A human-readable description from the sources ("" if none; the
        builder falls back to the README)."""
        return ""

    def benchmark_label(self, module_dir: Path, repo_root: Path, texts: dict[str, str]) -> str | None:
        """A name for the benchmark, if the sources carry one."""
        return None

    def build_hints(self, module_dir: Path, repo_root: Path) -> dict[str, Any]:
        """Extra facts the Snakefile needs (written to the build_hints
        sidecar, not to the description)."""
        return {}

    def executable_name(self, module_dir: Path, repo_root: Path) -> tuple[str | None, Path | None]:
        """(executable/solver name, file it was read from)."""
        return None, None

    # --- workflow --------------------------------------------------------------
    def render_snakefile(self, ctx: SnakefileContext) -> SnakefileResult:
        """The Snakefile text for this benchmark. snakefile.generator
        writes it, the optional zip and the per-case parameters.json."""
        raise NotImplementedError(f"{self.name}: Snakefile generation is not implemented yet.")

    # --- LLM context ------------------------------------------------------------
    def prompt_context(self) -> str:
        """Software-specific notes for the inference prompts: what a
        (section, key) pair is, what cpp_hint/code_context contain, and
        naming conventions that imply units. Read from prompt.md in the
        software's folder."""
        return self.folder_file("prompt.md").strip()

    # --- helpers ----------------------------------------------------------------
    @classmethod
    def folder(cls) -> Path:
        """This software's folder (software/<name>/)."""
        return Path(inspect.getfile(cls)).parent

    @classmethod
    def folder_file(cls, name: str) -> str:
        path = cls.folder() / name
        return path.read_text(encoding="utf-8") if path.is_file() else ""
