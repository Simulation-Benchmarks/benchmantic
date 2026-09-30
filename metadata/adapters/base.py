# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
metadata.adapters.base

The interface every supported simulation code implements. The builder
only talks to a SoftwareAdapter; everything that depends on how a code
stores its inputs, writes its metrics or lays out its cases lives in the
adapter.

An input parameter is identified by a (section, key) pair throughout the
pipeline (caches, LLM inference, review, graph, mapping). What the pair
means is up to the adapter -- e.g. an INI [Section] + key for DuMuX, or a
dictionary file + entry path for OpenFOAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from metadata.metrics import MetricCandidate
from metadata.parameters import ParameterCandidate


class SoftwareAdapter:
    #: Software label written to the description ("DuMux", "OpenFOAM", ...).
    name: str = "unknown"
    #: CLI value for --software.
    slug: str = "unknown"
    #: How the input files are called in messages and LLM prompts.
    input_label: str = "input files"
    #: Human-readable description of the default parameter selection.
    default_selection_label: str = "all parameters"

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

    # --- descriptive metadata -----------------------------------------
    def benchmark_description(self, module_dir: Path, texts: dict[str, str]) -> str:
        """A human-readable description from the sources ("" if none; the
        builder falls back to the README)."""
        return ""

    def benchmark_label(self, module_dir: Path, repo_root: Path, texts: dict[str, str]) -> str | None:
        """A name for the benchmark, if the sources carry one."""
        return None

    def build_hints(self, module_dir: Path, repo_root: Path) -> dict[str, Any]:
        """Extra facts the Snakefile generator needs (written to the
        build_hints sidecar, not to the description)."""
        return {}

    def executable_name(self, module_dir: Path, repo_root: Path) -> tuple[str | None, Path | None]:
        """(executable/solver name, file it was read from)."""
        return None, None
