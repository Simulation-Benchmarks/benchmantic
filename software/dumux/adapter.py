# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
software.dumux.adapter

DuMuX / DUNE: a module folder with main.cc + problem.hh, parameters in a
DUNE INI file (params.input) read via getParam<T>("Section.Key"), metrics
written from main.cc as JSON keys, one params.input per case, a CMake
test/executable target run with -Section.Key command-line flags.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from software.base import SnakefileContext, SnakefileResult, SoftwareAdapter
from software.dumux import snakefile
from software.dumux.inputs import (
    DEFAULT_SCENARIO_SECTIONS,
    attach_cpp_hints,
    default_scenario_candidates,
    discover_parameters,
    resolve_case_params,
)
from software.dumux.sources import (
    discover_cases,
    discover_metrics_from_maincc,
    extract_benchmark_description,
    extract_class_label,
    find_executable_name,
    find_module_dir,
)
from utils import read_text


class DumuxAdapter(SoftwareAdapter):
    name = "DuMux"
    slug = "dumux"
    input_label = "params.input"
    default_selection_label = "parameters under [" + ", ".join(sorted(DEFAULT_SCENARIO_SECTIONS)) + "]"
    default_container_shared_dir = "/dumux/shared"

    @classmethod
    def detect(cls, root: Path) -> bool:
        if (root / "main.cc").exists() and (root / "problem.hh").exists():
            return True
        return any((p.parent / "problem.hh").exists() for p in root.rglob("main.cc"))

    def find_module_dir(self, root: Path, args: Any) -> Path:
        return find_module_dir(root, getattr(args, "main_cc", None))

    def source_texts(self, module_dir: Path) -> dict[str, str]:
        return {
            "problem.hh": read_text(module_dir / "problem.hh"),
            "main.cc": read_text(module_dir / "main.cc"),
        }

    def _template_params_input(self, module_dir: Path) -> Path:
        path = module_dir / "params.input"
        if path.exists():
            return path
        discovered = sorted(module_dir.rglob("params.input"))
        if not discovered:
            raise SystemExit(f"Error: Could not find a template params.input under {module_dir}")
        return discovered[0]

    def discover_parameters(self, module_dir: Path):
        candidates = discover_parameters(self._template_params_input(module_dir))
        texts = self.source_texts(module_dir)
        attach_cpp_hints(candidates, texts["main.cc"], texts["problem.hh"])
        return candidates

    def default_scenario_candidates(self, candidates):
        return default_scenario_candidates(candidates)

    def discover_metrics(self, module_dir: Path, repo_root: Path):
        return discover_metrics_from_maincc(module_dir / "main.cc")

    def discover_cases(self, module_dir: Path):
        return discover_cases(module_dir)

    def resolve_case_params(self, case_dir: Path, parameter_fields: dict[str, Any]):
        return resolve_case_params(case_dir, parameter_fields)

    def input_display(self, section: str, key: str) -> str:
        return f"{section}.{key}"

    def benchmark_description(self, module_dir: Path, texts: dict[str, str]) -> str:
        return extract_benchmark_description(texts.get("problem.hh", ""), texts.get("main.cc", ""))

    def benchmark_label(self, module_dir: Path, repo_root: Path, texts: dict[str, str]) -> str | None:
        return extract_class_label(texts.get("problem.hh", ""), texts.get("main.cc", ""))

    def executable_name(self, module_dir: Path, repo_root: Path):
        return find_executable_name(module_dir, repo_root)

    # --- command-line options and workflow -------------------------------------
    @classmethod
    def add_arguments(cls, group: argparse._ArgumentGroup) -> None:
        group.add_argument("--main-cc", type=Path, default=None, dest="main_cc",
                           help="Explicit path to main.cc, only needed if more than one main.cc+problem.hh "
                                "pair is found under module_dir.")

    @classmethod
    def add_snakefile_arguments(cls, group: argparse._ArgumentGroup) -> None:
        snakefile.add_snakefile_arguments(group)

    def render_snakefile(self, ctx: SnakefileContext) -> SnakefileResult:
        return snakefile.render(ctx)
