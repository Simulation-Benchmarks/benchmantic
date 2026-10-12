# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
software._template.adapter

Starting point for a new simulation code. After copying the folder to
software/<name>/:

  1. rename the class, set `name` (the label written to the description),
     and replace `_template` in the imports with your folder name;
  2. implement the methods marked TODO -- see software/base.py for the full
     contract and software/dumux/ and software/openfoam/ for two complete
     examples (a compiled C++ code with an INI file, and a case-directory
     code with dictionary files);
  3. write snakefile.py (how one case is run) and prompt.md (what an input's
     (section, key) means, for the LLM);
  4. add a small example under tests/fixtures/<name>/ -- tests/test_software.py
     then checks the adapter against it automatically.

Reusable shared helpers:
  - metadata.parameters.ParameterCandidate  -- one discovered input
  - metadata.metrics.MetricCandidate        -- one discovered output metric
  - metadata.metrics.discover_metrics_in_python(text)
        metric keys of a dict a Python script json.dump()s to
        solution_metrics.json (handy for Python-based codes)
  - metadata.metrics.discover_metrics(text) -- escaped "key": JSON in source
  - metadata.repository.find_repo_file(...) -- look upward for a file
  - utils.read_text, utils.to_number
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from metadata.metrics import MetricCandidate
from metadata.parameters import ParameterCandidate
from software._template import snakefile
from software.base import SnakefileContext, SnakefileResult, SoftwareAdapter


class TemplateAdapter(SoftwareAdapter):
    name = "MyCode"                      # TODO: label in the description, e.g. "FEniCS"
    input_label = "the input file"       # TODO: how the inputs are called in messages
    default_selection_label = "all parameters"

    # --- discovery (required) --------------------------------------------------
    @classmethod
    def detect(cls, root: Path) -> bool:
        # TODO: True if `root` holds an implementation in this code, e.g.
        # a *.py file importing dolfinx. Keep it specific: two adapters
        # recognising the same folder is an error.
        raise NotImplementedError

    def find_module_dir(self, root: Path, args: Any) -> Path:
        # TODO: the folder with the implementation, searched under `root`.
        raise NotImplementedError

    def discover_parameters(self, module_dir: Path) -> list[ParameterCandidate]:
        # TODO: one ParameterCandidate(section, key, value, tokens) per input,
        # with cpp_hint/code_context set to where the value is read.
        raise NotImplementedError

    def discover_metrics(self, module_dir: Path, repo_root: Path) -> list[MetricCandidate]:
        # TODO: the keys the code (or its post-processing) writes to its
        # metrics file; sys.exit() with a helpful message if there are none.
        raise NotImplementedError

    def discover_cases(self, module_dir: Path) -> list[tuple[Path, str]]:
        # TODO: [(case_dir, case_id), ...] -- one per configuration.
        raise NotImplementedError

    def resolve_case_params(self, case_dir: Path, parameter_fields: dict[str, Any]) -> dict[str, Any]:
        # TODO: {field key: value} for one case. parameter_fields[k]["ini"]
        # is the (section, key) pair from discover_parameters();
        # "full_value"/"index" say whether to keep every token or one.
        raise NotImplementedError

    # --- optional (defaults in software/base.py) -------------------------------
    # source_texts, default_scenario_candidates, input_display,
    # benchmark_description, benchmark_label, build_hints, executable_name,
    # add_arguments, add_snakefile_arguments, default_container_shared_dir

    # --- workflow ------------------------------------------------------------------
    def render_snakefile(self, ctx: SnakefileContext) -> SnakefileResult:
        return snakefile.render(ctx)
