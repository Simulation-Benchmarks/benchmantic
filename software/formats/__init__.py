# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
software.formats

Readers for input file formats, chosen in the software description with
`inputs: {format: ...}` and shared by every software using that format.
Each format module provides:

  input_files(case_dir, settings)              the input files of one case
  discover(case_dir, settings)                 ParameterCandidates of a case
  resolve(case_dir, parameter_fields, settings)  a case's values
  read_entry(path, key)                        one value (e.g. the solver)

To support a new format, add a module here and list it in FORMATS.
"""

from __future__ import annotations

from types import ModuleType

from software.formats import foam_dict, ini, json_params, python_code

FORMATS: dict[str, ModuleType] = {
    "ini": ini,
    "foam_dict": foam_dict,
    "python": python_code,
    "json": json_params,
}


def get_format(name: str) -> ModuleType:
    try:
        return FORMATS[name]
    except KeyError:
        raise SystemExit(f"Error: unknown input format {name!r} in the software description; "
                         f"choose from {', '.join(FORMATS)}") from None
