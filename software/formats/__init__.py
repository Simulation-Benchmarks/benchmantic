# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
software.formats

Readers for input file formats, chosen in software.yaml with
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

from software.formats import foam_dict, ini

FORMATS: dict[str, ModuleType] = {
    "ini": ini,
    "foam_dict": foam_dict,
}


def get_format(name: str) -> ModuleType:
    try:
        return FORMATS[name]
    except KeyError:
        raise SystemExit(f"Error: unknown input format {name!r} in software.yaml; "
                         f"choose from {', '.join(FORMATS)}") from None
