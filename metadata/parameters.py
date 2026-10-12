# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
metadata.parameters

The software-neutral parameter types: ParameterCandidate (one input a
software adapter discovered -- see software/<name>/) and the RO-Crate field
dicts built from LLM-inferred (or cached) metadata.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParameterCandidate:
    section: str
    key: str
    value: str
    tokens: list[str]
    #: Optional hint the software adapter scraped from the source about
    #: where and how this input is read -- e.g. for DuMux "assigned to
    #: `density_` (C++ type `Scalar`)". Empty if none found.
    cpp_hint: str = ""
    #: A short window of the actual source around that spot (a few hundred
    #: characters) -- NOT the whole file. This, together with cpp_hint, is
    #: what ai.prompts sends the LLM as code-grounded evidence instead of
    #: embedding whole source files in every inference request (which used
    #: to dominate prompt size and exceed provider per-request limits).
    code_context: str = ""




def build_parameter_fields(metadata: list[dict]) -> dict:
    """Keyed by the parameter's own (section, key) identity from
    the software's inputs -- NOT by the LLM's semantic_name. Two different real
    parameters can never collide here even if the model (accidentally or
    not) assigns them the same semantic_name; semantic_name/explanation are
    carried along as descriptive metadata instead of used as the identity.
    """
    fields = {}
    for item in metadata:
        section, ini_key = item["ini"]
        stable_key = f"{section}::{ini_key}"
        fields[stable_key] = {
            "ini": (section, ini_key),
            "index": item.get("index", 0),
            "unit": item["unit"],
            "quantityKind": item.get("quantityKind"),
            "datatype": item["datatype"],
            "semantic_name": item["semantic_name"],
            "description": item.get("explanation", ""),
            #: When set, resolve_case_params() captures every whitespace-
            #: separated token of this parameter's raw INI value as a list,
            #: instead of collapsing it to a single value via "index" --
            #: see generate_metadata.py's --full-value-params.
            "full_value": item.get("full_value", False),
        }
    return fields
