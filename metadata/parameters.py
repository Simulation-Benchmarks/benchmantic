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

import re

from dataclasses import asdict, dataclass, field


@dataclass
class ParameterLocation:
    """Where one parameter sits inside its input file. Lines and columns
    are 1-based; columns count characters on that line."""
    #: The input file, relative to the case folder (e.g. "params.input",
    #: "constant/MRFProperties").
    file: str
    #: Line of the key.
    line: int
    #: Column where the value starts on that line.
    column: int
    #: (start column, end column) of each value token, end exclusive --
    #: so a multi-value entry like "Radial0 = 1.0 1.5 2.0" can point at
    #: the one token a parameter uses (its "index").
    token_spans: list[tuple[int, int]] = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        d["token_spans"] = [list(span) for span in self.token_spans]
        return d


def token_spans(line_text: str, value_start: int, value: str) -> list[tuple[int, int]]:
    """1-based (start, end) columns of the whitespace-separated tokens of
    `value`, found on `line_text` from 0-based offset `value_start`.
    Parentheses are skipped, like the readers' own tokenisation."""
    spans = []
    for m in re.finditer(r"[^\s()]+", value):
        start = line_text.find(m.group(), value_start)
        if start < 0:
            break
        spans.append((start + 1, start + 1 + len(m.group())))
        value_start = start + len(m.group())
    return spans


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
    #: Where the parameter was found in its input file (None if the
    #: reader can't tell).
    location: ParameterLocation | None = None




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
