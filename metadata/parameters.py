# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
metadata.parameters

Parameter discovery and extraction: scanning params.input for candidate
parameters, scraping getParam<Type>("Section.Key") C++ call-site hints,
selecting scenario-specific vs. global/constant parameters, resolving a
case's actual parameter values, and building the final RO-Crate field dicts
from LLM-inferred (or cached) metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils import parse_dune_ini, to_number

@dataclass
class ParameterCandidate:
    section: str
    key: str
    value: str
    tokens: list[str]
    #: Optional hint scraped from getParam<Type>("Section.Key") call sites in
    #: main.cc/problem.hh -- e.g. "assigned to `density_` (C++ type `Scalar`)".
    #: See scan_getparam_hints() / attach_cpp_hints(). Empty if none found.
    cpp_hint: str = ""
    #: A short window of the actual source code around that getParam<>()
    #: call site (a few hundred characters) -- NOT the whole file. This,
    #: together with cpp_hint, is what ai.prompts sends the LLM as
    #: code-grounded evidence INSTEAD OF embedding the full main.cc/
    #: problem.hh in every inference request (which used to dominate prompt
    #: size and regularly exceeded provider per-request token/payload
    #: limits). Empty if no getParam<>() call site was found for this
    #: candidate. See GETPARAM_CONTEXT_CHARS / scan_getparam_hints().
    code_context: str = ""


def discover_parameters(params_input: Path) -> list[ParameterCandidate]:
    ini = parse_dune_ini(params_input)
    return [
        ParameterCandidate(
            section=section,
            key=key,
            value=value,
            tokens=value.split(),
        )
        for section, kv in ini.items()
        for key, value in kv.items()
    ]


GETPARAM_PATTERN = re.compile(
    r'(?:(\w+)\s*=\s*)?'                       # optional "varname = "
    r'getParam(?:FromGroup)?\s*<(.+?)>\s*\('    # getParam<Type>( / getParamFromGroup<Type>(
    r'\s*(?:"[^"]*"\s*,\s*)?'                   # optional leading group-name arg
    r'"([^"]+)"'                                # the "Section.Key" string
)


#: Chars of raw source grabbed on each side of a getParam<>() call site for
#: ParameterCandidate.code_context -- deliberately small (this is meant as
#: a targeted snippet, not a substitute for the old full-file dump).
GETPARAM_CONTEXT_CHARS = 120


def scan_getparam_hints(*source_texts: str) -> dict[str, tuple[str, str]]:
    """Scan one or more C++ source files for getParam<Type>("Section.Key")
    call sites and return a dict keyed by lowercased "section.key" ->
    (human-readable hint, small code excerpt around the call site).
    """
    hints: dict[str, tuple[str, str]] = {}
    for text in source_texts:
        if not text:
            continue
        for match in GETPARAM_PATTERN.finditer(text):
            var, cpp_type, ini_key = match.group(1), match.group(2).strip(), match.group(3)
            if var:
                hint = f"assigned to variable `{var}` (C++ type `{cpp_type}`)"
            else:
                hint = f"read with C++ type `{cpp_type}`"
            start = max(0, match.start() - GETPARAM_CONTEXT_CHARS)
            end = min(len(text), match.end() + GETPARAM_CONTEXT_CHARS)
            hints[ini_key.lower()] = (hint, text[start:end].strip())
    return hints


def attach_cpp_hints(candidates: list[ParameterCandidate], *source_texts: str) -> None:
    """Mutate `candidates` in place, filling in cpp_hint/code_context for
    any candidate whose "Section.Key" matches a getParam<>() call site
    found in the given source texts (typically main.cc and problem.hh).
    """
    hints = scan_getparam_hints(*source_texts)
    for c in candidates:
        found = hints.get(f"{c.section}.{c.key}".lower())
        if found:
            c.cpp_hint, c.code_context = found


# ============================================================
# Benchmark-level description (Doxygen doc-comment) extraction
#
# problem.hh (and sometimes main.cc) usually opens its class/problem
# definition with a /*! \brief ... */ Doxygen block describing the physical
# scenario and often citing the source publication, e.g.:
#   /*!
#    * \brief Test problem for the (Navier-) Stokes model in a 3D channel
#    * Benchmark case from Turek, Schaefer et al (1996) ...
#    */
# That's high-value context for both parameter and metric inference (it
# tells the LLM *what benchmark this is*, not just what code surrounds each
# value), and doubles as a human-readable description for the benchmark
# graph node. We grab the longest such block as a heuristic for "the main
# one" (per-function doc-comments are usually much shorter).
# ============================================================


DEFAULT_SCENARIO_SECTIONS: set[str] = {"domain", "grid", "problem"}


def default_scenario_candidates(
    candidates: list[ParameterCandidate],
    sections: set[str] = DEFAULT_SCENARIO_SECTIONS,
) -> list[ParameterCandidate]:
    """Return the subset of candidates whose parent section is scenario-specific.

    A parameter is treated as scenario-specific by default when its parent
    INI section (matched case-insensitively) is one of `sections`, e.g.
    [Domain], [Grid], or [Problem]. Falls back to *all* candidates if none
    match, so callers never end up with an empty default selection.
    """
    wanted = {s.lower() for s in sections}
    selected = [c for c in candidates if c.section.lower() in wanted]
    return selected or list(candidates)


# ============================================================
# Pretty printing
# ============================================================


def extract_token(ini: dict, section: str, key: str, index: int) -> str:
    if section not in ini or key not in ini[section]:
        raise KeyError(f"[{section}] {key} not found in params.input")
    raw_value = ini[section][key]
    tokens = raw_value.split()
    if not tokens:
        raise ValueError(f"[{section}] {key} has no value")
    if not (-len(tokens) <= index < len(tokens)):
        raise ValueError(
            f"[{section}] {key} = '{raw_value}' has {len(tokens)} token(s) "
            f"(0-based indices 0..{len(tokens) - 1}), but the parameter metadata "
            f"requests token index {index}. This usually means either: (a) the "
            f"LLM picked an index that doesn't exist in the template params.input "
            f"it was shown -- check/edit the cached '.parameter_metadata_cache.json' "
            f"entry for [{section}] {key}, or (b) this specific case's params.input "
            f"has a different number of values for [{section}] {key} than the "
            f"template did -- check that file directly."
        )
    return tokens[index]


def extract_all_tokens(ini: dict, section: str, key: str) -> list[str]:
    """Like extract_token(), but returns every whitespace-separated token
    in the value instead of picking just one by index -- for parameters
    whose INI entry genuinely holds multiple numbers (e.g. 'Radial0 = 1.0
    1.5 2.0') where collapsing to a single value would silently discard the
    rest. See --full-value-params.
    """
    if section not in ini or key not in ini[section]:
        raise KeyError(f"[{section}] {key} not found in params.input")
    tokens = ini[section][key].split()
    if not tokens:
        raise ValueError(f"[{section}] {key} has no value")
    return tokens



def resolve_case_params(case_dir: Path, parameter_fields: dict[str, Any]) -> dict[str, Any]:
    ini = parse_dune_ini(case_dir / "params.input")
    values: dict[str, Any] = {}
    for key, spec in parameter_fields.items():
        section, ini_key = spec["ini"]
        try:
            if spec.get("full_value"):
                tokens = extract_all_tokens(ini, section, ini_key)
                values[key] = [to_number(t) for t in tokens]
            else:
                token = extract_token(ini, section, ini_key, spec["index"])
                values[key] = to_number(token)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"In case '{case_dir}': {exc}") from exc
    return values



def build_parameter_fields(metadata: list[dict]) -> dict:
    """Keyed by the parameter's own (section, key) identity from
    params.input -- NOT by the LLM's semantic_name. Two different real
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


