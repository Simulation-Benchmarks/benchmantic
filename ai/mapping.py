# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
ai.mapping

Asks the LLM how a simulation code's inputs are set from a reference
benchmark's software-neutral parameters -- one expression per input (see
metadata.mapping for the expression format and the checks applied).
"""

from __future__ import annotations

import json
from typing import Any

from ai.inference import DEFAULT_PROVIDER, DEFAULT_TPM_BUDGET, _query_llm_json_array
from metadata.mapping import check_expression

MAPPING_SYSTEM_PROMPT = """\
You are an expert in numerical simulation software (DuMux, OpenFOAM, FEniCS,
deal.II, ...) and in benchmark definitions.

A benchmark is defined with SOFTWARE-NEUTRAL parameters (e.g. inner_radius,
cells_radial). One simulation code implements it through its OWN inputs
(e.g. DuMux "Grid.Cells0", OpenFOAM "system/blockMeshDict: geom/nr"). Your
task: for every software input listed, say how its value is computed from
the benchmark parameters.

Answer with an "expression" per input: a Python f-string BODY (no leading
f, no surrounding quotes) whose {...} fields contain arithmetic over
benchmark parameter names. Examples:
  "{angular_velocity_inner_cylinder}"                       one value
  "{cells_radial // 2} {cells_radial // 2}"                 two mesh zones, half the cells each
  "{inner_radius} {(inner_radius + outer_radius) / 2} {outer_radius}"
  "{mesh_grading} -{mesh_grading}"                          grading mirrored in the second zone
  "{cells_angular // 4}"                                    mesh made of 4 blocks
Allowed inside {...}: benchmark parameter names, numbers, + - * / // % **,
parentheses, and int() float() round() abs() min() max() sqrt() len(). A
benchmark parameter holding several values (e.g. "1.0 1.5 2.0") can be
indexed with tokens(): {tokens(radial_coordinates)[0]} is its first value,
{tokens(radial_coordinates)[-1]} its last. Nothing else.

Use "expression": null when the input is NOT determined by any benchmark
parameter (solver settings, output names, physics switches the benchmark
doesn't vary) -- it then keeps its own value. Never invent a benchmark
parameter; only use names from the list you are given.

Each input comes with its reviewed "semantic_name", "unit" and
"quantity_kind" (what the software input means, confirmed by a human) --
match primarily on that meaning and on the units. Use the template value and
code context as further evidence: e.g. a template value "40 40" for a cell
count, when the benchmark has cells_radial, means two zones sharing the
radial cells. If the input's unit differs from the benchmark parameter's
(e.g. degrees vs radians), include the conversion in the expression.

Respond with a raw JSON array only (first character '[', last ']'), one
object per input, no markdown fences, no prose:
[{"ini": ["<section>", "<key>"], "expression": "<string or null>",
  "confidence": 0.0-1.0, "explanation": "<one sentence>"}]
"""

MAPPING_PROMPT_TEMPLATE = """\
=====================
benchmark
=====================
{benchmark_description}

=====================
benchmark parameters (software-neutral) -- the ONLY names allowed in expressions
=====================
{parameters_json}

=====================
{software} inputs -- return EXACTLY these {n_items} item(s), same "ini" pairs
=====================
{inputs_json}

Return the JSON array now.
"""


def build_mapping_prompt(candidates, reference_parameters: dict[str, dict[str, Any]],
                         example_values: dict[str, Any], software: str, benchmark_description: str,
                         semantics: dict | None = None) -> str:
    semantics = semantics or {}
    params = [
        {"name": name, "unit": info.get("unit"), "quantityKind": info.get("quantityKind"),
         "example_value": example_values.get(name), "description": info.get("description") or None}
        for name, info in reference_parameters.items()
    ]
    inputs = [
        {"ini": [c.section, c.key],
         "semantic_name": (semantics.get((c.section, c.key)) or {}).get("semantic_name"),
         "unit": (semantics.get((c.section, c.key)) or {}).get("unit"),
         "quantity_kind": (semantics.get((c.section, c.key)) or {}).get("quantityKind"),
         "meaning": (semantics.get((c.section, c.key)) or {}).get("explanation") or None,
         "template_value": c.value, "hint": c.cpp_hint or None,
         "code_context": c.code_context or None}
        for c in candidates
    ]
    return MAPPING_PROMPT_TEMPLATE.format(
        benchmark_description=benchmark_description or "(none)",
        parameters_json=json.dumps(params, indent=2),
        software=software,
        n_items=len(candidates),
        inputs_json=json.dumps(inputs, indent=2),
    )


def validate_mapping(data: Any, candidates, parameter_names: set[str]) -> list[dict]:
    """Exactly one item per candidate; expressions must pass
    metadata.mapping.check_expression. Raises ValueError (-> retry)."""
    if not isinstance(data, list):
        raise ValueError("Model returned something other than a JSON list.")
    wanted = {(c.section, c.key) for c in candidates}
    by_key: dict[tuple[str, str], dict] = {}
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("ini"), list) or len(item["ini"]) != 2:
            raise ValueError("Each item must be an object with 'ini': [section, key].")
        key = (str(item["ini"][0]), str(item["ini"][1]))
        if key not in wanted:
            continue  # unrequested item: dropped
        expr = item.get("expression")
        if expr in ("", "null", "None"):
            expr = None
        if expr is not None:
            if not isinstance(expr, str):
                raise ValueError(f"'expression' for {key} must be a string or null.")
            problem = check_expression(expr, parameter_names)
            if problem:
                raise ValueError(f"Invalid expression for {key[0]} {key[1]}: {problem}")
        by_key[key] = {
            "ini": list(key), "expression": expr,
            "confidence": float(item.get("confidence") or 0.0),
            "explanation": str(item.get("explanation") or ""),
        }
    missing = wanted - set(by_key)
    if missing:
        raise ValueError("Model did not return a mapping for: " + ", ".join(f"{s} {k}" for s, k in sorted(missing)))
    return [by_key[(c.section, c.key)] for c in candidates]


def infer_parameter_mapping(
    *,
    candidates,
    semantics: dict | None = None,
    reference_parameters: dict[str, dict[str, Any]],
    example_values: dict[str, Any],
    software: str,
    benchmark_description: str = "",
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
    verbose: bool = False,
    debug: bool = False,
    tpm_budget: int = DEFAULT_TPM_BUDGET,
    allow_model_fallback: bool = True,
) -> list[dict]:
    names = set(reference_parameters)
    return _query_llm_json_array(
        provider=provider,
        model=model,
        system_prompt=MAPPING_SYSTEM_PROMPT,
        prompt=build_mapping_prompt(candidates, reference_parameters, example_values, software,
                                    benchmark_description, semantics),
        item_count=len(candidates),
        item_label="input mapping",
        validator=lambda data: validate_mapping(data, candidates, names),
        retries=3,
        retry_delay=2.0,
        verbose=verbose,
        debug=debug,
        tpm_budget=tpm_budget,
        allow_model_fallback=allow_model_fallback,
    )
