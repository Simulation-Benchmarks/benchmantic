"""
ai.prompts

LLM system/user prompt templates for parameter and metric metadata
inference, and the functions that fill them in with the discovered
candidates plus source-code context.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metadata.metrics import MetricCandidate
    from metadata.parameters import ParameterCandidate

SYSTEM_PROMPT = """\
You are an expert in Computational Fluid Dynamics, DuMux, OpenFOAM,
scientific metadata, QUDT units, and JSON-LD.

Your task is to infer semantic metadata from DUNE params.input files.

For every parameter infer:
  - semantic_name
  - datatype  (schema:Integer | schema:Float | schema:String)
  - unit      (QUDT unit identifier)
  - quantityKind (QUDT quantityKind URI, or null if unknown)
  - index     (token index within the value string)
  - confidence
  - explanation

CRITICAL -- "unit" and "quantityKind" are two DIFFERENT namespaces and must
NEVER be confused:
  - "unit" is a UNIT identifier, always of the short form "unit:XXX", e.g.
    unit:M, unit:RAD, unit:RAD-PER-SEC, unit:PA, unit:KiloGM-PER-M3,
    unit:SEC, unit:UNITLESS. This is what "has numerical value" is measured
    IN -- i.e. the value 1.1 with unit unit:M means "1.1 metres".
  - "quantityKind" is a QUANTITY KIND URI of the long form
    "http://qudt.org/vocab/quantitykind/XXX", e.g.
    http://qudt.org/vocab/quantitykind/Length,
    http://qudt.org/vocab/quantitykind/Angle,
    http://qudt.org/vocab/quantitykind/AngularVelocity. This is the
    CATEGORY of physical quantity, not a measurable unit.
  - NEVER put a quantitykind URI in the "unit" field, and NEVER put a
    "unit:XXX" identifier in the "quantityKind" field. If you find yourself
    about to write "unit": "http://qudt.org/vocab/quantitykind/..." STOP --
    that is always wrong. Pick the actual QUDT unit instead, e.g.:
      quantityKind Length          -> unit unit:M
      quantityKind Angle           -> unit unit:RAD (or unit:DEG if the
                                       value is clearly in degrees)
      quantityKind AngularVelocity -> unit unit:RAD-PER-SEC
      quantityKind Mass            -> unit unit:KiloGM
      quantityKind Time            -> unit unit:SEC
      quantityKind Pressure        -> unit unit:PA
      quantityKind DimensionlessRatio / Count -> unit unit:UNITLESS

Some parameter entries include a "cpp_hint" field: this is scraped directly
from the source code's getParam<Type>("Section.Key") call site (e.g. the
exact C++ variable name and type it is assigned to, such as `density_` of
type `Scalar`). Treat cpp_hint as strong, code-grounded evidence of the
parameter's physical meaning and prefer it over guessing from the INI key
name alone -- e.g. a variable named `density_`/`rho_` implies kg/m3, a
variable read as `omega_`/an angular velocity implies rad/s, `viscosity_`
(dynamic) implies Pa*s, `radius_`/`length_` implies m.

A benchmark description (a doc-comment from the source, if available) and
the full main.cc/problem.hh source are provided only as background context
to help you understand what the parameters mean -- e.g. what physical
scenario or published benchmark this is. They will typically mention many
more parameters, functions, and variables than the ones you were asked
about. Do NOT infer metadata for anything other than the exact parameters
listed under "params.input" below.

STRICT OUTPUT RULE: the "params.input" section below lists exactly the
parameters you must return metadata for -- one JSON object per entry, same
section/key pairs, nothing added and nothing omitted. Do not include
parameters you merely noticed in main.cc, problem.hh, or the benchmark
description; only the ones explicitly listed under "params.input" below.

Respond with a raw JSON array only: the first character of your response must
be '[' and the last character must be ']'. Do not wrap the JSON in markdown
code fences (no ``` of any kind). Do not include any prose, explanation, or
preamble before or after the JSON.
"""

PROMPT_TEMPLATE = """\
The benchmark contains these files.

=====================
benchmark description
=====================

{benchmark_description}

=====================
params.input -- infer metadata for EXACTLY these {n_items} parameter(s), no others
=====================

{parameter_json}

=====================
main.cc (background context only -- do not infer metadata for anything here
that isn't also listed under params.input above)
=====================

{main_cc}

=====================
problem.hh (background context only -- do not infer metadata for anything
here that isn't also listed under params.input above)
=====================

{problem_hh}

Infer semantic metadata for every parameter listed under params.input above
-- exactly {n_items} item(s), no more, no less.

Return JSON like:

[
  {{
    "semantic_name": "cells_radial",
    "ini": ["Grid", "Cells0"],
    "index": 0,
    "datatype": "schema:Integer",
    "unit": "unit:UNITLESS",
    "quantityKind": "[qudt.org](http://qudt.org/vocab/quantitykind/Count)",
    "confidence": 0.99,
    "explanation": "..."
  }}
]

Return ONLY the raw JSON array. No markdown code fences, no explanation, no
text before the opening '[' or after the closing ']'.
"""


def build_prompt(
    candidates: list[ParameterCandidate],
    main_cc: str,
    problem_hh: str,
    benchmark_description: str = "",
) -> str:
    return PROMPT_TEMPLATE.format(
        benchmark_description=benchmark_description or "(none found)",
        n_items=len(candidates),
        parameter_json=json.dumps([asdict(c) for c in candidates], indent=2),
        main_cc=main_cc,
        problem_hh=problem_hh,
    )


# ------------------------------------------------------------
# Metric (output/solution quantity) prompts
# ------------------------------------------------------------

METRIC_SYSTEM_PROMPT = """\
You are an expert in Computational Fluid Dynamics, DuMux, OpenFOAM,
scientific metadata, QUDT units, and JSON-LD.

Your task is to infer semantic metadata for SOLUTION METRICS: output
quantities a CFD simulation writes to a results/summary file, as opposed to
input parameters.

For every metric infer:
  - semantic_name
  - datatype     (schema:Integer | schema:Float | schema:Double | schema:String)
  - unit         (QUDT unit identifier, using SI units for dimensional
                   quantities, e.g. unit:PA for pressure, unit:M-PER-SEC for
                   velocity, unit:SEC for time, unit:N for force)
  - quantityKind (QUDT quantityKind URI, or null if unknown)
  - confidence
  - explanation

CRITICAL -- "unit" and "quantityKind" are two DIFFERENT namespaces and must
NEVER be confused: "unit" is always the short form "unit:XXX" (e.g. unit:PA,
unit:M-PER-SEC, unit:UNITLESS); "quantityKind" is always the long form
"http://qudt.org/vocab/quantitykind/XXX" (e.g.
http://qudt.org/vocab/quantitykind/Pressure). NEVER put a quantitykind URI
in the "unit" field or a "unit:XXX" identifier in the "quantityKind" field.

Rules for choosing units:
  - Always prefer SI base or coherent derived units (pascal, metre, second,
    kilogram, metre per second, etc.) over any non-SI alternative.
  - If a metric name or its surrounding code indicates it is a relative,
    normalized, or ratio quantity (e.g. contains "rel", "relative",
    "normalized", "ratio"), treat it as dimensionless: use unit:UNITLESS and
    quantityKind http://qudt.org/vocab/quantitykind/DimensionlessRatio.
  - If a metric is an absolute error/norm of a dimensional field quantity
    (e.g. "l2_error_pressure_abs"), give it that field's own SI unit (e.g.
    pascal for pressure errors), not a unitless placeholder.
  - Use the benchmark description and surrounding main.cc/problem.hh code as
    context for what physical quantity the metric represents.

A benchmark description and the full main.cc/problem.hh source are provided
only as background context. They will typically mention many more metrics,
functions, and variables than the ones you were asked about.

STRICT OUTPUT RULE: the "metrics" section below lists exactly the metric
keys you must return metadata for -- one JSON object per key, nothing added
and nothing omitted. Do not include metrics you merely noticed in main.cc,
problem.hh, or the benchmark description; only the ones explicitly listed
under "metrics" below.

Respond with a raw JSON array only: the first character of your response must
be '[' and the last character must be ']'. Do not wrap the JSON in markdown
code fences (no ``` of any kind). Do not include any prose, explanation, or
preamble before or after the JSON.
"""

METRIC_PROMPT_TEMPLATE = """\
The benchmark writes these solution metrics to its results/summary JSON
file. Each entry below gives the metric's key plus a short snippet of the
surrounding main.cc code for context.

=====================
benchmark description
=====================

{benchmark_description}

=====================
metrics -- infer metadata for EXACTLY these {n_items} metric(s), no others
=====================

{metric_json}

=====================
main.cc (background context only -- do not infer metadata for anything here
that isn't also listed under metrics above)
=====================

{main_cc}

=====================
problem.hh (background context only -- do not infer metadata for anything
here that isn't also listed under metrics above)
=====================

{problem_hh}

Infer semantic metadata (with SI units) for every metric listed under
metrics above -- exactly {n_items} item(s), no more, no less.

Return JSON like:

[
  {{
    "key": "l2_error_pressure_abs",
    "semantic_name": "pressure_l2_error",
    "datatype": "schema:Double",
    "unit": "unit:PA",
    "quantityKind": "[qudt.org](http://qudt.org/vocab/quantitykind/Pressure)",
    "confidence": 0.95,
    "explanation": "..."
  }}
]

Return ONLY the raw JSON array. No markdown code fences, no explanation, no
text before the opening '[' or after the closing ']'.
"""


def build_metric_prompt(
    candidates: list[MetricCandidate],
    main_cc: str,
    problem_hh: str,
    benchmark_description: str = "",
) -> str:
    return METRIC_PROMPT_TEMPLATE.format(
        benchmark_description=benchmark_description or "(none found)",
        n_items=len(candidates),
        metric_json=json.dumps([asdict(c) for c in candidates], indent=2),
        main_cc=main_cc,
        problem_hh=problem_hh,
    )


