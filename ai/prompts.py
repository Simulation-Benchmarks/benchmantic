# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

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

Some parameter entries include a "cpp_hint" field and a "code_context"
field: cpp_hint is scraped directly from the source code's
getParam<Type>("Section.Key") call site (e.g. the exact C++ variable name
and type it is assigned to, such as `density_` of type `Scalar`);
code_context is a short excerpt of the actual surrounding source code at
that same call site. Treat these as strong, code-grounded evidence of the
parameter's physical meaning and prefer them over guessing from the INI key
name alone -- e.g. a variable named `density_`/`rho_` implies kg/m3, a
variable read as `omega_`/an angular velocity implies rad/s, `viscosity_`
(dynamic) implies Pa*s, `radius_`/`length_` implies m. A parameter with
neither field means no getParam<>() call site was found for it -- infer
from the key name, value, and benchmark description instead.

You may also be shown "known corrections from prior human review": cases
where a human reviewer corrected an earlier AI guess for a similarly named
or similarly typed item, possibly in a different benchmark module. Treat
these as strong evidence for how a human expert wants this KIND of item
handled -- but still infer each requested item independently and precisely
from its own code context; do not copy a correction's exact values onto a
different item just because the name is similar.

A benchmark description (a doc-comment from the source, if available) is
provided as background context to help you understand what physical
scenario or published benchmark this is. Do NOT infer metadata for
anything other than the exact parameters listed under "params.input"
below -- you are deliberately NOT shown the rest of the source file, only
each parameter's own code_context, so there is nothing else to notice.

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
=====================
benchmark description
=====================

{benchmark_description}

=====================
params.input -- infer metadata for EXACTLY these {n_items} parameter(s), no others.
Each entry's own "cpp_hint"/"code_context" (see the system prompt) is your
primary code-grounded evidence.
=====================

{parameter_json}

=====================
known corrections from prior human review (guidance only -- weigh this
evidence for similarly named/typed items, but still infer each item above
independently from its own code context)
=====================

{known_corrections}
{fallback_context}
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


#: Combined character budget for the fallback-search excerpts below --
#: deliberately small. This is a last-resort safety net for the rare
#: candidate with no per-item cpp_hint/code_context/context of its own, NOT
#: a replacement for the old full-file dump (that's exactly what regularly
#: blew past provider per-request token/payload limits -- see ai.inference's
#: 404/413 handling). Most candidates never need this at all.
FALLBACK_CONTEXT_BUDGET = 1200
#: Chars grabbed on each side of a fallback string match.
FALLBACK_CONTEXT_WINDOW = 150


def _search_snippet(needle: str, *source_texts: str, window: int = FALLBACK_CONTEXT_WINDOW) -> str:
    """First-match, case-insensitive search for `needle` across
    `source_texts`, returning a small window of surrounding text -- the
    same idea as metadata.metrics.discover_metrics()'s per-candidate
    context, just applied on demand for whichever items didn't already get
    one. Empty string if `needle` isn't found anywhere.
    """
    for text in source_texts:
        if not text:
            continue
        idx = text.lower().find(needle.lower())
        if idx == -1:
            continue
        start = max(0, idx - window // 2)
        end = min(len(text), idx + len(needle) + window // 2)
        return text[start:end].strip()
    return ""


def _fallback_context_section(
    missing: list[tuple[str, str]],
    *source_texts: str,
    budget: int = FALLBACK_CONTEXT_BUDGET,
) -> str:
    """Build the optional "additional source excerpts" prompt section for
    whichever `missing` items (a list of (label, search_key) pairs) had no
    per-item code context of their own. Returns "" (nothing added to the
    prompt at all) when every item already has its own context, which is
    the common case -- so most prompts never carry this section.
    """
    if not missing:
        return ""
    lines = []
    used = 0
    for label, needle in missing:
        if used >= budget:
            break
        snippet = _search_snippet(needle, *source_texts)
        if snippet:
            lines.append(f"- {label}: {snippet}")
            used += len(snippet)
    if not lines:
        return ""
    return (
        "\n=====================\n"
        "additional source excerpts (best-effort search context for the few items above "
        "with no code-grounded hint of their own -- NOT the full source file)\n"
        "=====================\n\n"
        + "\n".join(lines) + "\n"
    )


def _format_known_corrections(known_corrections: list[dict] | None) -> str:
    """Render ai.corrections entries (see ai.corrections.relevant_corrections_for)
    as short human-readable lines for the prompt. Returns a placeholder
    string when there's nothing relevant yet, so the prompt template always
    has something to show for this section.
    """
    if not known_corrections:
        return "(none available yet)"
    lines = []
    for c in known_corrections:
        loc = f'{c["section"]}.{c["key"]}' if c.get("section") else c.get("key", "?")
        lines.append(
            f'- {loc}: AI originally suggested {json.dumps(c.get("ai_suggested", {}))}, '
            f'a human reviewer corrected it to {json.dumps(c.get("user_corrected", {}))}.'
        )
    return "\n".join(lines)


def build_prompt(
    candidates: list[ParameterCandidate],
    main_cc: str,
    problem_hh: str,
    benchmark_description: str = "",
    known_corrections: list[dict] | None = None,
) -> str:
    """`main_cc`/`problem_hh` are used ONLY as a fallback-search source for
    whichever `candidates` have no cpp_hint/code_context of their own (see
    metadata.parameters.attach_cpp_hints) -- the full text is deliberately
    NOT embedded in the prompt itself; that's what used to dominate prompt
    size (see ai.inference's 404/413 handling for what that caused).
    """
    missing = [(f"{c.section}.{c.key}", c.key) for c in candidates if not c.cpp_hint]
    return PROMPT_TEMPLATE.format(
        benchmark_description=benchmark_description or "(none found)",
        n_items=len(candidates),
        parameter_json=json.dumps([asdict(c) for c in candidates], indent=2),
        known_corrections=_format_known_corrections(known_corrections),
        fallback_context=_fallback_context_section(missing, main_cc, problem_hh),
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
  - Use the benchmark description and each metric's own "context" field
    (a snippet of the main.cc code around where it's computed/written) to
    judge what physical quantity the metric represents.

A benchmark description is provided as background context. You are
deliberately NOT shown the rest of main.cc/problem.hh -- only each metric's
own "context" snippet -- so there is nothing else to notice or infer
metadata for.

STRICT OUTPUT RULE: the "metrics" section below lists exactly the metric
keys you must return metadata for -- one JSON object per key, nothing added
and nothing omitted. Do not include metrics you merely noticed in a
context snippet or the benchmark description; only the ones explicitly
listed under "metrics" below.

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
known corrections from prior human review (guidance only -- weigh this
evidence for similarly named/typed items, but still infer each item above
independently from its own code context)
=====================

{known_corrections}
{fallback_context}
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


def _format_known_metric_corrections(known_corrections: list[dict] | None) -> str:
    if not known_corrections:
        return "(none available yet)"
    lines = []
    for c in known_corrections:
        lines.append(
            f'- {c.get("key", "?")}: AI originally suggested {json.dumps(c.get("ai_suggested", {}))}, '
            f'a human reviewer corrected it to {json.dumps(c.get("user_corrected", {}))}.'
        )
    return "\n".join(lines)


def build_metric_prompt(
    candidates: list[MetricCandidate],
    main_cc: str,
    problem_hh: str,
    benchmark_description: str = "",
    known_corrections: list[dict] | None = None,
) -> str:
    """`main_cc`/`problem_hh` are used ONLY as a fallback-search source for
    whichever `candidates` somehow have no `context` of their own (in
    practice this is essentially never -- metadata.metrics.discover_metrics()
    always fills `context` for every candidate it creates) -- see
    build_prompt()'s docstring for why the full text isn't embedded outright.
    """
    missing = [(c.key, c.key) for c in candidates if not c.context]
    return METRIC_PROMPT_TEMPLATE.format(
        benchmark_description=benchmark_description or "(none found)",
        n_items=len(candidates),
        metric_json=json.dumps([asdict(c) for c in candidates], indent=2),
        known_corrections=_format_known_metric_corrections(known_corrections),
        fallback_context=_fallback_context_section(missing, main_cc, problem_hh),
    )


