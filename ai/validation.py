# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
ai.validation

Validates and repairs LLM-returned parameter/metric JSON: required-field
checks, requested-vs-returned reconciliation, index bounds-checking, and a
safety net that catches (and fixes) the LLM confusing a QUDT unit with a
QUDT quantityKind.
"""

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metadata.metrics import MetricCandidate
    from metadata.parameters import ParameterCandidate

REQUIRED_FIELDS = ("semantic_name", "ini", "datatype", "unit")

#: Sensible default QUDT unit for a given quantityKind's local name, used as
#: a safety net when the LLM puts a quantitykind URI in the "unit" field
#: instead of an actual unit (see _fix_unit_quantitykind_confusion() below).
#: Not exhaustive -- just the quantity kinds that have come up in practice.
QUANTITYKIND_DEFAULT_UNIT: dict[str, str] = {
    "Length": "unit:M",
    "Area": "unit:M2",
    "Volume": "unit:M3",
    "Angle": "unit:RAD",
    "SolidAngle": "unit:SR",
    "AngularVelocity": "unit:RAD-PER-SEC",
    "AngularAcceleration": "unit:RAD-PER-SEC2",
    "Velocity": "unit:M-PER-SEC",
    "Acceleration": "unit:M-PER-SEC2",
    "Mass": "unit:KiloGM",
    "MassDensity": "unit:KiloGM-PER-M3",
    "Density": "unit:KiloGM-PER-M3",
    "Time": "unit:SEC",
    "Frequency": "unit:HZ",
    "Force": "unit:N",
    "Pressure": "unit:PA",
    "DynamicViscosity": "unit:PA-SEC",
    "KinematicViscosity": "unit:M2-PER-SEC",
    "Energy": "unit:J",
    "Power": "unit:W",
    "Temperature": "unit:K",
    "ThermodynamicTemperature": "unit:K",
    "ElectricCurrent": "unit:A",
    "Count": "unit:UNITLESS",
    "DimensionlessRatio": "unit:UNITLESS",
}

#: Matches a QUDT quantitykind URI/CURIE, e.g.
#: "http://qudt.org/vocab/quantitykind/Length" or "quantitykind:Length".
QUANTITYKIND_URI_PATTERN = re.compile(r"quantitykind[/:]([A-Za-z0-9_]+)", re.IGNORECASE)


def _fix_unit_quantitykind_confusion(item: dict, label: str = "") -> None:
    """Safety net for a recurring LLM mistake: writing a quantityKind URI
    (e.g. 'http://qudt.org/vocab/quantitykind/Length') into the "unit"
    field instead of an actual QUDT unit (e.g. 'unit:M'). Mutates `item` in
    place. If the quantityKind's local name is in QUANTITYKIND_DEFAULT_UNIT,
    swaps in that default unit (and backfills "quantityKind" from the same
    URI if it was left null); otherwise leaves "unit" as-is but downgrades
    confidence and appends a note to "explanation" so the mix-up is visible
    downstream rather than silently wrong.
    """
    unit_value = item.get("unit")
    if not isinstance(unit_value, str):
        return
    m = QUANTITYKIND_URI_PATTERN.search(unit_value)
    if not m:
        return  # "unit" already looks like a unit, nothing to fix

    quantity_name = m.group(1)
    default_unit = QUANTITYKIND_DEFAULT_UNIT.get(quantity_name)
    item.setdefault("quantityKind", None)
    if item["quantityKind"] is None:
        item["quantityKind"] = f"http://qudt.org/vocab/quantitykind/{quantity_name}"

    if default_unit:
        print(
            f"warning: {label + ' ' if label else ''}'{item.get('semantic_name', item.get('key', '?'))}' "
            f"had a quantityKind URI in its 'unit' field -- auto-corrected to {default_unit}.",
            file=sys.stderr,
        )
        item["unit"] = default_unit
    else:
        print(
            f"warning: {label + ' ' if label else ''}'{item.get('semantic_name', item.get('key', '?'))}' "
            f"has a quantityKind URI in its 'unit' field and no default unit mapping exists for "
            f"'{quantity_name}' -- left as-is, please review manually.",
            file=sys.stderr,
        )
        item["confidence"] = min(item.get("confidence", 1.0), 0.3)
        note = "unit/quantityKind mix-up detected by validator; needs manual review."
        item["explanation"] = f"{item.get('explanation', '')} [{note}]".strip()


def extract_json_array(raw: str) -> str:
    """Recover a JSON array from an LLM response that may be wrapped in
    markdown code fences or preceded/followed by stray prose.

    Some models (notably smaller/open-weight ones on Groq) ignore
    "no markdown fences" instructions and return things like:
        ```json
        [ ... ]
        ```
    or add a sentence before/after the array. This strips that wrapping
    so json.loads() gets clean input.
    """
    text = raw.strip()

    # Strip ```json ... ``` or ``` ... ``` fences.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    if text.startswith("[") and text.endswith("]"):
        return text

    # Fall back to slicing out the first top-level array in the text,
    # in case the model added a preamble or trailing remark.
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    return text


def validate_metadata(
    data: list[dict],
    candidates: list[ParameterCandidate] | None = None,
) -> list[dict]:
    """Ensure the returned JSON has the expected structure. If `candidates`
    is given, also enforce that the response contains *exactly* one item per
    requested candidate -- no more, no less. Models (especially smaller ones
    on Groq) sometimes ignore the requested subset and hallucinate extra
    entries for parameters they merely noticed elsewhere in the source code
    context; silently keeping those poisons the cache with metadata for
    parameters that may not even exist in every case's params.input, which
    later crashes resolve_case_params(). So any unrequested item is dropped
    (with a warning) rather than kept, and any candidate the model skipped
    raises an error so the caller retries instead of silently under-covering.
    """
    if not isinstance(data, list):
        raise ValueError("Model returned something other than a JSON list.")

    validated = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each item must be a JSON object.")

        for field in REQUIRED_FIELDS:
            if field not in item:
                raise ValueError(f"Missing required field '{field}'.")

        if not isinstance(item["ini"], list) or len(item["ini"]) != 2:
            raise ValueError("'ini' must be a list of [section, key].")

        item.setdefault("index", 0)
        item.setdefault("confidence", 1.0)
        item.setdefault("quantityKind", None)
        item.setdefault("explanation", "")
        _fix_unit_quantitykind_confusion(item, label="parameter")
        validated.append(item)

    if candidates is not None:
        expected = {(c.section.lower(), c.key.lower()) for c in candidates}
        candidate_by_key = {(c.section.lower(), c.key.lower()): c for c in candidates}
        seen: set[tuple[str, str]] = set()
        filtered = []
        for item in validated:
            ini_pair = (item["ini"][0].lower(), item["ini"][1].lower())
            if ini_pair not in expected:
                print(
                    f"warning: dropping unrequested item '{item['semantic_name']}' "
                    f"({item['ini']}) -- not among the requested parameters",
                    file=sys.stderr,
                )
                continue
            if ini_pair in seen:
                print(f"warning: dropping duplicate item for {item['ini']}", file=sys.stderr)
                continue
            seen.add(ini_pair)

            # Bounds-check "index" against the token count of the *template*
            # params.input the model was actually shown -- catches a bad
            # index right here instead of letting it silently reach the
            # cache and later crash resolve_case_params() with a bare
            # IndexError on some downstream case's params.input.
            candidate = candidate_by_key[ini_pair]
            n_tokens = len(candidate.tokens)
            if not (0 <= item["index"] < n_tokens):
                print(
                    f"warning: '{item['semantic_name']}' ({item['ini']}) got index "
                    f"{item['index']} from the model, but the template value "
                    f"'{candidate.value}' only has {n_tokens} token(s) -- clamping to "
                    f"index 0. Please double check this parameter if it's meant to "
                    f"pick a specific element out of a multi-value field.",
                    file=sys.stderr,
                )
                item["index"] = 0
                item["confidence"] = min(item.get("confidence", 1.0), 0.5)

            filtered.append(item)

        missing = expected - seen
        if missing:
            missing_list = ", ".join(f"[{s}] {k}" for s, k in sorted(missing))
            raise ValueError(f"Model did not return metadata for: {missing_list}")

        validated = filtered

    return validated


REQUIRED_METRIC_FIELDS = ("key", "semantic_name", "datatype", "unit")


def validate_metric_metadata(
    data: list[dict],
    candidates: list[MetricCandidate] | None = None,
) -> list[dict]:
    """Ensure the returned metric JSON has the expected structure, and (if
    `candidates` is given) that it covers exactly the requested metric keys
    -- same rationale as validate_metadata() above.
    """
    if not isinstance(data, list):
        raise ValueError("Model returned something other than a JSON list.")

    validated = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each item must be a JSON object.")

        for field in REQUIRED_METRIC_FIELDS:
            if field not in item:
                raise ValueError(f"Missing required field '{field}'.")

        item.setdefault("confidence", 1.0)
        item.setdefault("quantityKind", None)
        item.setdefault("explanation", "")
        _fix_unit_quantitykind_confusion(item, label="metric")
        validated.append(item)

    if candidates is not None:
        expected = {c.key for c in candidates}
        seen: set[str] = set()
        filtered = []
        for item in validated:
            if item["key"] not in expected:
                print(
                    f"warning: dropping unrequested metric item '{item['key']}' "
                    f"-- not among the requested metrics",
                    file=sys.stderr,
                )
                continue
            if item["key"] in seen:
                print(f"warning: dropping duplicate item for metric '{item['key']}'", file=sys.stderr)
                continue
            seen.add(item["key"])
            filtered.append(item)

        missing = expected - seen
        if missing:
            raise ValueError(f"Model did not return metadata for metrics: {', '.join(sorted(missing))}")

        validated = filtered

    return validated


