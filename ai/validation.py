# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
ai.validation

Validates and repairs LLM-returned parameter/metric JSON: required-field
checks, requested-vs-returned reconciliation, index bounds-checking, a
safety net that catches (and fixes) the LLM confusing a QUDT unit with a
QUDT quantityKind, a fallback that backfills a missing quantityKind from
its unit alone (querying the unit's own QUDT vocabulary entry first, then
a best-effort live lookup against the SI Digital Framework's units page
-- see _backfill_missing_quantitykind()), and stripping of a recurring
hedging clause (e.g. "..., but the exact meaning is unclear without more
context.") from "explanation" before it's cached or copied into
benchmark.jsonld's "description" fields.
"""

from __future__ import annotations

import re
import sys
import urllib.error
import urllib.request
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


# =============================================================================
# quantityKind backfill from "unit" alone (QUDT vocabulary + SI Digital
# Framework reference)
# =============================================================================
#
# Sometimes the model returns a perfectly good "unit" but leaves
# "quantityKind" null. Two live fallback layers try to fill it in from the
# unit alone, in order -- see _backfill_missing_quantitykind() below:
#   1. _fetch_quantitykind_from_qudt(), which asks the unit's own QUDT
#      vocabulary entry (e.g. http://qudt.org/vocab/unit/PA) for its
#      qudt:hasQuantityKind relationship -- QUDT is the same vocabulary
#      this pipeline's units/quantityKind URIs already come from (see
#      QUANTITYKIND_URI_PATTERN), so this is the authoritative source,
#      not a guess, and it covers every unit QUDT knows about rather than
#      just a hand-picked subset.
#   2. _fetch_quantitykind_from_si_framework(), a best-effort lookup
#      against SI_UNITS_REFERENCE_URL, only attempted if the QUDT lookup
#      didn't turn up an answer (unit not found, network hiccup, etc.).
# Both are genuinely best-effort: this environment's own network access to
# either qudt.org or si-digital-framework.org returned no connection at
# all when this was written (and a real browser fetch of the SI page got a
# 403 even with a UA header set), so both functions are written
# defensively -- ANY failure (no network, timeout, a 403/blocked response,
# unexpected response structure, no match found) returns None silently
# rather than raising. Worst case is exactly today's behavior
# (quantityKind stays null); this only ever adds a chance to fill it in,
# never a new way for a run to fail. Verify their output before trusting
# it blindly -- neither was tested against the live services' actual
# responses.

QUDT_UNIT_BASE_URI = "http://qudt.org/vocab/unit/"
SI_UNITS_REFERENCE_URL = "https://si-digital-framework.org/SI/units?lang=en"

#: How long to wait for the QUDT vocabulary server before giving up --
#: kept short since this is a best-effort fallback for one missing field,
#: not something the pipeline should hang on.
_QUDT_TIMEOUT_SECONDS = 5.0


def _fetch_quantitykind_from_qudt(unit_value: str) -> str | None:
    """Best-effort live lookup: ask QUDT's own vocabulary entry for
    `unit_value` (e.g. "unit:PA" -> http://qudt.org/vocab/unit/PA) what
    quantity kind it belongs to, via content negotiation. QUDT unit
    resources publish their qudt:hasQuantityKind relationship as RDF, so
    a GET on the unit's URI (asking for a machine-readable format) is
    enough -- no separate SPARQL query needed.

    Deliberately loose about the response body's exact structure (RDF/JSON,
    JSON-LD, or even Turtle/RDF-XML would all work here) rather than a
    strict parse: it just decodes the body as text and regex-searches it
    for a quantitykind URI/CURIE whose local name is one this pipeline
    already recognizes (QUANTITYKIND_DEFAULT_UNIT's keys) -- the same
    "found something plausible" approach as the SI Digital Framework
    fallback below, since this codebase can't verify the exact response
    format against a live connection. Every failure mode (network
    unreachable, blocked, timeout, no recognizable match) returns None
    rather than raising -- see this section's module-level comment.
    """
    unit_symbol = unit_value.removeprefix("unit:")
    if not unit_symbol:
        return None
    url = QUDT_UNIT_BASE_URI + unit_symbol
    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; benchmantic/1.0; +metadata inference fallback)",
                "Accept": "application/rdf+json, application/ld+json;q=0.9, text/turtle;q=0.8, */*;q=0.5",
            },
        )
        with urllib.request.urlopen(request, timeout=_QUDT_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None

    for match in QUANTITYKIND_URI_PATTERN.finditer(body):
        quantity_name = match.group(1)
        if quantity_name in QUANTITYKIND_DEFAULT_UNIT:
            return quantity_name
    return None


#: For the SI Digital Framework live-fetch fallback: quantityKind name -> the human-readable
#: phrase to look for near the unit's symbol on SI_UNITS_REFERENCE_URL's
#: page (e.g. the page lists "pascal" next to "pressure"). Only the
#: quantity kinds this pipeline recognizes elsewhere (QUANTITYKIND_DEFAULT_UNIT's
#: keys) are worth searching for -- returning something outside that set
#: wouldn't be usable by _fix_unit_quantitykind_confusion() or the rest of
#: this pipeline anyway.
_QUANTITYKIND_SEARCH_PHRASES: dict[str, str] = {
    "Length": "length",
    "Mass": "mass",
    "Time": "time",
    "ElectricCurrent": "electric current",
    "ThermodynamicTemperature": "thermodynamic temperature",
    "AmountOfSubstance": "amount of substance",
    "LuminousIntensity": "luminous intensity",
    "Area": "area",
    "Volume": "volume",
    "Angle": "plane angle",
    "SolidAngle": "solid angle",
    "AngularVelocity": "angular velocity",
    "AngularAcceleration": "angular acceleration",
    "Velocity": "velocity",
    "Acceleration": "acceleration",
    "MassDensity": "density",
    "Frequency": "frequency",
    "Force": "force",
    "Pressure": "pressure",
    "DynamicViscosity": "viscosity",
    "KinematicViscosity": "viscosity",
    "Energy": "energy",
    "Power": "power",
}

_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

#: How long to wait for SI_UNITS_REFERENCE_URL before giving up -- kept
#: short since this is a best-effort fallback for one missing field, not
#: something the pipeline should hang on.
_SI_FRAMEWORK_TIMEOUT_SECONDS = 5.0


def _fetch_quantitykind_from_si_framework(unit_value: str) -> str | None:
    """Best-effort live fallback: look for `unit_value`'s quantity on
    SI_UNITS_REFERENCE_URL when UNIT_TO_QUANTITYKIND above doesn't cover
    it. See this section's module-level comment for why this is written
    defensively -- every failure mode here (network unreachable, blocked/
    403, timeout, unexpected HTML, no match) returns None rather than
    raising, so a run can never fail because of this function.

    Loosely: strips HTML tags to plain text, finds every place the unit's
    symbol (e.g. "Pa" from "unit:PA") appears, and checks a window of text
    around each occurrence for a recognizable quantity-name phrase (see
    _QUANTITYKIND_SEARCH_PHRASES). Deliberately loose rather than a precise
    table-cell parse, since this page's exact markup isn't something this
    codebase controls -- "found something plausible nearby" is the goal,
    not exact extraction. Returns the first quantity kind whose phrase
    turns up near ANY occurrence of the symbol; ambiguous/no matches
    return None, same as not having looked at all.
    """
    unit_symbol = unit_value.removeprefix("unit:")
    if not unit_symbol:
        return None
    try:
        request = urllib.request.Request(
            SI_UNITS_REFERENCE_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; benchmantic/1.0; +metadata inference fallback)"},
        )
        with urllib.request.urlopen(request, timeout=_SI_FRAMEWORK_TIMEOUT_SECONDS) as response:
            html = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None

    text = _HTML_TAG_PATTERN.sub(" ", html)
    text = re.sub(r"\s+", " ", text).lower()
    symbol_lower = unit_symbol.lower()

    for match in re.finditer(re.escape(symbol_lower), text):
        window = text[max(0, match.start() - 150): match.end() + 150]
        for quantity_kind, phrase in _QUANTITYKIND_SEARCH_PHRASES.items():
            if phrase in window:
                return quantity_kind
    return None


def _lookup_quantitykind_for_unit(unit_value: str | None) -> str | None:
    """QUDT's own vocabulary entry for the unit first (authoritative,
    covers anything QUDT knows about), then a best-effort live
    SI_UNITS_REFERENCE_URL lookup if that didn't resolve it. Both layers
    need network access -- this backfill is skipped entirely (item just
    keeps its missing quantityKind) if neither is reachable. Returns None
    if neither layer finds anything.
    """
    if not unit_value:
        return None
    return _fetch_quantitykind_from_qudt(unit_value) or _fetch_quantitykind_from_si_framework(unit_value)


def _backfill_missing_quantitykind(item: dict, label: str = "") -> None:
    """If the model returned a "unit" but left "quantityKind" null, try to
    fill it in via _lookup_quantitykind_for_unit() (see above). Mutates
    `item` in place; no-op if quantityKind is already set, there's no unit
    to look up, or the item isn't a numerical quantity in the first place
    (a schema:String field's "unit:UNITLESS" -- e.g. a boolean flag or a
    name field -- has no physical quantity kind at all; only numeric
    datatypes are worth backfilling here).

    Marked "_needs_verification" rather than silently indistinguishable
    from the model's own answer -- this is OUR inference from the unit
    alone (via a reference lookup), not something the model itself
    assessed, same rationale as ai.validation's other structural backfills
    (see _backfill_missing_ini()'s docstring). Does NOT touch "confidence"
    for the same reason that backfill doesn't.
    """
    if item.get("quantityKind") or not item.get("unit"):
        return
    if item.get("datatype") == "schema:String":
        return
    quantity_name = _lookup_quantitykind_for_unit(item["unit"])
    if not quantity_name:
        return
    item["quantityKind"] = f"http://qudt.org/vocab/quantitykind/{quantity_name}"
    item["_needs_verification"] = (
        f"quantityKind was missing from the model's response -- filled in from unit "
        f"{item['unit']!r} ({quantity_name}) via a live lookup ({QUDT_UNIT_BASE_URI}{item['unit'].removeprefix('unit:')} "
        f"or, as a fallback, {SI_UNITS_REFERENCE_URL}); please verify."
    )
    print(
        f"warning: {label + ' ' if label else ''}'{item.get('semantic_name', item.get('key', '?'))}' "
        f"was missing quantityKind -- filled in as {quantity_name!r} from its unit ({item['unit']!r}), "
        f"flagged for review.",
        file=sys.stderr,
    )

#: Matches a recurring LLM hedging clause tacked onto an otherwise-useful
#: "explanation" -- e.g. "..., but the exact meaning is unclear without
#: more context." This is filler: it doesn't add any information the
#: confidence score doesn't already convey, and it reads badly once
#: "explanation" is copied verbatim into benchmark.jsonld as a parameter's/
#: metric's "description" (see metadata.parameters.build_parameter_fields()
#: and metadata.metrics.build_metric_fields()). Deliberately broad enough to
#: catch "but"/"though"/"although"/"however" + "exact"/"precise" meaning
#: is unclear + "more"/"further"/"additional" context/information, not just
#: the one literal wording -- but still anchored on "meaning is unclear" so
#: it can't eat an unrelated sentence that happens to contain "but".
_HEDGE_CLAUSE_PATTERN = re.compile(
    r"\s*[,;]?\s*\b(?:but|though|although|however)\b[^.!?]*?"
    r"\b(?:exact|precise)\s+meaning\s+is\s+unclear\b"
    r"[^.!?]*?\b(?:context|information)\b[^.!?]*[.!?]?",
    re.IGNORECASE,
)


def _strip_hedging(text: str | None) -> str:
    """Remove _HEDGE_CLAUSE_PATTERN matches from `text` and tidy up the
    leftover punctuation/whitespace (e.g. a dangling comma where the clause
    used to start, or a missing final period). Safe to call on empty/None
    input.
    """
    if not text:
        return text or ""
    cleaned = _HEDGE_CLAUSE_PATTERN.sub("", text)
    cleaned = re.sub(r"\s+([.,;!?])", r"\1", cleaned)  # "grid ," -> "grid,"
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


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


def _backfill_missing_ini(data: list, candidates: list[ParameterCandidate] | None) -> None:
    """Some models (seen with Groq's openai/gpt-oss-120b) drop the requested
    "ini" field from every item entirely, despite the prompt's explicit
    schema example -- while still returning the same count of items, in the
    same order, as the requested candidate list (the STRICT OUTPUT RULE
    the prompt asks for). Rather than failing the whole batch and burning a
    retry attempt on what's really just a formatting slip, backfill "ini"
    positionally from the candidates the model was actually asked about.

    Deliberately conservative: only fires when EVERY item is missing "ini"
    AND the item count matches the candidate count exactly -- anything else
    (partial omission, a different count) is too ambiguous to guess safely,
    so it's left alone for the normal required-field/count checks below to
    catch.

    This does NOT touch the model's own reported "confidence" -- that number
    is the model's actual assessment of the physical inference (unit,
    quantity kind, semantic name), which the backfill has no bearing on and
    shouldn't overwrite or hide. Instead, backfilled items are marked with
    "_needs_verification" -- a separate signal ai.review's review gate
    treats the same way as low confidence (it forces a look before a plain
    accept), without claiming the model itself was unsure. See ai.review's
    _flagged_indices().
    """
    if not candidates or len(data) != len(candidates):
        return
    if not all(isinstance(item, dict) and "ini" not in item for item in data):
        return
    for item, candidate in zip(data, candidates):
        item["ini"] = [candidate.section, candidate.key]
        item["_needs_verification"] = "'ini' backfilled positionally -- model omitted it; please verify this mapping."
    print(
        f"warning: model response omitted the 'ini' field for all {len(data)} item(s); "
        "backfilled positionally from the requested parameter order and flagged for review "
        "(the model's own confidence scores are kept as-is).",
        file=sys.stderr,
    )


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

    _backfill_missing_ini(data, candidates)

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
        item["explanation"] = _strip_hedging(item["explanation"])
        _fix_unit_quantitykind_confusion(item, label="parameter")
        _backfill_missing_quantitykind(item, label="parameter")
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


def _backfill_missing_key(data: list, candidates: list[MetricCandidate] | None) -> None:
    """Metric-side counterpart of _backfill_missing_ini() above -- same
    positional-backfill rationale, same conservative all-or-nothing/exact-
    count guard. Also leaves the model's own "confidence" untouched and uses
    "_needs_verification" instead -- see _backfill_missing_ini()'s docstring.
    """
    if not candidates or len(data) != len(candidates):
        return
    if not all(isinstance(item, dict) and "key" not in item for item in data):
        return
    for item, candidate in zip(data, candidates):
        item["key"] = candidate.key
        item["_needs_verification"] = "'key' backfilled positionally -- model omitted it; please verify this mapping."
    print(
        f"warning: model response omitted the 'key' field for all {len(data)} item(s); "
        "backfilled positionally from the requested metric order and flagged for review "
        "(the model's own confidence scores are kept as-is).",
        file=sys.stderr,
    )


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

    _backfill_missing_key(data, candidates)

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
        item["explanation"] = _strip_hedging(item["explanation"])
        _fix_unit_quantitykind_confusion(item, label="metric")
        _backfill_missing_quantitykind(item, label="metric")
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


