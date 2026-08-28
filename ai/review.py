#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
ai.review

Interactive CLI review/edit step for AI-inferred parameter and metric
metadata, run inside metadata.builder.build() between "Infer" and "Build":

    discover -> infer -> REVIEW (this module) -> build -> validate

Prints the current parameter/metric table, lets the reviewer edit any field
of any row -- looping until they say the result is final -- and returns both
the (possibly edited) metadata and a list of correction records
((ai_suggested -> user_corrected) pairs) for whatever was actually changed,
so ai.corrections can remember it for future runs on other benchmarks.

Not run at all when --skip-review is passed (required for non-interactive/CI
use -- same rationale as describe_benchmark.py's --scenario-params prompt:
this needs a real terminal for input()).
"""

from __future__ import annotations

import copy
from typing import Any

PARAM_FIELDS = ("semantic_name", "datatype", "unit", "quantityKind", "explanation")
METRIC_FIELDS = ("semantic_name", "datatype", "unit", "quantityKind", "explanation")

#: Fields a correction record can capture -- deliberately excludes
#: "explanation" (free text, not a semantic fact worth matching on later).
DIFFABLE_FIELDS = ("semantic_name", "datatype", "unit", "quantityKind")

ALLOWED_DATATYPES = ("schema:Integer", "schema:Float", "schema:Double", "schema:String")

#: Below this confidence, a row is flagged ('!') in the table so it's easy
#: to spot before pressing Enter -- it doesn't block accepting on its own;
#: Enter always accepts the table as shown, flagged rows included. Override
#: via --review-confidence-threshold.
DEFAULT_CONFIDENCE_THRESHOLD = 0.7


def _label(item: dict, kind: str) -> str:
    if kind == "parameter":
        section, key = item["ini"]
        return f"{section}.{key}"
    return item["key"]


def _truncate(text: Any, width: int) -> str:
    s = "" if text is None else str(text)
    s = s.replace("\n", " ")
    return s if len(s) <= width else s[: width - 1] + "…"


def _confidence(item: dict) -> float:
    try:
        return float(item.get("confidence", 1.0))
    except (TypeError, ValueError):
        return 1.0


def _flagged_indices(items: list[dict], threshold: float) -> list[int]:
    """Rows that still need a look before a plain accept, and haven't been
    touched this session yet (an edit always resets confidence to 1.0 and
    clears "_needs_verification" -- see _edit_row()). Two independent
    reasons flag a row:
      - low confidence: the model's own reported uncertainty about the
        physical inference (unit, quantity kind, semantic name).
      - "_needs_verification": a structural repair (e.g. ai.validation
        backfilling a missing 'ini'/'key' field positionally) that has
        nothing to do with the model's confidence in the physics, but still
        needs a human to confirm the row-to-parameter mapping is correct.
    These are deliberately kept distinct (see _print_table()'s '!' vs '?')
    rather than folded into one number, so a low "confidence" always means
    "the AI itself wasn't sure" and nothing else.
    """
    return [
        i for i, item in enumerate(items)
        if not item.get("_edited") and (_confidence(item) < threshold or item.get("_needs_verification"))
    ]


def _print_table(items: list[dict], kind: str, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> None:
    label_header = "Parameter" if kind == "parameter" else "Metric"
    w = {"idx": 3, "label": 22, "name": 22, "dtype": 14, "unit": 16, "qkind": 20, "conf": 5}
    header = (
        f"{'#':>{w['idx']}}  {label_header:<{w['label']}}  "
        f"{'semantic_name':<{w['name']}}  {'datatype':<{w['dtype']}}  "
        f"{'unit':<{w['unit']}}  {'quantityKind':<{w['qkind']}}  {'conf':>{w['conf']}}"
    )
    print("\n" + header)
    print("-" * len(header))
    for i, item in enumerate(items):
        qk = item.get("quantityKind") or ""
        qk_short = qk.rsplit("/", 1)[-1] if qk else ""
        if item.get("_edited"):
            row_flag = "*"
        elif item.get("_needs_verification"):
            row_flag = "?"
        elif _confidence(item) < threshold:
            row_flag = "!"
        else:
            row_flag = " "
        print(
            f"{i:>{w['idx']}}{row_flag} {_truncate(_label(item, kind), w['label']):<{w['label']}}  "
            f"{_truncate(item.get('semantic_name'), w['name']):<{w['name']}}  "
            f"{_truncate(item.get('datatype'), w['dtype']):<{w['dtype']}}  "
            f"{_truncate(item.get('unit'), w['unit']):<{w['unit']}}  "
            f"{_truncate(qk_short, w['qkind']):<{w['qkind']}}  "
            f"{_confidence(item):>{w['conf']}.2f}"
        )
    notes = []
    if any(item.get("_edited") for item in items):
        notes.append("* = edited this session")
    if any(not item.get("_edited") and item.get("_needs_verification") for item in items):
        notes.append("? = mapping needs verification (not a confidence issue) -- review before accepting")
    if any(not item.get("_edited") and not item.get("_needs_verification") and _confidence(item) < threshold
           for item in items):
        notes.append(f"! = confidence below {threshold:.2f} -- review before accepting")
    if notes:
        print("(" + "; ".join(notes) + ")")


def _edit_row(item: dict, kind: str) -> None:
    fields = PARAM_FIELDS if kind == "parameter" else METRIC_FIELDS
    while True:
        print(f"\nEditing {_label(item, kind)}:")
        for i, f in enumerate(fields):
            print(f"  {i}) {f:<14} = {item.get(f)!r}")
        choice = input("Field number to edit, or Enter to go back to the table: ").strip()
        if choice == "":
            return
        if not choice.isdigit() or not (0 <= int(choice) < len(fields)):
            print("Not a valid field number.")
            continue
        field = fields[int(choice)]
        new_value = input(f"New value for {field} (current: {item.get(field)!r}): ").strip()
        if new_value == "":
            print("No change (empty input).")
            continue

        if field == "datatype" and new_value not in ALLOWED_DATATYPES:
            print(f"warning: {new_value!r} isn't one of {ALLOWED_DATATYPES} -- saved as-is, please double check.")
        if field == "quantityKind" and not new_value.startswith("http"):
            expanded = f"http://qudt.org/vocab/quantitykind/{new_value}"
            print(f"  (expanded {new_value!r} -> {expanded!r})")
            new_value = expanded
        if field == "unit" and not new_value.startswith("unit:"):
            print(f"warning: {new_value!r} doesn't look like a QUDT unit CURIE (expected 'unit:XXX') -- saved as-is.")

        item[field] = new_value
        item["confidence"] = 1.0
        _append_note(item, "manually corrected by reviewer")
        item["_edited"] = True
        item.pop("_needs_verification", None)
        print(f"  {field} -> {new_value!r}")


def _append_note(item: dict, note: str) -> None:
    if note not in (item.get("explanation") or ""):
        item["explanation"] = f"{item.get('explanation', '')} [{note}]".strip()


def _diff_corrections(originals: list[dict], finals: list[dict], kind: str) -> list[dict]:
    corrections = []
    for before, after in zip(originals, finals):
        if not after.get("_edited"):
            continue
        ai_suggested, user_corrected = {}, {}
        for field in DIFFABLE_FIELDS:
            if before.get(field) != after.get(field):
                ai_suggested[field] = before.get(field)
                user_corrected[field] = after.get(field)
        if not user_corrected:
            continue
        entry: dict[str, Any] = {
            "type": kind,
            "key": after["key"] if kind == "metric" else after["ini"][1],
            "ai_suggested": ai_suggested,
            "user_corrected": user_corrected,
        }
        if kind == "parameter":
            entry["section"] = after["ini"][0]
        corrections.append(entry)
    return corrections


def interactive_review(
    items: list[dict],
    kind: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> tuple[list[dict], list[dict]]:
    """Run the CLI row-editor loop over `items` (parameter or metric
    metadata dicts, as produced by ai.validation / loaded from cache).

    Simple by design: one prompt. Rows below `confidence_threshold` (or
    flagged with "_needs_verification", e.g. a backfilled 'ini'/'key' --
    see ai.validation) show a '!'/'?' marker in the table so they're easy to
    spot, but pressing Enter always accepts the table as shown, flagged rows
    included -- there's no separate "type 'accept' to confirm" gate to get
    through, and no separate "reprint" command either: the table is simply
    reprinted every time you edit a row and loop back around. If you want
    to fix something, type its row number first; Enter only when you're
    done looking.

    Returns (final_items, correction_records): final_items is the same list
    object, mutated in place (any transient "_edited"/"_needs_verification"
    markers are stripped before returning); correction_records is one entry
    per field the reviewer actually changed, ready for
    ai.corrections.append_corrections().
    """
    if not items:
        return items, []

    originals = copy.deepcopy(items)
    label_kind = "parameter" if kind == "parameter" else "metric"
    print(f"\n=== Review {label_kind}s ({len(items)}) ===")
    print("Values below were inferred automatically -- check units, quantity kinds, and names before continuing.")

    while True:
        _print_table(items, kind, confidence_threshold)
        flagged = _flagged_indices(items, confidence_threshold)
        if flagged:
            labels = ", ".join(_label(items[i], kind) for i in flagged)
            print(f"\n{len(flagged)} row(s) flagged above -- check them before pressing Enter: {labels}")

        try:
            choice = input(
                "\nEnter a row number to edit, or press Enter to accept and continue: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = ""

        if choice != "":
            if not choice.isdigit() or not (0 <= int(choice) < len(items)):
                print("Not a valid row number.")
                continue
            _edit_row(items[int(choice)], kind)
            continue

        # Enter -- accept the table as shown, including any still-flagged
        # rows (each gets a note recording that it went through unedited,
        # same as before, just without a separate confirmation step).
        for i in flagged:
            note = (
                "mapping accepted as-is without correction" if items[i].get("_needs_verification")
                else "accepted at low confidence without correction"
            )
            _append_note(items[i], note)
            items[i].pop("_needs_verification", None)
        break

    corrections = _diff_corrections(originals, items, kind)
    for item in items:
        item.pop("_edited", None)
        # Defensive only -- every code path that can break out of the loop
        # above already clears this on every remaining item (either via
        # _edit_row() or the "accept" branch), so this should always be a
        # no-op by the time we get here.
        item.pop("_needs_verification", None)
    return items, corrections


def review_or_skip(
    items: list[dict],
    kind: str,
    skip: bool,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> tuple[list[dict], list[dict]]:
    """interactive_review(), unless `skip` is set (--skip-review) or there's
    nothing to review -- in which case the items are returned unchanged and
    no corrections are recorded.

    Note: with --skip-review, any "_needs_verification" marker (see
    ai.validation's positional-backfill repair) is left in place on the
    returned items rather than silently dropped -- there's no review step
    here to resolve it, so it stays visible in the cached
    .parameter_metadata_cache.json / .metric_metadata_cache.json entry as a
    signal that this row's parameter mapping was reconstructed, not
    confirmed, and is worth a manual look later. It's stripped from the
    metadata4ing graph either way (build_parameter_fields()/
    build_metric_fields() only ever copy the specific fields the graph
    needs).
    """
    if skip or not items:
        return items, []
    return interactive_review(items, kind, confidence_threshold)
