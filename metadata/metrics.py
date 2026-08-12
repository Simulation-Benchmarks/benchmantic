# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
metadata.metrics

Output/solution metric discovery and extraction: scanning main.cc for JSON
keys the simulation writes to its results file, and building the final
RO-Crate field dicts from LLM-inferred (or cached) metadata.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class MetricCandidate:
    """A solution/output metric discovered in main.cc (e.g. a JSON key that
    the simulation writes to its results/summary file)."""
    key: str
    context: str = ""


METRIC_KEY_PATTERN = re.compile(r'\\"([A-Za-z0-9_]+)\\"\s*:')


def discover_metrics(main_cc: str, context_chars: int = 100) -> list[MetricCandidate]:
    """Find output metric keys in main.cc and grab a small snippet of
    surrounding code around each occurrence, so the LLM has some context
    (e.g. what's being computed/printed) to infer a sensible SI unit from.
    """
    candidates: list[MetricCandidate] = []
    seen: set[str] = set()
    for m in METRIC_KEY_PATTERN.finditer(main_cc):
        key = m.group(1)
        if key in seen:
            continue
        seen.add(key)
        start = max(0, m.start() - context_chars)
        end = min(len(main_cc), m.end() + context_chars)
        candidates.append(MetricCandidate(key=key, context=main_cc[start:end].strip()))
    return candidates



def discover_metrics_from_maincc(main_cc_path: Path):
    """Wrapper around ai_parameter_inference.discover_metrics() that also
    exits with a helpful error if no metric keys are found, and reads the
    file for the caller.
    """
    text = main_cc_path.read_text(encoding="utf-8")
    candidates = discover_metrics(text)
    if not candidates:
        sys.exit(
            f"No JSON keys found in {main_cc_path} -- expected lines like "
            r'out << "  \"some_key\": " << value;'
        )
    return candidates



#: Optional manual overrides. Metric units/quantityKinds are inferred by the
#: LLM by default (see infer_metric_metadata / build_metric_fields), but any
#: key listed here takes precedence over the LLM's answer -- useful for
#: pinning a metric down without depending on/re-querying the model.
KNOWN_METRIC_UNITS: dict[str, dict[str, Any]] = {}
DEFAULT_METRIC_UNIT: dict[str, Any] = {"unit": "unit:UNITLESS", "quantityKind": None}


def build_metric_fields(metadata: list[dict]) -> dict:
    """Keyed by the raw metric key (as it appears in main.cc / the summary
    JSON file), since that's what generate_metadata.py's metric_keys list
    (and its downstream extract nodes) already use for lookups.
    """
    return {
        item["key"]: {
            "semantic_name": item["semantic_name"],
            "unit": item["unit"],
            "quantityKind": item.get("quantityKind"),
            "datatype": item["datatype"],
            "description": item.get("explanation", ""),
        }
        for item in metadata
    }
