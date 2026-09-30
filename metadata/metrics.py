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


#: The metrics file every benchmark implementation writes (see the
#: Simulation-Benchmarks Snakefiles / run_benchmark.py).
METRICS_FILE_NAME = "solution_metrics.json"


def discover_metrics_in_python(text: str, metrics_file: str = METRICS_FILE_NAME,
                               context_chars: int = 100) -> list[MetricCandidate]:
    """Metric keys from a Python post-processing script: the string keys of
    the dict that is json.dump()ed in the same function (or module) that
    mentions `metrics_file`. Scoped this way so other dict literals in the
    script (e.g. a summary table) aren't mistaken for metrics.
    """
    import ast

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    def mentions_metrics_file(node) -> bool:
        return any(isinstance(n, ast.Constant) and isinstance(n.value, str) and metrics_file in n.value
                   for n in ast.walk(node))

    scopes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    scopes = [s for s in scopes if mentions_metrics_file(s)] or ([tree] if mentions_metrics_file(tree) else [])

    lines = text.splitlines()
    candidates: list[MetricCandidate] = []
    seen: set[str] = set()

    def add(key: str, lineno: int) -> None:
        if key in seen:
            return
        seen.add(key)
        snippet = "\n".join(lines[max(0, lineno - 3):lineno + 2]).strip()
        candidates.append(MetricCandidate(key=key, context=snippet[: 2 * context_chars + len(key)]))

    def dict_keys(node, scope) -> None:
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    add(k.value, k.lineno)
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "dict":
            for kw in node.keywords:
                if kw.arg:
                    add(kw.arg, node.lineno)
        elif isinstance(node, ast.Name):
            for n in ast.walk(scope):
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == node.id for t in n.targets):
                    dict_keys(n.value, scope)
                elif (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Subscript)
                      and getattr(n.targets[0].value, "id", None) == node.id
                      and isinstance(n.targets[0].slice, ast.Constant) and isinstance(n.targets[0].slice.value, str)):
                    add(n.targets[0].slice.value, n.lineno)

    for scope in scopes:
        for n in ast.walk(scope):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in ("dump", "dumps") and getattr(n.func.value, "id", None) == "json" and n.args):
                dict_keys(n.args[0], scope)
    return candidates
