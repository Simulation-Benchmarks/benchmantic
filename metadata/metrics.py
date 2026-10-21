# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
metadata.metrics

Software-neutral metric helpers: MetricCandidate, scanners for the JSON
keys a simulation or post-processing script writes to its results file
(as escaped JSON in source text, or as a Python dict dumped to
solution_metrics.json), and the RO-Crate field dicts built from
LLM-inferred (or cached) metadata. Which files a code's metrics come from is
up to its adapter (software/<name>/).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

@dataclass
class MetricCandidate:
    """A solution/output metric discovered in the sources (a JSON key the
    simulation or its post-processing writes to its results file)."""
    key: str
    context: str = ""


METRIC_KEY_PATTERN = re.compile(r'\\"([A-Za-z0-9_]+)\\"\s*:')


def discover_metrics(text: str, context_chars: int = 100) -> list[MetricCandidate]:
    """Find escaped JSON keys (\"key\": as written from C++ or a shell
    script) in `text` and grab a small snippet of
    surrounding code around each occurrence, so the LLM has some context
    (e.g. what's being computed/printed) to infer a sensible SI unit from.
    """
    candidates: list[MetricCandidate] = []
    seen: set[str] = set()
    for m in METRIC_KEY_PATTERN.finditer(text):
        key = m.group(1)
        if key in seen:
            continue
        seen.add(key)
        start = max(0, m.start() - context_chars)
        end = min(len(text), m.end() + context_chars)
        candidates.append(MetricCandidate(key=key, context=text[start:end].strip()))
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

    def dumps_metrics(node) -> bool:
        """json.dump(metrics, ...) -- the dumped variable is named like metrics
        (for scripts that are told the file name, e.g. --output_metrics_file)."""
        return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("dump", "dumps")
                   and getattr(n.func.value, "id", None) == "json" and n.args
                   and "metric" in getattr(n.args[0], "id", "").lower() for n in ast.walk(node))

    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    scopes = [s for s in functions if mentions_metrics_file(s)] or ([tree] if mentions_metrics_file(tree) else [])
    if not scopes:
        scopes = [s for s in functions if dumps_metrics(s)] or ([tree] if dumps_metrics(tree) else [])

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
