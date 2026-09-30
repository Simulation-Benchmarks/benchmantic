# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
metadata.mapping

How a benchmark's software-neutral parameters (inner_radius, cells_radial,
...) become one simulation code's inputs (Grid.Radial0, MRF1/omega, ...).

Each software input gets an *expression*: a Python f-string body whose
{...} fields are arithmetic over benchmark parameter names, e.g.

    Grid.Cells0   <- "{cells_radial // 2} {cells_radial // 2}"
    Grid.Radial0  <- "{inner_radius} {(inner_radius + outer_radius) / 2} {outer_radius}"
    geom/ntheta   <- "{cells_angular // 4}"
    MRF1/omega    <- "{angular_velocity_inner_cylinder}"

An input with expression None is not driven by the benchmark and keeps its
own value. The same expressions are pasted into the generated Snakefile,
so what's checked here is exactly what runs.

The mapping is written next to the benchmark description as
<name>_mapping.json -- it's specific to one software, whereas the
description is software-neutral.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAPPING_SUFFIX = "_mapping.json"
_FIELD_PATTERN = re.compile(r"\{([^{}]+)\}")
def tokens(value: Any) -> list:
    """The values of a multi-value parameter as a list of numbers (text
    tokens stay text): tokens("1.0 1.5 2.0") -> [1.0, 1.5, 2.0], so an
    expression can pick one, e.g. {tokens(radial_coordinates)[0]}. The
    generated Snakefile defines the same function."""
    items = value if isinstance(value, (list, tuple)) else str(value).replace("(", " ").replace(")", " ").split()
    out = []
    for item in items:
        try:
            num = float(item)
            out.append(int(num) if num.is_integer() and "." not in str(item) else num)
        except (TypeError, ValueError):
            out.append(item)
    return out


_ALLOWED_FUNCS = {"int": int, "float": float, "round": round, "abs": abs, "min": min, "max": max,
                  "str": str, "sqrt": math.sqrt, "pi": math.pi, "tokens": tokens, "len": len}
_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load, ast.Call, ast.Subscript,
                  ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
                  ast.IfExp, ast.Compare, ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq)


def neutral_name(text: str) -> str:
    """A safe snake_case identifier for a benchmark parameter name
    ('Inner radius' -> 'inner_radius'), usable in expressions and as a
    Python variable in the Snakefile."""
    name = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower() or "parameter"
    return f"p_{name}" if name[0].isdigit() else name


def expression_fields(expression: str) -> list[str]:
    return _FIELD_PATTERN.findall(expression or "")


def expression_names(expression: str) -> set[str]:
    names: set[str] = set()
    for f in expression_fields(expression):
        try:
            tree = ast.parse(f.strip(), mode="eval")
        except SyntaxError:
            continue
        names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id not in _ALLOWED_FUNCS}
    return names


def check_expression(expression: str, parameters: set[str]) -> str | None:
    """None if `expression` is valid over `parameters`, else the problem."""
    fields = expression_fields(expression)
    if not fields:
        return "no {...} field -- use a literal only if the input shouldn't be mapped (then use null)"
    for f in fields:
        try:
            tree = ast.parse(f.strip(), mode="eval")
        except SyntaxError as exc:
            return f"'{{{f}}}' is not a valid expression ({exc.msg})"
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                return f"'{{{f}}}' uses {type(node).__name__}, which isn't allowed"
            if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS):
                return f"'{{{f}}}' calls a function that isn't allowed"
            if isinstance(node, ast.Subscript) and not (
                    isinstance(node.value, ast.Call) and getattr(node.value.func, "id", None) == "tokens"):
                return (f"'{{{f}}}' indexes a parameter directly -- use tokens(name)[i] to pick one value "
                        "of a multi-value parameter")
        unknown = expression_names(f"{{{f}}}") - parameters
        if unknown:
            return f"unknown benchmark parameter(s): {', '.join(sorted(unknown))}"
    return None


def _eval_field(expr: str, values: dict[str, Any]) -> Any:
    tree = ast.parse(expr.strip(), mode="eval")
    return eval(compile(tree, "<mapping>", "eval"), {"__builtins__": {}, **_ALLOWED_FUNCS}, dict(values))  # noqa: S307


def render(expression: str, values: dict[str, Any]) -> Any:
    """Evaluate an expression for one configuration. A lone field keeps its
    type ('{omega}' -> 100.0); anything else becomes the formatted string,
    exactly as the Snakefile's f-string would produce it."""
    stripped = expression.strip()
    fields = expression_fields(stripped)
    if len(fields) == 1 and stripped == f"{{{fields[0]}}}":
        return _eval_field(fields[0], values)
    return _FIELD_PATTERN.sub(lambda m: str(_eval_field(m.group(1), values)), expression)


def values_match(rendered: Any, actual: Any, rel_tol: float = 1e-9) -> bool:
    """Token-wise comparison of a rendered value with a case's actual value."""
    def tokens(v):
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
        return str(v).replace("(", " ").replace(")", " ").split()
    a, b = tokens(rendered), tokens(actual)
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        try:
            if not math.isclose(float(x), float(y), rel_tol=rel_tol, abs_tol=1e-12):
                return False
        except ValueError:
            if x != y:
                return False
    return True


@dataclass
class InputMapping:
    section: str
    key: str
    display: str                 # "Grid.Cells0" / "constant/MRFProperties:MRF1/omega"
    expression: str | None       # None = not driven by the benchmark
    template_value: str = ""
    confidence: float | None = None
    explanation: str = ""
    placeholder: str | None = None   # OpenFOAM template placeholder name, if any

    def to_json(self) -> dict[str, Any]:
        d = {"input": self.display, "section": self.section, "key": self.key, "expression": self.expression}
        if self.expression:
            d["benchmark_parameters"] = sorted(expression_names(self.expression))
        if self.placeholder:
            d["placeholder"] = self.placeholder
        if self.template_value:
            d["template_value"] = self.template_value
        if self.confidence is not None:
            d["confidence"] = self.confidence
        if self.explanation:
            d["explanation"] = self.explanation
        return d


@dataclass
class ParameterMapping:
    software: str
    inputs: list[InputMapping]
    benchmark_parameters: dict[str, dict[str, Any]]    # name -> {"unit": ..., "quantityKind": ...}
    reference: dict[str, Any] | None = None
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def mapped(self) -> list[InputMapping]:
        return [m for m in self.inputs if m.expression]

    def used_parameters(self) -> set[str]:
        return set().union(*(expression_names(m.expression) for m in self.mapped)) if self.mapped else set()

    def unused_parameters(self) -> list[str]:
        return sorted(set(self.benchmark_parameters) - self.used_parameters())

    def to_json(self) -> dict[str, Any]:
        return {
            "software": self.software,
            "reference": self.reference,
            "benchmark_parameters": self.benchmark_parameters,
            "inputs": [m.to_json() for m in self.inputs],
            "unused_benchmark_parameters": self.unused_parameters(),
            "checks": self.checks,
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_mapping(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def input_display(section: str, key: str, software: str) -> str:
    return f"{section}.{key}" if software.lower().startswith("dumu") else f"{section}:{key}"


def identity_expression(name: str, template_value: str, index: int | None, full_value: bool) -> str:
    """The expression for a 1:1 input (no reference benchmark): the input
    takes the benchmark parameter's value. For a multi-token input where
    only one token is the parameter, the other tokens stay as in the
    template -- e.g. 'Grading0 = 1.1 -1.1', index 0 -> '{mesh_grading} -1.1'."""
    # Full-value (multi-token) parameters are stored as one space-joined
    # string, so a plain {name} reproduces the whole value.
    stripped = template_value.strip()
    prefix = "uniform " if stripped.startswith("uniform ") else ""
    body = stripped[len(prefix):].strip()
    if body.startswith("(") and body.endswith(")"):  # OpenFOAM vector: keep the brackets
        return f"{prefix}({{{name}}})"
    if prefix:
        return f"{prefix}{{{name}}}"
    tokens = template_value.split()
    if full_value or len(tokens) <= 1 or index is None or not -len(tokens) <= index < len(tokens):
        return f"{{{name}}}"
    tokens[index] = f"{{{name}}}"
    return " ".join(tokens)


def check_against_configurations(mapping: ParameterMapping, configurations: list[dict[str, Any]]) -> list[str]:
    """Evaluate every mapped input for every configuration. Returns the
    problems found (empty = all good) and stores a per-input example."""
    problems = []
    names = set(mapping.benchmark_parameters)
    for m in mapping.mapped:
        issue = check_expression(m.expression, names)
        if issue:
            problems.append(f"{m.display}: {issue}")
            continue
        for conf in configurations:
            try:
                render(m.expression, conf["values"])
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{m.display}: fails for configuration {conf['id']} ({exc})")
                break
    mapping.checks["expression_problems"] = problems
    return problems


def template_consistency(mapping: ParameterMapping, configurations: list[dict[str, Any]]) -> dict[str, list[str]]:
    """For inputs with a concrete template value: which configurations
    reproduce it. An empty list means the expression never reproduces the
    template -- not necessarily wrong (the template may be none of the
    configurations), but worth a look."""
    result = {}
    for m in mapping.mapped:
        if not m.template_value or m.placeholder:
            continue
        hits = []
        for conf in configurations:
            try:
                if values_match(render(m.expression, conf["values"]), m.template_value):
                    hits.append(conf["id"])
            except Exception:  # noqa: BLE001
                pass
        result[m.display] = hits
    mapping.checks["template_matches"] = result
    return result


def mapping_path_for(benchmark_path: Path) -> Path | None:
    """The mapping file that belongs to a benchmark description:
    'x_benchmark.jsonld' -> 'x_mapping.json' (final output), or
    'staged.jsonld' -> 'staged.mapping.json' (inside a run)."""
    stem = benchmark_path.stem
    names = [f"{stem[: -len('_benchmark')]}{MAPPING_SUFFIX}"] if stem.endswith("_benchmark") else []
    names += [f"{stem}.mapping.json", f"{stem}{MAPPING_SUFFIX}"]
    for name in names:
        path = benchmark_path.with_name(name)
        if path.exists():
            return path
    return None
