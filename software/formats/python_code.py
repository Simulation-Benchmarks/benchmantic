# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
software.formats.python_code

Inputs written in code: the top-level constants of a Python script (.py) or
notebook (.ipynb), e.g. `r = 0.05`, `dt = 1 / 1600`, or the one number
inside a call, as in `mu = Constant(mesh, PETSc.ScalarType(0.001))`. A
parameter's section is the file (relative to the case folder), its key the
constant's name. Values computed from other names (`res_min = r / 3`) are
not inputs: they follow from the inputs.

The generated workflow sets a case's values by rewriting the constants in a
copy of the code (see python_code_runtime and snakefile/styles/python_script.py).

Settings in the software description (under `inputs:`):
  files: [main.py]          the script(s) or notebook(s); default: the file
                            the implementation link points at

For metrics computed by the code (`metrics: {from: variables}`),
result_variables() finds candidates: values the code prints, and arrays it
fills in a loop (their largest and their last value).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from metadata.parameters import ParameterCandidate, ParameterLocation
from software.formats import python_code_runtime as runtime
from utils import read_text, to_number

SUFFIXES = (".py", ".ipynb")
#: Arrays named like these hold times, not results.
TIME_NAMES = re.compile(r"^(t|ts|time|times|t_\w+|\w+_t|time_\w+|\w+_time|\w+_times)$", re.IGNORECASE)
_CELL = re.compile(r"^# %% \[cell (\d+)\]$")


def read_code(path: Path) -> str:
    """The code of a script, or of a notebook's code cells (python_code_runtime.code_view)."""
    return runtime.code_view(read_text(path), notebook=path.suffix == ".ipynb")


def _files(settings: dict[str, Any]) -> list[str]:
    return list(settings.get("files") or [])


def input_files(case_dir: Path, settings: dict[str, Any]) -> list[Path]:
    return [case_dir / name for name in _files(settings) if (case_dir / name).is_file()]


def sniff(path: Path) -> int:
    """How many constants `path` holds if it is a script or notebook, else 0."""
    if path.suffix not in SUFFIXES:
        return 0
    try:
        return len(runtime.constants(read_code(path)))
    except (OSError, ValueError):
        return 0


def _text(value: Any) -> str:
    return " ".join(repr(v) for v in value) if isinstance(value, list) else repr(value)


def _cell(lines: list[str], line: int) -> int | None:
    for text in reversed(lines[:line]):
        m = _CELL.match(text)
        if m:
            return int(m.group(1))
    return None


def _usage(lines: list[str], name: str, own: int, limit: int = 6) -> list[str]:
    pattern = re.compile(rf"(?<![\w.]){re.escape(name)}(?!\w)")
    return [f"{n}: {t.strip()}" for n, t in enumerate(lines, start=1)
            if n != own and pattern.search(t) and not t.lstrip().startswith("#")][:limit]


def _location(section: str, lines: list[str], c: dict[str, Any]) -> ParameterLocation:
    line, column = c["start"]
    text = lines[line - 1] if 0 < line <= len(lines) else ""
    start = len(text.encode()[:column].decode(errors="ignore"))
    spans = []
    if c["end"][0] == line:
        end = len(text.encode()[:c["end"][1]].decode(errors="ignore"))
        spans = [(start + 1, end + 1)]
    return ParameterLocation(file=section, line=line, column=start + 1, token_spans=spans)


def discover(case_dir: Path, settings: dict[str, Any]) -> list[ParameterCandidate]:
    candidates = []
    for path in input_files(case_dir, settings):
        section = str(path.relative_to(case_dir))
        code = read_code(path)
        lines = code.splitlines()
        for c in runtime.constants(code):
            own = c["line"]
            context = "\n".join(lines[max(0, own - 3):own + 2]).strip()
            usage = _usage(lines, c["name"], own)
            if usage:
                context += "\n... used in:\n" + "\n".join(usage)
            cell = _cell(lines, own) if path.suffix == ".ipynb" else None
            hint = f"Python constant `{c['name']}` in {section}" + (f", notebook cell {cell}" if cell else "")
            if c["wrapped"]:
                hint += " -- the number inside the call on that line"
            value = _text(c["value"])
            candidates.append(ParameterCandidate(
                section=section, key=c["name"], value=value, tokens=value.split(),
                cpp_hint=hint, code_context=context, location=_location(section, lines, c),
            ))
    return candidates


def resolve(case_dir: Path, parameter_fields: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    known: dict[str, dict[str, dict[str, Any]]] = {}
    values: dict[str, Any] = {}
    for key, spec in parameter_fields.items():
        section, name = spec["ini"]
        if section not in known:
            known[section] = {c["name"]: c for c in runtime.constants(read_code(case_dir / section))}
        c = known[section].get(name)
        if c is None:
            raise ValueError(f"In case '{case_dir}': {section} has no top-level constant {name}")
        numbers = c["value"] if isinstance(c["value"], list) else [c["value"]]
        if spec.get("full_value") or (len(numbers) > 1 and spec.get("index") is None):
            values[key] = list(numbers)
        else:
            index = spec.get("index", 0) or 0
            if not -len(numbers) <= index < len(numbers):
                raise ValueError(f"In case '{case_dir}': {section} {name} has no value {index}")
            values[key] = to_number(repr(numbers[index]))
    return values


def read_entry(path: Path, key: str) -> str | None:
    """The value of the constant named `key`."""
    for c in runtime.constants(read_code(path)):
        if c["name"] == key:
            return _text(c["value"])
    return None


# --- metrics computed by the code ----------------------------------------------------------
def _slug(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()


def result_variables(code: str) -> list[tuple[str, str, int]]:
    """Candidate metrics of a script: (metric key, Python expression over
    the script's variables at its end, line). From
      - printed values: print(f"Drag coefficient: {C_D}") -> drag_coefficient = C_D
      - arrays filled in a loop: C_D[i] = ... -> C_D_max = max(C_D), C_D_final = C_D[-1]
        (arrays named like times are left out)."""
    try:
        tree = runtime.parse(code)
    except SyntaxError:
        return []
    found: dict[str, tuple[str, int]] = {}
    # dicts written out with string keys: {'skfem': ..., 'exact': ...} -> one metric per entry
    dicts: dict[str, list[str]] = {}
    for statement in runtime._top_level(tree.body):
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name) and isinstance(statement.value, ast.Dict)
                and statement.value.keys
                and all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in statement.value.keys)):
            dicts[statement.targets[0].id] = [k.value for k in statement.value.keys]

    def add(label: str, expression: str | None, line: int) -> None:
        key = _slug(label) or _slug(expression or "")
        if expression in dicts:
            for entry in dicts[expression]:
                add(f"{key}_{entry}", f"{expression}[{entry!r}]", line)
            return
        if expression and key and key not in found and not key.isdigit():
            found[key] = (expression, line)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print"):
            continue
        label = ""
        for arg in node.args:
            if isinstance(arg, ast.JoinedStr):
                text = ""
                for part in arg.values:
                    if isinstance(part, ast.Constant):
                        text += str(part.value)
                    elif isinstance(part, ast.FormattedValue):
                        add(text, ast.get_source_segment(code, part.value), node.lineno)
                        text = ""
            elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                label = arg.value
            else:
                add(label, ast.get_source_segment(code, arg), node.lineno)
                label = ""

    allocators = {"zeros", "empty", "ones", "full", "zeros_like", "empty_like", "ones_like", "full_like"}
    arrays: dict[str, int] = {}
    for statement in runtime._top_level(tree.body):
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            value = statement.value
            func = value.func if isinstance(value, ast.Call) else None
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name in allocators or (isinstance(value, ast.List) and not value.elts):
                arrays.setdefault(statement.targets[0].id, statement.lineno)
    filled: dict[str, int] = {}
    for loop in ast.walk(tree):
        if not isinstance(loop, (ast.For, ast.While)):
            continue
        for node in ast.walk(loop):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                        filled.setdefault(target.value.id, node.lineno)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "append" and isinstance(node.func.value, ast.Name)):
                filled.setdefault(node.func.value.id, node.lineno)
    for name, line in filled.items():
        if name in arrays and not TIME_NAMES.match(name):
            add(f"{name}_max", f"max({name})", line)
            add(f"{name}_final", f"{name}[-1]", line)
    return [(key, expression, line) for key, (expression, line) in found.items()]


def expression_context(code: str, expression: str, line: int | None = None, limit: int = 6) -> str:
    """Code lines that explain a metric expression: where its names are set."""
    lines = code.splitlines()
    names = sorted({n.id for n in ast.walk(ast.parse(expression, mode="eval")) if isinstance(n, ast.Name)}
                   - {"max", "min", "abs", "sum", "len", "float"})
    shown: list[str] = []
    for name in names:
        shown += _usage(lines, name, own=-1, limit=limit)
    head = f"metric value = {expression}, evaluated when the script ends"
    if line:
        head += f" (found at line {line})"
    return head + ("\n" + "\n".join(dict.fromkeys(shown)) if shown else "")
