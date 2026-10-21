# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
software.formats.json_params

Inputs in a JSON parameter file, e.g.

    {"configuration": "1",
     "radius": {"value": 0.33, "unit": "m"},
     "element-order": 1,
     "quadrature-rule": "gauss",
     "material": {"young-modulus": {"value": 210e9, "unit": "Pa"}}}

Every entry is an input; an object with a "value" is one input (its "unit"
is passed on as a hint), other objects are walked into, and their entries
are written "material/young-modulus". A list of numbers is one input with
several values. A top-level "configuration" names the case and is not an
input.

Settings in the software description (under `inputs:`):
  files: [parameters.json]  the parameter file(s) of a case (with
                            `cases.files`, each case's file is found by that)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from metadata.parameters import ParameterCandidate, ParameterLocation, token_spans
from utils import read_text, to_number

#: Top-level keys that name the case rather than set an input.
NOT_INPUTS = {"configuration"}


def _files(settings: dict[str, Any]) -> list[str]:
    return list(settings.get("files") or [])


def input_files(case_dir: Path, settings: dict[str, Any]) -> list[Path]:
    return [case_dir / name for name in _files(settings) if (case_dir / name).is_file()]


def load(path: Path) -> dict[str, Any]:
    data = json.loads(read_text(path))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return data


def entries(data: dict[str, Any], prefix: str = "") -> list[tuple[str, Any, str | None]]:
    """(key path, value, unit) for every input in a parameter file."""
    out = []
    for key, value in data.items():
        if not prefix and key in NOT_INPUTS:
            continue
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict) and "value" in value:
            out.append((path, value["value"], value.get("unit")))
        elif isinstance(value, dict):
            out += entries(value, path)
        else:
            out.append((path, value, None))
    return out


def sniff(path: Path) -> int:
    """How many inputs `path` holds if it is a JSON parameter file, else 0."""
    if path.suffix.lower() != ".json":
        return 0
    try:
        if path.stat().st_size > 1_000_000:
            return 0
        return len(entries(load(path)))
    except (OSError, ValueError):
        return 0


def _text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(_text(v) for v in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return value if isinstance(value, str) else json.dumps(value)


def _locate(lines: list[str], path: str, section: str, value: Any) -> ParameterLocation | None:
    """Where an entry is: each key of the path is looked for after the
    previous one; the value starts after the last key (or after its
    "value":)."""
    line, column = 0, 0
    for name in path.split("/"):
        pattern = re.compile(r'"' + re.escape(name) + r'"\s*:\s*')
        for n in range(line, len(lines)):
            m = pattern.search(lines[n], column if n == line else 0)
            if m:
                line, column = n, m.end()
                break
        else:
            return None
    text = lines[line]
    m = re.compile(r'\{\s*"value"\s*:\s*').match(text, column)
    if m:
        column = m.end()
    elif text[column:].strip() in ("", "{"):            # the "value" is on a later line
        for n in range(line + 1, min(line + 5, len(lines))):
            m = re.search(r'"value"\s*:\s*', lines[n])
            if m:
                line, column, text = n, m.end(), lines[n]
                break
    shown = _text(value)
    return ParameterLocation(file=section, line=line + 1, column=column + 1,
                             token_spans=token_spans(text, column, shown) if not isinstance(value, str) else [])


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _raw_number(lines: list[str], where: ParameterLocation) -> str | None:
    """The number as written in the file (210e9, not 210000000000.0)."""
    m = _NUMBER.match(lines[where.line - 1], where.column - 1)
    if m:
        where.token_spans = [(m.start() + 1, m.end() + 1)]
        return m.group()
    return None


def discover(case_dir: Path, settings: dict[str, Any]) -> list[ParameterCandidate]:
    candidates = []
    for path in input_files(case_dir, settings):
        section = str(path.relative_to(case_dir))
        lines = read_text(path).splitlines()
        for key, value, unit in entries(load(path)):
            where = _locate(lines, key, section, value)
            shown = _raw_number(lines, where) if where and _is_number(value) else None
            shown = shown or _text(value)
            hint = f"JSON parameter file entry \"{key}\"" + (f" with unit \"{unit}\"" if unit else "")
            context = "\n".join(lines[max(0, where.line - 3):where.line + 2]).strip() if where else ""
            candidates.append(ParameterCandidate(
                section=section, key=key, value=shown,
                tokens=shown.split() if not isinstance(value, str) else [shown],
                cpp_hint=hint, code_context=context, location=where,
            ))
    return candidates


def _get(data: dict[str, Any], key: str) -> Any:
    node: Any = data
    for name in key.split("/"):
        if not isinstance(node, dict) or name not in node:
            raise KeyError(key)
        node = node[name]
    return node["value"] if isinstance(node, dict) and "value" in node else node


def resolve(case_dir: Path, parameter_fields: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    loaded: dict[str, dict[str, Any]] = {}
    values: dict[str, Any] = {}
    for field, spec in parameter_fields.items():
        section, key = spec["ini"]
        if section not in loaded:
            loaded[section] = load(case_dir / section)
        try:
            value = _get(loaded[section], key)
        except KeyError:
            raise ValueError(f"In case '{case_dir}': {section} has no entry {key}") from None
        items = value if isinstance(value, list) else [value]
        if spec.get("full_value") or (len(items) > 1 and spec.get("index") is None):
            values[field] = list(items)
        else:
            index = spec.get("index", 0) or 0
            if not -len(items) <= index < len(items):
                raise ValueError(f"In case '{case_dir}': {section} {key} has no value {index}")
            item = items[index]
            values[field] = to_number(str(item)) if isinstance(item, (int, float)) and not isinstance(item, bool) \
                else item
    return values


def read_entry(path: Path, key: str) -> str | None:
    try:
        return _text(_get(load(path), key))
    except (KeyError, ValueError, OSError):
        return None
