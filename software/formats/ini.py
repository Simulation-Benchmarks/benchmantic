# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
software.formats.ini

INI input files ([Section] + key = value), e.g. DUNE/DuMux params.input.
A parameter's (section, key) is the INI section and the key; a value's
tokens are its whitespace-separated parts.

Settings in the software description (under `inputs:`):
  files: [params.input]     the input file(s) in each case folder
  search: {...}             find the file if it isn't called that
                            (see software/param_files.py)
"""

from __future__ import annotations

import configparser
import re
from pathlib import Path
from typing import Any

from metadata.parameters import ParameterCandidate, ParameterLocation, token_spans
from utils import to_number


def parse(path: Path) -> dict[str, dict[str, str]]:
    """{section: {key: value}}"""
    cp = configparser.ConfigParser(inline_comment_prefixes=("#",))
    cp.optionxform = str
    cp.read_string(path.read_text(encoding="utf-8"))
    return {section: dict(cp.items(section)) for section in cp.sections()}


_SECTION_LINE = re.compile(r"^\s*\[([^\]]+)\]")
_KEY_LINE = re.compile(r"^\s*([^=:#;\[\s][^=:]*?)\s*[=:]\s*")


def locate(path: Path, case_dir: Path | None = None) -> dict[tuple[str, str], ParameterLocation]:
    """{(section, key): where it is} for one INI file. Reads the lines
    itself, because configparser doesn't keep positions. A key set twice
    keeps its last position, as configparser keeps its last value."""
    rel = str(path.relative_to(case_dir)) if case_dir else path.name
    found: dict[tuple[str, str], ParameterLocation] = {}
    section = None
    for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if text.lstrip().startswith(("#", ";")):
            continue
        m = _SECTION_LINE.match(text)
        if m:
            section = m.group(1).strip()
            continue
        m = _KEY_LINE.match(text)
        if m and section is not None:
            value = text[m.end():].split("#", 1)[0].rstrip()
            found[(section, m.group(1).strip())] = ParameterLocation(
                file=rel, line=number, column=m.end() + 1, token_spans=token_spans(text, m.end(), value))
    return found


def sniff(path: Path) -> int:
    """How many entries `path` holds if it reads as an INI file with at
    least one [Section] and one key = value; 0 if it isn't one."""
    try:
        if path.stat().st_size > 1_000_000:
            return 0
        sections = parse(path)
    except (OSError, UnicodeDecodeError, configparser.Error):
        return 0
    return sum(len(kv) for kv in sections.values())


def _files(settings: dict[str, Any]) -> list[str]:
    return list(settings.get("files") or [])


def input_files(case_dir: Path, settings: dict[str, Any]) -> list[Path]:
    return [case_dir / name for name in _files(settings) if (case_dir / name).is_file()]


def discover(case_dir: Path, settings: dict[str, Any]) -> list[ParameterCandidate]:
    candidates = []
    for path in input_files(case_dir, settings):
        where = locate(path, case_dir)
        for section, kv in parse(path).items():
            for key, value in kv.items():
                candidates.append(ParameterCandidate(section=section, key=key, value=value, tokens=value.split(),
                                                     location=where.get((section, key))))
    return candidates


def _merged(case_dir: Path, settings: dict[str, Any]) -> dict[str, dict[str, str]]:
    ini: dict[str, dict[str, str]] = {}
    for name in _files(settings):
        for section, kv in parse(case_dir / name).items():
            ini.setdefault(section, {}).update(kv)
    return ini


def _tokens(ini: dict, section: str, key: str, where: str) -> list[str]:
    if section not in ini or key not in ini[section]:
        raise KeyError(f"[{section}] {key} not found in {where}")
    tokens = ini[section][key].split()
    if not tokens:
        raise ValueError(f"[{section}] {key} has no value")
    return tokens


def _token(ini: dict, section: str, key: str, index: int, where: str) -> str:
    tokens = _tokens(ini, section, key, where)
    raw_value = ini[section][key]
    if not (-len(tokens) <= index < len(tokens)):
        raise ValueError(
            f"[{section}] {key} = '{raw_value}' has {len(tokens)} token(s) "
            f"(0-based indices 0..{len(tokens) - 1}), but the parameter metadata "
            f"requests token index {index}. This usually means either: (a) the "
            f"LLM picked an index that doesn't exist in the template {where} "
            f"it was shown -- check/edit the cached '.parameter_metadata_cache.json' "
            f"entry for [{section}] {key}, or (b) this specific case's {where} "
            f"has a different number of values for [{section}] {key} than the "
            f"template did -- check that file directly."
        )
    return tokens[index]


def resolve(case_dir: Path, parameter_fields: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    where = ", ".join(_files(settings))
    ini = _merged(case_dir, settings)
    values: dict[str, Any] = {}
    for key, spec in parameter_fields.items():
        section, ini_key = spec["ini"]
        try:
            if spec.get("full_value"):
                values[key] = [to_number(t) for t in _tokens(ini, section, ini_key, where)]
            else:
                values[key] = to_number(_token(ini, section, ini_key, spec["index"], where))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"In case '{case_dir}': {exc}") from exc
    return values


def read_entry(path: Path, key: str) -> str | None:
    """`key` as "Section.Key"."""
    section, _, name = key.partition(".")
    try:
        return parse(path).get(section, {}).get(name)
    except (OSError, configparser.Error):
        return None
