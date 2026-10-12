# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
software.formats.foam_dict

OpenFOAM dictionary input files (system/controlDict,
constant/transportProperties, 0/U, ...). A parameter's (section, key) is the
dictionary file relative to the case ("constant/MRFProperties") and the
entry's slash-separated path inside it ("MRF1/omega"). Template placeholders
"{name}" (e.g. "omega {omega};") mark inputs a benchmark sets per run.

The reader handles // and /* */ comments, nested sub-dictionaries, lists and
vectors "( ... )" (kept as raw value text), quoted strings, $variable
references and directives such as #include, #eval{ ... } / #calc "...". Not
a full OpenFOAM parser -- it's for discovering input parameters, not for
evaluating a case.

Settings in software.yaml (under `inputs:`):
  folders: [system, constant, "0", "0.orig"]   case sub-folders whose
                                               dictionaries are inputs
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from metadata.parameters import ParameterCandidate
from utils import read_text, to_number

PLACEHOLDER_PATTERN = re.compile(r"^\{(\w+)\}$")
_NUMBER_PATTERN = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


@dataclass
class FoamEntry:
    path: str        # "MRF1/omega"
    value: str       # raw value text, e.g. "100", "(0 0 1)", "{omega}", "uniform (0 0 0)"
    line: int        # 1-based line of the key

    @property
    def placeholder(self) -> str | None:
        m = PLACEHOLDER_PATTERN.match(self.value.strip())
        return m.group(1) if m else None

    @property
    def tokens(self) -> list[str]:
        """Value tokens with list parentheses removed: '(0 0 1)' -> ['0','0','1']."""
        return self.value.replace("(", " ").replace(")", " ").split()

    @property
    def is_numeric(self) -> bool:
        toks = self.tokens
        return bool(toks) and all(_NUMBER_PATTERN.match(t) for t in toks)


def _tokenize(text: str) -> list[tuple[str, int]]:
    """(token, line) pairs; comments dropped."""
    tokens: list[tuple[str, int]] = []
    i, n, line = 0, len(text), 1
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
            i += 1
        elif c.isspace():
            i += 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j == -1 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j == -1 else j + 2
            line += text.count("\n", i, j)
            i = j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            tokens.append((text[i:j + 1], line))
            line += text.count("\n", i, j + 1)
            i = j + 1
        elif c == "#":
            m = re.match(r"#\w+", text[i:])
            word = m.group(0) if m else "#"
            j = i + len(word)
            k = j
            while k < n and text[k] in " \t":
                k += 1
            if k < n and text[k] == "{":  # #eval{ ... } / #codeStream{ ... }: one token
                depth, k2 = 0, k
                while k2 < n:
                    depth += {"{": 1, "}": -1}.get(text[k2], 0)
                    k2 += 1
                    if depth == 0:
                        break
                tokens.append((text[i:k2], line))
                line += text.count("\n", i, k2)
                i = k2
            else:
                tokens.append((word, line))
                i = j
        elif c == "{":
            m = re.match(r"\{\w+\}", text[i:])
            if m:  # template placeholder
                tokens.append((m.group(0), line))
                i += len(m.group(0))
            else:
                tokens.append(("{", line))
                i += 1
        elif c in "}();[]":
            tokens.append((c, line))
            i += 1
        else:
            m = re.match(r'[^\s{}();\[\]"]+', text[i:])
            tokens.append((m.group(0), line))
            i += len(m.group(0))
    return tokens


def parse_foam_dict(text: str) -> list[FoamEntry]:
    """Flatten a dictionary's entries. The FoamFile header is skipped."""
    toks = _tokenize(text)
    entries: list[FoamEntry] = []

    def parse_block(i: int, prefix: list[str]) -> int:
        while i < len(toks):
            tok, line = toks[i]
            if tok == "}":
                return i + 1
            if tok in (";", "(", ")", "[", "]"):
                i += 1
                continue
            if tok.startswith("#") and not tok.startswith("#eval") and "{" not in tok:
                # #include "file", #includeEtc "...", #inputMode merge: one argument
                i += 2 if tok not in ("#remove",) else 2
                continue
            key = tok.strip('"')
            if i + 1 < len(toks) and toks[i + 1][0] == "{":
                if key == "FoamFile" and not prefix:
                    i = _skip_block(i + 1)
                else:
                    i = parse_block(i + 2, prefix + [key])
                continue
            # value: tokens up to ';' at nesting depth 0
            j, depth, parts = i + 1, 0, []
            while j < len(toks):
                t = toks[j][0]
                if t in ("(", "[", "{"):
                    depth += 1
                elif t in (")", "]", "}"):
                    if depth == 0:
                        break
                    depth -= 1
                elif t == ";" and depth == 0:
                    break
                parts.append(t)
                j += 1
            if parts:
                entries.append(FoamEntry("/".join(prefix + [key]), _join(parts), line))
            i = j + 1 if j < len(toks) and toks[j][0] == ";" else j
        return i

    def _skip_block(i: int) -> int:
        depth = 0
        while i < len(toks):
            depth += {"{": 1, "}": -1}.get(toks[i][0], 0)
            i += 1
            if depth == 0:
                return i
        return i

    parse_block(0, [])
    return entries


def _join(parts: list[str]) -> str:
    out = " ".join(parts)
    return re.sub(r"\( ", "(", re.sub(r" \)", ")", out))


def read_foam_dict(path: Path) -> list[FoamEntry]:
    try:
        return parse_foam_dict(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:  # noqa: BLE001 -- a file we can't read is just skipped
        return []


def is_foam_dict(path: Path) -> bool:
    """True for OpenFOAM dictionary files (FoamFile header near the top)."""
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:3000]
    except Exception:  # noqa: BLE001
        return False
    return "FoamFile" in head


# ---------------------------------------------------------------------------
# Inputs of a case
# ---------------------------------------------------------------------------

#: Case sub-folders whose dictionaries are inputs (default for `folders`).
DEFAULT_FOLDERS = ("system", "constant", "0", "0.orig")
#: Never parameters: mesh/geometry data and generated folders.
SKIP_PARTS = {"polyMesh", "triSurface", "extendedFeatureEdgeMesh", "processor0", ".benchmantic"}
#: Entry paths that are bookkeeping or post-processing, not inputs.
SKIP_ENTRY = re.compile(r"^(functions/|dimensions$|libs$|convertToMeters$|scale$)")
#: Longest value (in tokens) still treated as a parameter (vectors, small lists).
MAX_VALUE_TOKENS = 9


def input_files(case_dir: Path, settings: dict[str, Any]) -> list[Path]:
    files = []
    for sub in settings.get("folders") or DEFAULT_FOLDERS:
        base = case_dir / str(sub)
        if base.is_dir():
            files += [p for p in sorted(base.rglob("*"))
                      if p.is_file() and not (SKIP_PARTS & set(p.relative_to(case_dir).parts)) and is_foam_dict(p)]
    return files


def _usage_lines(entry: FoamEntry, entries: list[FoamEntry], lines: list[str], limit: int = 8) -> list[str]:
    """Where an entry is referenced in its file ($/geom/nr, $nr), following
    one level of macros -- e.g. blockMeshDict's `blockInfo` uses $/geom/nr
    and each of the 4 hex blocks uses $blockInfo. Tells the LLM how the
    value is used (per block, per zone, ...)."""
    refs = {f"$/{entry.path}", f"${entry.path}", f"${entry.path.split('/')[-1]}"}

    def lines_referencing(names: set[str]) -> list[int]:
        pattern = re.compile("|".join(re.escape(n) + r"(?![\w/])" for n in names))
        return [i for i, text in enumerate(lines) if pattern.search(text) and i + 1 != entry.line]

    hits = lines_referencing(refs)
    macros = {f"${e.path}" for e in entries if "/" not in e.path and any(r in e.value for r in refs)}
    if macros:
        hits += [i for i in lines_referencing(macros) if i not in hits]
    return [lines[i].strip() for i in sorted(hits)[:limit]]


def _is_parameter(section: str, entry: FoamEntry) -> bool:
    if SKIP_ENTRY.search(entry.path):
        return False
    value = entry.value.strip()
    if entry.placeholder:
        return True
    if value.startswith(("$", "#")) or "$" in value:
        return False  # derived from another entry
    tokens = entry.tokens
    if tokens and tokens[0] == "uniform":
        tokens = tokens[1:]
    if not tokens or len(tokens) > MAX_VALUE_TOKENS:
        return False
    if all(re.match(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$", t) for t in tokens):
        return True
    # single-word choices (models etc.) only in constant/
    return section.startswith("constant/") and len(tokens) == 1


def discover(case_dir: Path, settings: dict[str, Any]) -> list[ParameterCandidate]:
    candidates = []
    for path in input_files(case_dir, settings):
        section = str(path.relative_to(case_dir))
        lines = read_text(path).splitlines()
        entries = read_foam_dict(path)
        for entry in entries:
            if not _is_parameter(section, entry):
                continue
            context = "\n".join(lines[max(0, entry.line - 3):entry.line + 2]).strip()
            usage = _usage_lines(entry, entries, lines)
            if usage:
                context += "\n... used in:\n" + "\n".join(usage)
            if entry.placeholder:
                hint = (f"template placeholder {{{entry.placeholder}}} in OpenFOAM dictionary {section}, "
                        f"entry {entry.path} -- set per benchmark run")
            else:
                hint = f"OpenFOAM dictionary {section}, entry {entry.path}"
            candidates.append(ParameterCandidate(
                section=section, key=entry.path, value=entry.value, tokens=entry.tokens,
                cpp_hint=hint, code_context=context,
            ))
    return candidates


def resolve(case_dir: Path, parameter_fields: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    parsed: dict[str, dict[str, FoamEntry]] = {}
    values: dict[str, Any] = {}
    placeholders = []
    for key, spec in parameter_fields.items():
        section, entry_path = spec["ini"]
        if section not in parsed:
            parsed[section] = {e.path: e for e in read_foam_dict(case_dir / section)}
        entry = parsed[section].get(entry_path)
        if entry is None:
            raise ValueError(f"In case '{case_dir}': {section} has no entry {entry_path}")
        if entry.placeholder:
            placeholders.append(f"{{{entry.placeholder}}} ({section}: {entry_path})")
            continue
        tokens = entry.tokens[1:] if entry.tokens[:1] == ["uniform"] else entry.tokens
        if spec.get("full_value") or len(tokens) > 1 and spec.get("index") is None:
            values[key] = [to_number(t) for t in tokens]
        else:
            index = spec.get("index", 0) or 0
            if not -len(tokens) <= index < len(tokens):
                raise ValueError(f"In case '{case_dir}': {section} {entry_path} = '{entry.value}' "
                                 f"has no token {index}")
            values[key] = to_number(tokens[index])
    if placeholders:
        raise ValueError(
            f"In case '{case_dir}': these inputs are template placeholders with no value of their own: "
            + ", ".join(placeholders)
            + ". Pass --reference-benchmark <benchmark.jsonld> to take the configurations from an existing "
              "benchmark description, or point module_dir at a case with concrete values."
        )
    return values


def read_entry(path: Path, key: str) -> str | None:
    """`key` as the entry path, e.g. "application"."""
    for e in read_foam_dict(path):
        if e.path == key:
            return e.value.strip()
    return None
