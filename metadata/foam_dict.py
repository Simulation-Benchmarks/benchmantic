# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
metadata.foam_dict

A small reader for OpenFOAM dictionary files (system/controlDict,
constant/transportProperties, 0/U, ...). It flattens a dictionary into
entries with a slash-separated path, e.g. "MRF1/omega" or
"boundaryField/innerWall/type", keeping each entry's raw value text and
line number.

Handled: // and /* */ comments, nested sub-dictionaries, lists and vectors
"( ... )" (kept as raw value text), quoted strings, $variable references,
directives such as #include, #eval{ ... } / #calc "...", and template
placeholders "{name}" (as used by benchmark templates, e.g. "omega {omega};").
Not a full OpenFOAM parser -- it's for discovering input parameters, not
for evaluating a case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

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
