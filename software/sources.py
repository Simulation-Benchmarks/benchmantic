# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
software.sources

Ways of reading a benchmark implementation's sources that any software can
pick in its software.yaml -- the label and description of the benchmark,
the executable, hints on where the code reads an input, and the files its
metrics come from. Each function here is one named choice in the config
(e.g. `label: cpp_class`, `executable: {from: cmake}`).
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

from metadata.parameters import ParameterCandidate
from metadata.repository import find_repo_file
from utils import camel_to_label, read_text

# ---------------------------------------------------------------------------
# label / description
# ---------------------------------------------------------------------------

CLASS_NAME_PATTERN = re.compile(r"\bclass\s+([A-Z]\w+)\s*(?::\s*public|\{)")
DOXYGEN_BLOCK_PATTERN = re.compile(r"/\*!(.*?)\*/", re.DOTALL)


def cpp_class_label(texts: list[str]) -> str | None:
    """`label: cpp_class` -- the first C++ class name, as words
    ('RotatingCylindersProblem' -> 'rotating cylinders problem')."""
    for text in texts:
        if not text:
            continue
        m = CLASS_NAME_PATTERN.search(text)
        if m:
            return camel_to_label(m.group(1))
    return None


def doxygen_description(texts: list[str]) -> str:
    """`description: doxygen` -- the longest /*! ... */ comment block, with
    the leading '*' decoration and a leading \\brief tag stripped."""
    best = ""
    for text in texts:
        if not text:
            continue
        for block in DOXYGEN_BLOCK_PATTERN.findall(text):
            cleaned = re.sub(r"^[ \t]*\*[ \t]?", "", block, flags=re.MULTILINE).strip()
            cleaned = re.sub(r"^\\brief\s*", "", cleaned)
            if len(cleaned) > len(best):
                best = cleaned
    return best


def markdown_docs(module_dir: Path, repo_root: Path) -> list[Path]:
    """README*.md and docs/*.md next to the implementation, one folder up,
    and at the repository root."""
    docs = []
    for d in (module_dir, module_dir.parent, repo_root):
        docs += sorted(d.glob("README*.md")) + sorted((d / "docs").glob("*.md"))
    seen, out = set(), []
    for p in docs:
        if p.exists() and p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    return out


def markdown_title_label(module_dir: Path, repo_root: Path) -> str | None:
    """`label: markdown_title` -- the first '# Title' of the docs, without
    parentheses and the word 'benchmark', lower case."""
    for doc in markdown_docs(module_dir, repo_root):
        m = re.search(r"^#\s+(.+)$", read_text(doc), re.MULTILINE)
        if m:
            title = re.sub(r"\s*\(.*?\)\s*", " ", m.group(1))          # drop "(Taylor-Couette Flow)"
            title = re.sub(r"\bbenchmark\b", "", title, flags=re.IGNORECASE)
            title = " ".join(title.split())
            if title:
                return title.lower()
    return None


def markdown_section_description(module_dir: Path, repo_root: Path) -> str:
    """`description: markdown_section` -- title + 'Problem description'
    (or 'Description'/'Overview') section of the docs."""
    for doc in markdown_docs(module_dir, repo_root):
        text = read_text(doc)
        m = re.search(r"^#{1,3}\s*(problem description|description|overview)\s*\n(.*?)(?=^#{1,3}\s|\Z)",
                      text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if m:
            title = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
            body = re.sub(r"\s+", " ", m.group(2)).strip()
            return f"{title.group(1).strip()}\n\n{body}" if title else body
    return ""


# ---------------------------------------------------------------------------
# executable from CMake
# ---------------------------------------------------------------------------

#: CMake's literal add_executable(<target> <source1> <source2> ...).
ADD_EXECUTABLE_PATTERN = re.compile(r"add_executable\s*\(\s*([\w.\-]+)\s+([^)]*)\)", re.IGNORECASE | re.DOTALL)
NAME_ARG_PATTERN = re.compile(r"\bNAME\s+([\w.\-]+)", re.IGNORECASE)


def cmake_executable_in(text: str, test_macros: list[str], source: str | None) -> str | None:
    """The executable target in one CMakeLists.txt: first a test macro
    taking NAME <target> (e.g. dumux_add_test(NAME ... SOURCES main.cc)),
    then a literal add_executable(...). Within each, a call that mentions
    `source` wins; otherwise the first target of that kind."""
    if not text:
        return None
    source_pattern = re.compile(rf"\b{re.escape(source)}\b", re.IGNORECASE) if source else None

    if test_macros:
        macro_pattern = re.compile(r"(?:" + "|".join(re.escape(m) for m in test_macros) + r")\s*\((.*?)\)",
                                   re.IGNORECASE | re.DOTALL)
        first = None
        for m in macro_pattern.finditer(text):
            block = m.group(1)
            name_match = NAME_ARG_PATTERN.search(block)
            if not name_match:
                continue
            if first is None:
                first = name_match.group(1)
            if source_pattern and source_pattern.search(block):
                return name_match.group(1)
        if first:
            return first

    first_target = None
    for m in ADD_EXECUTABLE_PATTERN.finditer(text):
        target, sources = m.group(1), m.group(2)
        if first_target is None:
            first_target = target
        if source_pattern and source_pattern.search(sources):
            return target
    return first_target


def cmake_executable(module_dir: Path, repo_root: Path, test_macros: list[str],
                     source: str | None) -> tuple[str | None, Path | None]:
    """`executable: {from: cmake}` -- the CMakeLists.txt next to the
    implementation (or the nearest one above it), else the first one
    anywhere in the repository that declares a target."""
    direct = find_repo_file(("CMakeLists.txt",), module_dir, repo_root)
    if direct:
        name = cmake_executable_in(read_text(direct), test_macros, source)
        if name:
            return name, direct
    for candidate in sorted(repo_root.rglob("CMakeLists.txt")):
        if candidate == direct:
            continue
        name = cmake_executable_in(read_text(candidate), test_macros, source)
        if name:
            return name, candidate
    return None, None


# ---------------------------------------------------------------------------
# where the code reads an input (context for the LLM)
# ---------------------------------------------------------------------------

#: Chars of source grabbed on each side of a match for code_context.
HINT_CONTEXT_CHARS = 120


def attach_code_hints(candidates: list[ParameterCandidate], texts: list[str], settings: dict[str, Any]) -> None:
    """`inputs.code_hints` -- find where the source reads each input with a
    regular expression whose `key` group is the input's name as the code
    writes it (`key_format`, e.g. "{section}.{key}"); optional `var` and
    `type` groups fill in the hint text."""
    pattern = re.compile(settings["pattern"])
    key_format = settings.get("key_format", "{section}.{key}")
    with_var = settings.get("hint", "read as `{var}`")
    without_var = settings.get("hint_without_var", "read here")
    hints: dict[str, tuple[str, str]] = {}
    for text in texts:
        if not text:
            continue
        for match in pattern.finditer(text):
            groups = match.groupdict()
            fields = {"var": groups.get("var"), "type": (groups.get("type") or "").strip()}
            hint = (with_var if fields["var"] else without_var).format(**fields)
            start = max(0, match.start() - HINT_CONTEXT_CHARS)
            end = min(len(text), match.end() + HINT_CONTEXT_CHARS)
            hints[groups["key"].lower()] = (hint, text[start:end].strip())
    for c in candidates:
        found = hints.get(key_format.format(section=c.section, key=c.key).lower())
        if found:
            c.cpp_hint, c.code_context = found


# ---------------------------------------------------------------------------
# files the metrics come from
# ---------------------------------------------------------------------------

#: Folders never searched for metric scripts.
SKIP_PARTS = {"polyMesh", "triSurface", "extendedFeatureEdgeMesh", "processor0", ".benchmantic"}


def metric_files(module_dir: Path, repo_root: Path, settings: dict[str, Any]) -> list[Path]:
    """`metrics.files` patterns, in the implementation folder (and its
    sub-folders with `subfolders: true`, and the folder above it with
    `parent_folder: true`), keeping only files that mention `must_mention`."""
    patterns = settings.get("files") or []
    roots = [module_dir]
    if (settings.get("parent_folder") and module_dir.parent != module_dir
            and repo_root in (module_dir.parent, *module_dir.parent.parents)):
        roots.append(module_dir.parent)
    files: list[Path] = []
    for r in roots:
        walk = r.rglob("*") if r == module_dir and settings.get("subfolders") else r.glob("*")
        for p in walk:
            if (p.is_file() and any(fnmatch.fnmatchcase(p.name, pat) for pat in patterns)
                    and not (SKIP_PARTS & set(p.parts)) and p not in files):
                files.append(p)
    must = settings.get("must_mention")
    return [p for p in files if must in read_text(p)] if must else files
