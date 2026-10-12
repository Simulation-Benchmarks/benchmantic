# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
software.dumux.sources

DuMux / DUNE source layout: locating the module (main.cc + problem.hh) in a
larger checkout, the C++ problem class name, the Doxygen benchmark
description, the executable target from CMakeLists.txt, case discovery (one
params.input per case), and the metric keys main.cc writes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from metadata.metrics import discover_metrics
from metadata.repository import find_repo_file
from utils import camel_to_label, read_text


# REUSE-IgnoreEnd
CLASS_NAME_PATTERN = re.compile(r"\bclass\s+([A-Z]\w+)\s*(?::\s*public|\{)")


def extract_class_label(*source_texts: str) -> str | None:
    for text in source_texts:
        if not text:
            continue
        m = CLASS_NAME_PATTERN.search(text)
        if m:
            return camel_to_label(m.group(1))
    return None


DOXYGEN_BLOCK_PATTERN = re.compile(r"/\*!(.*?)\*/", re.DOTALL)


def extract_benchmark_description(*source_texts: str) -> str:
    """Return the longest Doxygen-style /*! ... */ comment block found
    across the given source texts, with the leading '*' decoration and any
    leading \\brief tag stripped. Returns "" if none is found.
    """
    best = ""
    for text in source_texts:
        if not text:
            continue
        for block in DOXYGEN_BLOCK_PATTERN.findall(text):
            cleaned = re.sub(r"^[ \t]*\*[ \t]?", "", block, flags=re.MULTILINE).strip()
            cleaned = re.sub(r"^\\brief\s*", "", cleaned)
            if len(cleaned) > len(best):
                best = cleaned
    return best


def find_module_dir(root: Path, main_cc_override: Path | None) -> Path:
    """Resolve the directory that actually holds the benchmark's main.cc +
    problem.hh, so the `module_dir` CLI argument can point at a top-level
    repo checkout (e.g. .../dumux_test/rotating-cylinders) instead of
    requiring the exact leaf directory
    (.../test/freeflow/navierstokes/rotatingcylinders).

    Search order:
      1. --main-cc was given explicitly -> use its parent directory.
      2. `root` itself already contains both main.cc and problem.hh -> use it.
      3. Otherwise, recursively search under `root` for a main.cc that has a
         problem.hh sitting next to it (the pairing is what distinguishes
         "the benchmark module" from an unrelated main.cc elsewhere in a
         larger checkout, e.g. other DuMux examples/tests).
    """
    if main_cc_override is not None:
        if not main_cc_override.exists():
            sys.exit(f"--main-cc {main_cc_override} does not exist")
        return main_cc_override.parent

    if (root / "main.cc").exists() and (root / "problem.hh").exists():
        return root

    candidates = sorted({
        p.parent for p in root.rglob("main.cc")
        if (p.parent / "problem.hh").exists()
    })

    if not candidates:
        sys.exit(
            f"Error: could not find a benchmark module (a directory containing "
            f"both main.cc and problem.hh) anywhere under {root}.\n"
            "Pass --main-cc to point directly at the right main.cc, or double-check the path."
        )
    if len(candidates) > 1:
        listing = "\n  ".join(str(c) for c in candidates)
        sys.exit(
            f"Error: found multiple benchmark modules (main.cc + problem.hh pairs) under {root}:\n"
            f"  {listing}\n"
            "Pass --main-cc to pick one, or point module_dir directly at the one you want."
        )
    return candidates[0]


#: Matches CMake's literal add_executable(<target> <source1> <source2> ...).
#: Less common in DuMux/DUNE test suites, which normally go through the
#: dumux_add_test/dune_add_test macro below -- checked as a fallback.
ADD_EXECUTABLE_PATTERN = re.compile(r"add_executable\s*\(\s*([\w.\-]+)\s+([^)]*)\)", re.IGNORECASE | re.DOTALL)


#: Matches DuMux/DUNE's dumux_add_test(...)/dune_add_test(...) CMake macros
#: -- the standard way DuMux/DUNE test executables are actually declared,
#: e.g.:
#:   dumux_add_test(NAME test_ff_navierstokes_rotatingcylinders
#:                   SOURCES main.cc
#:                   COMPILE_DEFINITIONS ...)
#: These take CMake keyword arguments (NAME/SOURCES/...) in any order, so
#: the whole parenthesized block is captured and searched rather than
#: assuming a fixed argument order.
ADD_TEST_MACRO_PATTERN = re.compile(r"(?:dumux|dune)_add_test\s*\((.*?)\)", re.IGNORECASE | re.DOTALL)


NAME_ARG_PATTERN = re.compile(r"\bNAME\s+([\w.\-]+)", re.IGNORECASE)


def extract_executable_name(cmakelists_text: str) -> str | None:
    """Scan one CMakeLists.txt's text for the compiled executable target
    built from main.cc -- this is the exact binary name a Snakemake
    workflow needs to invoke, and it's not always guessable from the
    directory name alone (e.g. DuMux abbreviates 'freeflow' to 'ff':
    test/freeflow/navierstokes/rotatingcylinders builds
    test_ff_navierstokes_rotatingcylinders, not
    test_freeflow_navierstokes_rotatingcylinders).

    Tries dumux_add_test/dune_add_test (the actual DuMux/DUNE convention)
    first, then a literal add_executable(...) as a fallback. Within each,
    prefers a call block that explicitly references main.cc; if none does
    (e.g. the source list is a CMake variable rather than a literal
    filename), falls back to the first target name found of that kind.
    Returns None if neither pattern matches at all.
    """
    if not cmakelists_text:
        return None

    first_test_macro_name = None
    for m in ADD_TEST_MACRO_PATTERN.finditer(cmakelists_text):
        block = m.group(1)
        name_match = NAME_ARG_PATTERN.search(block)
        if not name_match:
            continue
        if first_test_macro_name is None:
            first_test_macro_name = name_match.group(1)
        if re.search(r"\bmain\.cc\b", block, re.IGNORECASE):
            return name_match.group(1)
    if first_test_macro_name:
        return first_test_macro_name

    first_executable_target = None
    for m in ADD_EXECUTABLE_PATTERN.finditer(cmakelists_text):
        target, sources = m.group(1), m.group(2)
        if first_executable_target is None:
            first_executable_target = target
        if re.search(r"\bmain\.cc\b", sources, re.IGNORECASE):
            return target
    return first_executable_target


def find_executable_name(module_dir: Path, repo_root: Path) -> tuple[str | None, Path | None]:
    """Find the compiled executable's name by scanning CMakeLists.txt
    file(s), trying (in order):
      1. Right next to main.cc/problem.hh, then walking up toward
         repo_root (the normal DuMux/DUNE layout -- see find_repo_file()).
      2. If that file doesn't exist, or exists but extract_executable_name()
         doesn't find a match in it (e.g. an unrecognized macro), fall back
         to traversing every CMakeLists.txt anywhere under repo_root and
         using the first one that does match.
    Returns (executable_name, path_it_was_found_in), both None if nothing
    in the whole repo matches.
    """
    direct = find_repo_file(("CMakeLists.txt",), module_dir, repo_root)
    if direct:
        name = extract_executable_name(read_text(direct))
        if name:
            return name, direct

    for candidate in sorted(repo_root.rglob("CMakeLists.txt")):
        if candidate == direct:
            continue
        name = extract_executable_name(read_text(candidate))
        if name:
            return name, candidate

    return None, None


def case_id_for(params_path: Path, root: Path) -> str:
    rel_dir = params_path.parent.relative_to(root)
    if str(rel_dir) == ".":
        return params_path.parent.name
    return "_".join(rel_dir.parts)


def discover_cases(root: Path) -> list[tuple[Path, str]]:
    found = sorted(root.rglob("params.input"))
    if not found:
        sys.exit(f"No params.input files found anywhere under {root}")

    cases: list[tuple[Path, str]] = []
    seen_ids: dict[str, Path] = {}
    for params_path in found:
        case_dir = params_path.parent
        case_id = case_id_for(params_path, root)
        if case_id in seen_ids:
            sys.exit(
                f"Duplicate case id '{case_id}' derived from two different directories:\n"
                f"  {seen_ids[case_id]}\n  {case_dir}"
            )
        seen_ids[case_id] = case_dir
        cases.append((case_dir, case_id))
    return cases


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
