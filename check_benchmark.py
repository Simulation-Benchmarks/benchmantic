#!/usr/bin/env python3
"""
check_benchmark.py

Checks a metadata.builder-produced benchmark.jsonld against what
semantic_benchmark.BenchmarkLoader and rocrate/create.py actually need to
parse it correctly -- e.g. that every m4i:ParameterSet has an identifier
and at least one part with a real value, that at least one m4i:ProcessingStep
exists, and that metric field mappings resolve.

Deliberately uses the REAL `semantic_benchmark` package to parse the file
(specifically its BenchmarkLoader), rather than reimplementing its logic --
so this validator can never drift out of sync with what the actual
consumer expects. If `semantic_benchmark` changes upstream, re-running this
script picks that up automatically; only the human-readable check
descriptions below might need updating.

No `pip install` needed -- BenchmarkLoader only needs `rdflib` (a normal
wheel, no build-system issues) and pure-Python source, so a plain clone is
enough:

    git clone https://github.com/Simulation-Benchmarks/semantic-benchmark.git
    pip install rdflib   # only real dependency BenchmarkLoader needs
    python3 check_benchmark.py benchmark.jsonld

This script auto-detects a sibling/child ./semantic-benchmark checkout (or
pass --semantic-benchmark-src explicitly, or set the
SEMANTIC_BENCHMARK_SRC environment variable) and adds its src/ directory
to sys.path directly -- no packaging/build step involved. Falls back to a
normal `import semantic_benchmark` if it's pip-installed instead.

Usage
-----
    python3 check_benchmark.py benchmark.jsonld
    python3 check_benchmark.py benchmark.jsonld --semantic-benchmark-src /path/to/semantic-benchmark
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _fail(msg: str) -> None:
    print(f"FAIL  {msg}")


def _ok(msg: str) -> None:
    print(f"OK    {msg}")


def _warn(msg: str) -> None:
    print(f"WARN  {msg}")


def _candidate_src_dirs() -> list[Path]:
    """Places a plain `git clone` of semantic-benchmark might sit, checked
    in order (as repo roots -- _resolve_src_dir() handles finding src/
    underneath, or using the path directly if it's already a src/ dir)."""
    candidates = []
    env = os.environ.get("SEMANTIC_BENCHMARK_SRC")
    if env:
        candidates.append(Path(env))
    for base in (Path.cwd(), SCRIPT_DIR, SCRIPT_DIR.parent):
        candidates.append(base / "semantic-benchmark")
        candidates.append(base / "semantic_benchmark")
    return candidates


def _resolve_src_dir(path: Path) -> Path | None:
    """A given directory might be the repo root (containing src/semantic_benchmark/)
    or already the src/ directory itself (containing semantic_benchmark/ directly).
    Returns the actual src/ directory to add to sys.path, or None if neither shape matches.
    """
    if (path / "semantic_benchmark").is_dir():
        return path
    if (path / "src" / "semantic_benchmark").is_dir():
        return path / "src"
    return None


def _import_benchmark_loader(explicit_src: Path | None):
    """Locate and import semantic_benchmark.semantics without requiring
    `pip install` -- tries an explicit --semantic-benchmark-src path, then
    common clone locations, adding src/ to sys.path directly; falls back to
    a normal pip-installed import if none of those have it either.
    """
    search_paths = [explicit_src] if explicit_src else _candidate_src_dirs()
    for candidate in search_paths:
        if not candidate:
            continue
        src_dir = _resolve_src_dir(candidate)
        if src_dir is None:
            continue
        sys.path.insert(0, str(src_dir))
        try:
            module = importlib.import_module("semantic_benchmark.semantics")
            return module, src_dir
        except ImportError:
            sys.path.pop(0)
            continue

    try:
        module = importlib.import_module("semantic_benchmark.semantics")
        return module, None  # already pip-installed
    except ImportError:
        pass

    sys.exit(
        "Error: couldn't find or import semantic_benchmark.\n\n"
        "No pip install needed -- it's pure Python, just clone it and point rdflib at it:\n"
        "  git clone https://github.com/Simulation-Benchmarks/semantic-benchmark.git\n"
        "  pip install rdflib\n\n"
        "Then either run this script from the same directory as that clone (auto-detected), "
        "or pass --semantic-benchmark-src /path/to/semantic-benchmark explicitly."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmark_jsonld", type=Path)
    ap.add_argument("--semantic-benchmark-src", type=Path, default=None,
                     help="Path to a semantic-benchmark git clone (or its src/ directory directly). "
                          "Default: auto-detect a sibling/child ./semantic-benchmark checkout, or the "
                          "SEMANTIC_BENCHMARK_SRC environment variable.")
    return ap


def run(args: argparse.Namespace) -> int:
    """Run all checks. Returns 0 if everything passed, 1 if any check
    failed -- doesn't call sys.exit() itself, so callers (e.g. workflow.py)
    can decide what to do with the result.
    """
    if not args.benchmark_jsonld.exists():
        sys.exit(f"Error: {args.benchmark_jsonld} does not exist")

    semantics_module, used_src = _import_benchmark_loader(args.semantic_benchmark_src)
    BenchmarkLoader = semantics_module.BenchmarkLoader
    TextParameter = semantics_module.TextParameter
    NumericalVariable = semantics_module.NumericalVariable
    if used_src:
        print(f"(using semantic_benchmark from {used_src}, not pip-installed)\n")

    failures = 0
    warnings = 0

    # --- 1. Load ---
    try:
        loader = BenchmarkLoader(str(args.benchmark_jsonld))
        benchmark = loader.load()
    except Exception as exc:  # noqa: BLE001 -- want to report any parse failure, not just ValueError
        _fail(f"BenchmarkLoader failed to load {args.benchmark_jsonld}: {exc}")
        sys.exit(1)
    _ok(f"BenchmarkLoader parsed {args.benchmark_jsonld} (label: {benchmark.label!r})")

    # --- 2. Processing steps (create_rocrate.py hard-requires >=1) ---
    if not benchmark.processing_steps:
        _fail(
            "benchmark.processing_steps is empty -- create_rocrate.py's "
            '_add_configuration_nodes() will raise ValueError("Benchmark has no processing steps.")'
        )
        failures += 1
    else:
        _ok(f"{len(benchmark.processing_steps)} processing step(s) found")
        for step in benchmark.processing_steps:
            if not step.configurations:
                _fail(f"processing step {step.label!r} has no configurations (missing 'has configuration'/usesConfiguration)")
                failures += 1
            else:
                _ok(f"  step {step.label!r} -> {len(step.configurations)} configuration(s)")

    # --- 3. Parameter sets: identifier + parts + values ---
    if not benchmark.parameter_sets:
        _warn("benchmark.parameter_sets is empty -- no case-varying parameters at all")
        warnings += 1
    for pset in benchmark.parameter_sets:
        if not pset.identifier:
            _fail(
                f"ParameterSet {pset.label!r} has no identifier (check 'identifier' maps to m4i:identifier, "
                "not schema:identifier) -- run_benchmark.py will silently skip this configuration entirely"
            )
            failures += 1
        else:
            _ok(f"ParameterSet identifier={pset.identifier!r}")

        if not pset.parts:
            _fail(
                f"ParameterSet {pset.identifier or pset.label!r} has no parts (check 'has part' maps to "
                "obo:BFO_0000051, and that 'obo' is defined in @context)"
            )
            failures += 1
            continue

        for part in pset.parts:
            if isinstance(part, NumericalVariable):
                _fail(
                    f"  parameter {part.label!r} is typed m4i:NumericalVariable -- its value will "
                    "ALWAYS resolve to None downstream (see contract §4). Use a different @type "
                    "for parameter nodes (metric/evaluates nodes are the only place this type belongs)."
                )
                failures += 1
                continue
            value = part.string_value if isinstance(part, TextParameter) else getattr(part, "numerical_value", None)
            if value is None:
                _warn(f"  parameter {part.label!r} ({type(part).__name__}) has no value set")
                warnings += 1
            else:
                _ok(f"  parameter {part.label!r} = {value!r} ({type(part).__name__}, unit={part.unit!r})")

    # --- 4. Metrics: field mapping needed to read values post-run ---
    if not benchmark.evaluates:
        _warn("benchmark.evaluates is empty -- no metrics defined")
        warnings += 1
    for metric in benchmark.evaluates:
        fm = metric.field_mapping
        if fm is None:
            _fail(
                f"metric {metric.label!r} has no field_mapping (needs a cr:Field node whose "
                "sio:SIO_000210 'represents' points at it -- check 'sio' is defined in @context, "
                "and that cr:source -> cr:extract -> cr:jsonPath / cr:FileObject are all wired up)"
            )
            failures += 1
            continue
        missing = [f for f in ("json_path", "file_object_label") if not getattr(fm, f)]
        if missing:
            _fail(f"metric {metric.label!r}'s field_mapping is missing: {', '.join(missing)}")
            failures += 1
        else:
            _ok(f"metric {metric.label!r} -> {fm.file_object_label}{fm.json_path}")

    print()
    if failures:
        print(f"{failures} failure(s), {warnings} warning(s) -- see the check messages above for what each one means.")
        return 1
    else:
        print(f"All checks passed ({warnings} warning(s)).")
        return 0


def main(argv: list[str] | None = None) -> None:
    sys.exit(run(build_arg_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
