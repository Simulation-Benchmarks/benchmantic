#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
verify_description.py

Validates a metadata.builder-produced benchmark.jsonld in two stages:

1. SHACL validation (the main gate) -- the file is checked against the
   SHACL shapes bundled with the `semantic_benchmark` package
   (BenchmarkLoader.load_shapes()), using pyshacl with the same settings
   the loader itself uses (RDFS inference). Every sh:Violation is printed,
   grouped by constraint, with the offending nodes listed. Any violation
   fails the run.

   Note: BenchmarkLoader also runs these shapes internally, but it never
   raises -- it only writes a `<file>.shacl.log` next to the input and
   keeps loading. That's why this script runs the validation itself and
   reports the result directly.

2. Loader checks -- the file is loaded with the real
   semantic_benchmark.BenchmarkLoader, and a few things the shapes don't
   express are checked: at least one m4i:ProcessingStep exists, every
   parameter carries a value (and isn't mis-typed m4i:NumericalVariable),
   and every metric has a resolvable field mapping. Skip with --shacl-only.

Install:

    pip install semantic-benchmark      # brings rdflib + pyshacl

A local git clone of semantic-benchmark still works too (auto-detected as a
sibling/child ./semantic-benchmark, or via --semantic-benchmark-src /
SEMANTIC_BENCHMARK_SRC), as long as it's recent enough to have
BenchmarkLoader.load_shapes().

Usage
-----
    python3 verify_description.py benchmark.jsonld
    python3 verify_description.py benchmark.jsonld --shacl-only
    python3 verify_description.py benchmark.jsonld --shapes my-shapes.ttl
    python3 verify_description.py benchmark.jsonld --report shacl-report.txt
    python3 verify_description.py --export-shapes benchmark-shapes.ttl
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

#: How many focus nodes to list per grouped SHACL violation before
#: summarising the rest as "... and N more".
MAX_NODES_PER_GROUP = 5


def _fail(msg: str) -> None:
    print(f"FAIL  {msg}")


def _ok(msg: str) -> None:
    print(f"OK    {msg}")


def _warn(msg: str) -> None:
    print(f"WARN  {msg}")


# ---------------------------------------------------------------------------
# Locating semantic_benchmark
# ---------------------------------------------------------------------------

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


def _import_semantics(explicit_src: Path | None):
    """Import semantic_benchmark.semantics -- from an explicit/auto-detected
    clone first, otherwise the pip-installed package. Exits with install
    instructions if neither works, or if the version found predates
    BenchmarkLoader.load_shapes().
    """
    module, used_src = None, None
    search_paths = [explicit_src] if explicit_src else _candidate_src_dirs()
    for candidate in search_paths:
        if not candidate:
            continue
        src_dir = _resolve_src_dir(candidate)
        if src_dir is None:
            continue
        sys.path.insert(0, str(src_dir))
        try:
            module, used_src = importlib.import_module("semantic_benchmark.semantics"), src_dir
            break
        except ImportError:
            sys.path.pop(0)

    if module is None:
        try:
            module = importlib.import_module("semantic_benchmark.semantics")
        except ImportError:
            sys.exit(
                "Error: couldn't import semantic_benchmark.\n\n"
                "Install it (this also installs rdflib and pyshacl):\n"
                "  pip install semantic-benchmark\n\n"
                "or point --semantic-benchmark-src at a local clone."
            )

    if not hasattr(module.BenchmarkLoader, "load_shapes"):
        where = used_src or "the pip-installed package"
        sys.exit(
            f"Error: semantic_benchmark from {where} has no BenchmarkLoader.load_shapes() -- "
            "it's too old to provide SHACL shapes.\n"
            "Update it:  pip install -U semantic-benchmark   (or `git pull` your clone)."
        )
    return module, used_src


# ---------------------------------------------------------------------------
# Stage 1: SHACL
# ---------------------------------------------------------------------------

def _load_shapes(semantics_module, shapes_path: Path | None):
    from rdflib import Graph

    if shapes_path is None:
        return semantics_module.BenchmarkLoader.load_shapes(), "semantic_benchmark (bundled)"
    if not shapes_path.exists():
        sys.exit(f"Error: shapes file {shapes_path} does not exist")
    return Graph().parse(str(shapes_path), format="turtle"), str(shapes_path)


def _short(term, nm) -> str:
    """Compact a node/path for display (CURIE where a prefix is known)."""
    from rdflib import BNode, URIRef

    if term is None:
        return "-"
    if isinstance(term, URIRef):
        try:
            return term.n3(nm)
        except Exception:  # noqa: BLE001
            return str(term)
    if isinstance(term, BNode):
        return "[blank node]"
    return str(term)


def _path_label(results, res, nm) -> str:
    """sh:resultPath, rendering sh:alternativePath lists as 'a | b'."""
    from rdflib import BNode
    from rdflib.collection import Collection
    from rdflib.namespace import SH

    path = results.value(res, SH.resultPath)
    if isinstance(path, BNode):
        alt = results.value(path, SH.alternativePath)
        if alt is not None:
            return " | ".join(_short(p, nm) for p in Collection(results, alt))
    return _short(path, nm)


def run_shacl(data_graph, shapes_graph, report_path: Path | None) -> tuple[int, int]:
    """Validate and print grouped results. Returns (violations, warnings)."""
    from pyshacl import validate
    from rdflib.namespace import SH

    conforms, results, report_text = validate(
        data_graph=data_graph,
        shacl_graph=shapes_graph,
        inference="rdfs",  # same as BenchmarkLoader._validate()
        abort_on_first=False,
        allow_infos=True,
        allow_warnings=True,
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(str(report_text), encoding="utf-8")

    nm = data_graph.namespace_manager
    # Group identical problems (same severity + constraint + path + message
    # template) so a missing property on 40 Field nodes is one entry, not 40.
    groups: dict[tuple, list[str]] = defaultdict(list)
    for res in results.subjects(SH.resultSeverity, None):
        severity = results.value(res, SH.resultSeverity)
        component = _short(results.value(res, SH.sourceConstraintComponent), nm).split("#")[-1]
        component = component.replace("sh:", "").replace("ConstraintComponent", "")
        path = _path_label(results, res, nm)
        focus = results.value(res, SH.focusNode)
        focus_s = _short(focus, nm)
        value = results.value(res, SH.value)
        message = str(results.value(res, SH.resultMessage) or "")
        # Messages embed the focus node; strip it so identical problems group.
        message = message.replace(str(focus), "<node>").replace(focus_s, "<node>")
        entry = focus_s if value is None else f"{focus_s} (value: {_short(value, nm)})"
        groups[(str(severity), component, path, message)].append(entry)

    violations = warnings = 0
    if conforms:
        _ok("SHACL: conforms to all shapes")
        return 0, 0

    for (severity, component, path, message), nodes in sorted(groups.items()):
        is_violation = severity == str(SH.Violation)
        emit = _fail if is_violation else _warn
        if is_violation:
            violations += len(nodes)
        else:
            warnings += len(nodes)
        emit(f"SHACL {component} on {path}: {message}  [{len(nodes)} node(s)]")
        for n in sorted(nodes)[:MAX_NODES_PER_GROUP]:
            print(f"        - {n}")
        if len(nodes) > MAX_NODES_PER_GROUP:
            print(f"        ... and {len(nodes) - MAX_NODES_PER_GROUP} more")
    return violations, warnings


# ---------------------------------------------------------------------------
# Stage 2: loader checks (things the shapes don't cover)
# ---------------------------------------------------------------------------

def run_loader_checks(semantics_module, path: Path) -> tuple[int, int]:
    BenchmarkLoader = semantics_module.BenchmarkLoader
    TextParameter = semantics_module.TextParameter
    NumericalVariable = semantics_module.NumericalVariable

    failures = warnings = 0
    try:
        benchmark = BenchmarkLoader(str(path)).load()
    except Exception as exc:  # noqa: BLE001 -- report any parse failure
        _fail(f"BenchmarkLoader failed to load {path}: {exc}")
        return 1, 0
    _ok(f"BenchmarkLoader parsed the file (label: {benchmark.label!r})")

    # Processing steps: create_rocrate.py hard-requires >= 1; the shapes don't.
    if not benchmark.processing_steps:
        _fail(
            "no m4i:ProcessingStep found -- create_rocrate.py's _add_configuration_nodes() "
            'will raise ValueError("Benchmark has no processing steps.")'
        )
        failures += 1
    else:
        _ok(f"{len(benchmark.processing_steps)} processing step(s) found")
        for step in benchmark.processing_steps:
            if not step.configurations:
                _fail(f"processing step {step.label!r} has no configurations (missing 'has configuration')")
                failures += 1

    # Parameter values: the shapes only require that 'has part' exists.
    if not benchmark.parameter_sets:
        _warn("no parameter sets -- no case-varying parameters at all")
        warnings += 1
    for pset in benchmark.parameter_sets:
        for part in pset.parts:
            if isinstance(part, NumericalVariable):
                _fail(
                    f"parameter {part.label!r} in {pset.identifier!r} is typed m4i:NumericalVariable -- "
                    "its value will always resolve to None downstream. That type is for metric "
                    "(evaluates) nodes only."
                )
                failures += 1
                continue
            value = part.string_value if isinstance(part, TextParameter) else getattr(part, "numerical_value", None)
            if value is None:
                _warn(f"parameter {part.label!r} in {pset.identifier!r} has no value set")
                warnings += 1
    if benchmark.parameter_sets and not failures:
        n = sum(len(p.parts) for p in benchmark.parameter_sets)
        _ok(f"{len(benchmark.parameter_sets)} parameter set(s), {n} parameter value(s) resolved")

    # Metric field mappings: the shapes validate Field nodes that exist, but
    # can't tell you a metric has no Field pointing at it at all.
    for metric in benchmark.evaluates:
        fm = metric.field_mapping
        if fm is None:
            _fail(
                f"metric {metric.label!r} has no field mapping (no cr:Field whose 'represents' "
                "points at it) -- its value can't be read back after a run"
            )
            failures += 1
            continue
        missing = [f for f in ("json_path", "file_object_label") if not getattr(fm, f)]
        if missing:
            _fail(f"metric {metric.label!r}'s field mapping is missing: {', '.join(missing)}")
            failures += 1
        else:
            _ok(f"metric {metric.label!r} -> {fm.file_object_label}{fm.json_path}")

    return failures, warnings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmark_jsonld", type=Path, nargs="?",
                    help="The benchmark JSON-LD file to verify (optional with --export-shapes).")
    ap.add_argument("--semantic-benchmark-src", type=Path, default=None,
                    help="Path to a semantic-benchmark git clone (or its src/ directory). Default: "
                         "auto-detect a sibling/child ./semantic-benchmark, the SEMANTIC_BENCHMARK_SRC "
                         "environment variable, or the pip-installed package.")
    ap.add_argument("--shapes", type=Path, default=None,
                    help="Validate against this Turtle shapes file instead of the shapes bundled "
                         "with semantic_benchmark (e.g. to try a draft of new shapes).")
    ap.add_argument("--shacl-only", action="store_true",
                    help="Run only the SHACL stage, skip the BenchmarkLoader checks.")
    ap.add_argument("--report", type=Path, default=None,
                    help="Also write pyshacl's full text report to this file.")
    ap.add_argument("--export-shapes", type=Path, default=None, metavar="PATH",
                    help="Write the shapes being used to PATH as Turtle (e.g. to include in an "
                         "LLM prompt). Without a benchmark file, exits after exporting.")
    return ap


def run(args: argparse.Namespace) -> int:
    """Run all checks. Returns 0 if everything passed, 1 otherwise -- doesn't
    call sys.exit() itself, so callers (e.g. workflow.py) decide what to do.
    """
    semantics_module, used_src = _import_semantics(args.semantic_benchmark_src)
    if used_src:
        print(f"(using semantic_benchmark from {used_src}, not the pip-installed package)")

    shapes_graph, shapes_source = _load_shapes(semantics_module, getattr(args, "shapes", None))

    export_path = getattr(args, "export_shapes", None)
    if export_path is not None:
        shapes_graph.serialize(destination=str(export_path), format="turtle")
        print(f"Wrote SHACL shapes ({shapes_source}) to {export_path}")
        if args.benchmark_jsonld is None:
            return 0

    if args.benchmark_jsonld is None:
        sys.exit("Error: a benchmark JSON-LD file is required (unless only using --export-shapes)")
    if not args.benchmark_jsonld.exists():
        sys.exit(f"Error: {args.benchmark_jsonld} does not exist")

    from rdflib import Graph

    try:
        data_graph = Graph().parse(str(args.benchmark_jsonld), format="json-ld")
    except Exception as exc:  # noqa: BLE001
        _fail(f"{args.benchmark_jsonld} is not valid JSON-LD: {exc}")
        return 1

    print(f"\n--- SHACL validation (shapes: {shapes_source}) ---")
    violations, warnings = run_shacl(data_graph, shapes_graph, getattr(args, "report", None))
    if getattr(args, "report", None):
        print(f"(full SHACL report written to {args.report})")

    failures = 0
    if not getattr(args, "shacl_only", False):
        print("\n--- BenchmarkLoader checks ---")
        failures, loader_warnings = run_loader_checks(semantics_module, args.benchmark_jsonld)
        warnings += loader_warnings

    print()
    if violations or failures:
        print(f"{violations} SHACL violation(s), {failures} loader failure(s), {warnings} warning(s).")
        return 1
    print(f"All checks passed ({warnings} warning(s)).")
    return 0


def main(argv: list[str] | None = None) -> None:
    sys.exit(run(build_arg_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
