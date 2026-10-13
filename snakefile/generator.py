# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
snakefile.generator

Reads everything the generated Snakefile needs from a benchmark.jsonld
(case-varying parameters and their units, the build_hints sidecar, the input
mapping), asks the benchmark's software adapter (software/<name>/) for the
Snakefile text, and writes it with the per-case parameters.json previews.
Provides the CLI (build_arg_parser/generate) that describe_benchmark.py
calls directly; each software adds its own options to it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

from metadata.graph import find_all, load_graph
from metadata.graph import node_id as _id
from metadata.mapping import load_mapping, mapping_path_for
from snakefile.renderer import DEFAULT_UNIT_SYMBOLS, render_parameters_json
from snakefile import styles as snakefile_styles
from software import SnakefileContext

def extract_case_parameters(by_id: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """{case_id: {"Section.Key": value, ...}} for every m4i:ParameterSet in
    the graph. This is exactly the set of parameters generate_metadata.py's
    --scenario-params (or its default/interactive selection) treated as
    case-varying -- nothing more, nothing less.

    Reads "has numerical value" for numeric scalars, falling back to "has
    string value" for text parameters and full_value (multi-token, e.g.
    'Radial0 = 1.0 1.5 2.0') parameters -- the latter are stored as a single
    space-joined string (see generate_metadata.py's add_parameter_variable()
    for why: RDF can't reliably round-trip an ordered list through a single
    graph.value() lookup, which is what semantic_benchmark.BenchmarkLoader
    uses).
    """
    cases: dict[str, dict[str, Any]] = {}
    for pset in find_all(by_id, "m4i:ParameterSet"):
        case_id = pset.get("identifier") or pset.get("label") or "case"
        values: dict[str, Any] = {}
        for part_ref in pset.get("has part", []):
            var = by_id.get(_id(part_ref))
            if not var:
                continue
            label = var.get("label")  # "Section.Key", set verbatim by GraphBuilder
            if label is not None:
                if "has numerical value" in var:
                    values[label] = var["has numerical value"]
                else:
                    values[label] = var.get("has string value")
        cases[case_id] = values
    return cases


def extract_parameter_units(by_id: dict[str, dict[str, Any]]) -> dict[str, str]:
    """{"Section.Key": qudt_unit_curie} for every parameter in every
    m4i:ParameterSet. Needed to mirror run_benchmark.py's
    parameter_json_key() unit-suffix convention (see
    UNIT_SYMBOLS/config_key() below) -- that script writes the ACTUAL
    parameters.json this Snakefile reads at runtime, and it names some
    keys with a bracketed unit suffix (e.g. "Grid.Radial0[m]" for unit:M),
    not the bare Section.Key.

    Walks ParameterSet "has part" links directly (like
    extract_case_parameters()) rather than searching by @type -- parameter
    nodes are deliberately NOT typed "numerical variable" (see
    generate_metadata.py's add_parameter_variable()), since
    semantic_benchmark.BenchmarkLoader treats that type as value-less.
    """
    units: dict[str, str] = {}
    for pset in find_all(by_id, "m4i:ParameterSet"):
        for part_ref in pset.get("has part", []):
            var = by_id.get(_id(part_ref))
            if not var:
                continue
            label = var.get("label")
            unit = _id(var.get("has unit"))
            if label and unit and label not in units:
                units[label] = unit
    return units


def derive_software_name(by_id: dict[str, dict[str, Any]]) -> str:
    """Slugify local:software's label ('DuMux' -> 'dumux') for use in the
    default 'outputs/<software_name>' output directory -- mirrors the same
    convention used elsewhere in this pipeline (e.g. run_benchmark.py's
    TOOL_NAME/--software-name), without depending on that script.
    """
    software = by_id.get("local:software") or {}
    label = software.get("label") or "benchmark"
    slug = re.sub(r"[^0-9a-zA-Z]+", "-", label).strip("-").lower()
    return slug or "benchmark"


def _adapter_from_hints(build_hints: dict[str, Any]):
    """The software config the description was made with, recorded in the
    build_hints sidecar (see metadata.builder._software_spec_hint)."""
    spec = build_hints.get("software_spec")
    if not spec:
        return None
    from software.generic import GenericAdapter

    return GenericAdapter.from_spec(spec)()


def load_build_hints(metadata_jsonld_path: Path) -> dict[str, Any]:
    """Load the non-RO-Crate sidecar file generate_metadata.py writes next
    to metadata.jsonld (e.g. 'metadata.build_hints.json' for
    'metadata.jsonld') -- contains executable_name/module_relative_path,
    deliberately kept out of the RO-Crate graph itself. Returns {} if the
    sidecar doesn't exist (e.g. metadata.jsonld predates this feature, or
    the CMakeLists.txt scan found nothing).
    """
    hints_path = metadata_jsonld_path.with_name(metadata_jsonld_path.stem + ".build_hints.json")
    if not hints_path.exists():
        return {}
    return json.loads(hints_path.read_text(encoding="utf-8"))


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metadata_jsonld", type=Path, nargs="?", default=Path("metadata.jsonld"),
                     help="Path to a generate_metadata.py output (default: ./metadata.jsonld).")
    ap.add_argument("--container-image", required=True,
                     help="Container image reference, e.g. git.iws.uni-stuttgart.de:4567/benchmarks/rotating-cylinders:3.1 "
                          "-- can't be recovered from metadata.jsonld, always required.")
    ap.add_argument("--container-shared-dir", default=None,
                     help="Mount point inside the container used to save results back to the host. "
                          "Default: the software's own default, if it has one (DuMux: /dumux/shared).")
    ap.add_argument("--executable", default=None,
                     help="Override the executable name instead of using the one in the build_hints.json sidecar.")
    ap.add_argument("--build-dir", default=None,
                     help="Override the in-container build directory instead of deriving it from "
                          "--container-shared-dir + the repo name (codeRepository) + the module's relative path "
                          "(build_hints.json sidecar).")
    ap.add_argument("--results-subdir", default="results",
                     help="Subdirectory (under --container-shared-dir) that per-case results are written into (default: results).")
    ap.add_argument("--unit-symbol", action="append", default=None, metavar="QUDT_UNIT=SYMBOL",
                     help="Override/extend the unit-symbol mapping used to build parameters.json keys the same way "
                          "run_benchmark.py's parameter_json_key() does (e.g. 'unit:M=m' makes a parameter with "
                          "unit:M read as config['Section.Key[m]'] instead of config['Section.Key']). Repeatable. "
                          f"Defaults: {DEFAULT_UNIT_SYMBOLS}. Keep this in sync with run_benchmark.py's own "
                          "UNIT_SYMBOLS -- a mismatch here causes a KeyError at Snakemake runtime, not at "
                          "generation time.")
    ap.add_argument("--software-name", default=None,
                     help="Software name used to build the default --output-dir ('outputs/<software-name>'). "
                          "Defaults to metadata.jsonld's local:software label, slugified (e.g. 'DuMux' -> 'dumux') "
                          "-- matching the 'outputs/<software_name>' convention used elsewhere in this pipeline.")
    ap.add_argument("--output-dir", type=Path, default=None,
                     help="Directory to write the Snakefile and per-case parameters.json files into. "
                          "Default: outputs/<software-name>/.")
    ap.add_argument("--zip", type=Path, default=None, metavar="PATH",
                     help="Also package just the Snakefile (flat, no subfolders) into a zip at PATH, ready to "
                          "hand directly to run_benchmark.py's --benchmark-zip. run_benchmark.py generates its "
                          "own parameters.json per configuration from the benchmark file at runtime, so only "
                          "the Snakefile itself needs to be in this archive.")
    snakefile_styles.add_arguments(ap)
    return ap


def generate(args: argparse.Namespace) -> None:

    if not args.metadata_jsonld.exists():
        sys.exit(f"Error: {args.metadata_jsonld} does not exist")

    by_id = load_graph(args.metadata_jsonld)
    build_hints = load_build_hints(args.metadata_jsonld)
    mapping_file = mapping_path_for(args.metadata_jsonld)
    mapping = load_mapping(mapping_file) if mapping_file else None

    software_name = args.software_name or derive_software_name(by_id)
    adapter = _adapter_from_hints(build_hints)
    if adapter is None:
        sys.exit(f"Error: {args.metadata_jsonld.stem}.build_hints.json next to {args.metadata_jsonld} has no "
                 "software description -- make the benchmark description again with this version of benchmantic.")
    if args.container_shared_dir is None:
        args.container_shared_dir = adapter.default_container_shared_dir
    if args.output_dir is None:
        args.output_dir = Path("outputs") / software_name

    cases = extract_case_parameters(by_id)
    if not cases:
        sys.exit(
            f"Error: no m4i:ParameterSet nodes found in {args.metadata_jsonld} -- "
            "nothing to build case-varying CLI flags from."
        )
    units = extract_parameter_units(by_id)

    unit_symbols = dict(DEFAULT_UNIT_SYMBOLS)
    for entry in args.unit_symbol or []:
        if "=" not in entry:
            sys.exit(f"Error: --unit-symbol expects 'QUDT_UNIT=symbol', got {entry!r}")
        qudt_unit, symbol = entry.split("=", 1)
        unit_symbols[qudt_unit.strip()] = symbol.strip()

    result = adapter.render_snakefile(SnakefileContext(
        args=args, metadata_jsonld=args.metadata_jsonld, by_id=by_id, build_hints=build_hints,
        cases=cases, units=units, unit_symbols=unit_symbols, mapping=mapping, mapping_file=mapping_file,
    ))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snakefile_path = args.output_dir / "Snakefile"
    snakefile_path.write_text(result.text, encoding="utf-8")
    print(f"Wrote {snakefile_path}{result.note}")

    if args.zip:
        args.zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(snakefile_path, arcname="Snakefile")
        print(f"Wrote {args.zip} (Snakefile only -- ready for run_benchmark.py's --benchmark-zip)")

    for case_id, values in cases.items():
        case_dir = args.output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        params_path = case_dir / "parameters.json"
        params_path.write_text(render_parameters_json(case_id, values, units, unit_symbols), encoding="utf-8")
        print(f"Wrote {params_path}")

    for line in result.summary:
        print(line)


def namespace_for(args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    """The Namespace generate() expects, taken from another parser's
    parsed options (describe_benchmark.py's) -- every option this module
    and the Snakefile styles define, so callers don't list them."""
    dests = [a.dest for a in build_arg_parser()._actions if a.dest != "help"]  # noqa: SLF001
    values = {d: getattr(args, d, None) for d in dests}
    values.update(overrides)
    return argparse.Namespace(**values)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    generate(args)


if __name__ == "__main__":
    main()
