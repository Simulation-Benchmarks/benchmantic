"""
snakefile.generator

Derives everything the generated Snakefile needs from a benchmark.jsonld
(case-varying parameters, executable/build-dir, software name) and
provides the CLI (build_arg_parser/generate) that describe_benchmark.py calls
directly.
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
from snakefile.renderer import DEFAULT_UNIT_SYMBOLS, render_parameters_json, render_snakefile

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


def repo_name_from_url(repo_url: str | None) -> str | None:
    """'https://.../benchmarks/rotating-cylinders.git' -> 'rotating-cylinders'."""
    if not repo_url:
        return None
    name = repo_url.rstrip("/").rsplit("/", 1)[-1]
    return name[:-4] if name.endswith(".git") else name


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


def derive_executable(build_hints: dict[str, Any]) -> str | None:
    return build_hints.get("executable_name")


def derive_build_dir(build_hints: dict[str, Any], by_id: dict[str, dict[str, Any]], container_shared_dir: str) -> str | None:
    """{container_shared_dir's parent}/{repo name}/build-cmake/{module path}
    -- matches the convention seen in the real rotating-cylinders Snakefile
    (container_shared_dir=/dumux/shared, build dir under /dumux/<repo>/...).
    Returns None if either the repo URL (a real RO-Crate field, read from
    metadata.jsonld's root entity) or the module's relative path (from the
    build_hints.json sidecar) wasn't recoverable when metadata.jsonld was
    generated.
    """
    module_relative_path = build_hints.get("module_relative_path")
    root = by_id.get("./") or {}
    repo_name = repo_name_from_url(root.get("codeRepository"))
    if not module_relative_path or not repo_name:
        return None
    container_root = str(Path(container_shared_dir).parent)
    return f"{container_root}/{repo_name}/build-cmake/{module_relative_path}"




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
    ap.add_argument("--container-shared-dir", default="/dumux/shared",
                     help="Mount point inside the container used to save results back to the host (default: /dumux/shared). "
                          "Also used to derive --build-dir's container-root prefix when --build-dir isn't given explicitly.")
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
    ap.add_argument("--name-flag", default="Problem.Name",
                     help="Which 'Section.Key' (if present among the case-varying parameters) is the run's output-path/name "
                          "flag, so its value gets the {container_shared_dir}/{results_subdir}/{configuration}/... prefix "
                          "instead of being passed through as-is (default: Problem.Name).")
    ap.add_argument("--zip-name-flag", default=None,
                     help="Name the output zip '{value of this Section.Key}.zip' instead of the fixed 'results.zip' "
                          "(e.g. --zip-name-flag Problem.Name). Off by default -- must be one of the case-varying "
                          "parameters, or this has no effect.")
    ap.add_argument("--exclude-flag", action="append", default=None,
                     help="A 'Section.Key' to drop entirely from the CLI flags passed to the executable (repeatable), "
                          "e.g. for parameters that are compile-time-only and shouldn't be passed at runtime.")
    ap.add_argument("--mesh-split", action="store_true",
                     help="Opt in to the rotating-cylinders benchmark's specific radial-mesh-splitting math: split a "
                          "cells flag in half, mirror a grading flag's sign, and combine inner/outer radius with "
                          "their midpoint into one flag. This is ONE benchmark's specific meshing convention, not "
                          "general-purpose -- leave off (default) for any other benchmark.")
    ap.add_argument("--radial-cells-flag", default="Grid.Cells0",
                     help="[--mesh-split] Section.Key holding the total radial cell count to split in half (default: Grid.Cells0).")
    ap.add_argument("--angular-cells-flag", default="Grid.Cells1",
                     help="[--mesh-split] Section.Key holding the angular/azimuthal cell count, passed through unsplit (default: Grid.Cells1).")
    ap.add_argument("--grading-flag", default="Grid.Grading0",
                     help="[--mesh-split] Section.Key holding the radial grading factor, mirrored with a sign flip for "
                          "the second mesh block (default: Grid.Grading0).")
    ap.add_argument("--inner-radius-flag", default="Grid.Radial0",
                     help="[--mesh-split] Section.Key holding the inner radius (r1) -- must be one of the case-varying "
                          "parameters (default: Grid.Radial0). If this parameter's value is a list (captured via "
                          "generate_metadata.py's --full-value-params, e.g. 'Radial0 = 1.0 1.5 2.0'), it's used "
                          "directly and --outer-radius/--outer-radius-flag are not needed.")
    ap.add_argument("--outer-radius-flag", default=None,
                     help="[--mesh-split] Section.Key holding the outer radius (r2), if it's itself case-varying and "
                          "--inner-radius-flag is a plain scalar (not a --full-value-params list). Mutually exclusive "
                          "with --outer-radius.")
    ap.add_argument("--outer-radius", type=float, default=None,
                     help="[--mesh-split] Fixed outer radius (r2) value, if it's constant across all cases and "
                          "--inner-radius-flag is a plain scalar (not a --full-value-params list). Mutually "
                          "exclusive with --outer-radius-flag.")
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
    return ap


def generate(args: argparse.Namespace) -> None:

    if not args.metadata_jsonld.exists():
        sys.exit(f"Error: {args.metadata_jsonld} does not exist")

    if args.mesh_split and args.outer_radius_flag and args.outer_radius is not None:
        sys.exit("Error: pass at most one of --outer-radius-flag / --outer-radius, not both.")

    by_id = load_graph(args.metadata_jsonld)
    build_hints = load_build_hints(args.metadata_jsonld)

    software_name = args.software_name or derive_software_name(by_id)
    if args.output_dir is None:
        args.output_dir = Path("outputs") / software_name

    cases = extract_case_parameters(by_id)
    if not cases:
        sys.exit(
            f"Error: no m4i:ParameterSet nodes found in {args.metadata_jsonld} -- "
            "nothing to build case-varying CLI flags from."
        )
    # Any one case's values are enough to know each flag's *shape* (list vs
    # scalar) -- that's determined once, at generate_metadata.py time, by
    # whether --full-value-params was used for it, not per-case.
    sample_case_values = next(iter(cases.values()))
    units = extract_parameter_units(by_id)

    unit_symbols = dict(DEFAULT_UNIT_SYMBOLS)
    for entry in args.unit_symbol or []:
        if "=" not in entry:
            sys.exit(f"Error: --unit-symbol expects 'QUDT_UNIT=symbol', got {entry!r}")
        qudt_unit, symbol = entry.split("=", 1)
        unit_symbols[qudt_unit.strip()] = symbol.strip()

    executable = args.executable or derive_executable(build_hints)
    if not executable:
        hints_path = args.metadata_jsonld.with_name(args.metadata_jsonld.stem + ".build_hints.json")
        sys.exit(
            f"Error: no executable name found in {hints_path} (generate_metadata.py couldn't "
            "find/parse a CMakeLists.txt declaring it -- or that sidecar file doesn't exist at "
            "all, e.g. this metadata.jsonld predates that feature) and none was given via "
            "--executable."
        )

    build_dir = args.build_dir or derive_build_dir(build_hints, by_id, args.container_shared_dir)
    if not build_dir:
        sys.exit(
            "Error: could not derive a build directory (needs both a codeRepository URL on "
            "metadata.jsonld's root entity and module_relative_path in the build_hints.json "
            "sidecar) and none was given via --build-dir."
        )

    # Union of "Section.Key" flags across all cases, order-preserving. In
    # practice every case shares the same key set (same scenario-specific
    # selection was used to build the whole graph); union just guards
    # against a case that happens to be missing one.
    flag_keys: list[str] = []
    seen = set()
    for values in cases.values():
        for k in values:
            if k not in seen:
                seen.add(k)
                flag_keys.append(k)

    if args.mesh_split and args.inner_radius_flag not in flag_keys:
        sys.exit(
            f"Error: --mesh-split needs --inner-radius-flag ({args.inner_radius_flag!r}) to be one of the "
            f"case-varying parameters, but it's not in: {', '.join(flag_keys)}"
        )
    if args.mesh_split and args.outer_radius_flag and args.outer_radius_flag not in flag_keys:
        sys.exit(
            f"Error: --outer-radius-flag ({args.outer_radius_flag!r}) is not one of the case-varying "
            f"parameters: {', '.join(flag_keys)}. Use --outer-radius for a fixed value instead."
        )
    def _is_multi_value(v: Any) -> bool:
        return (isinstance(v, str) and len(v.split()) > 1) or isinstance(v, list)

    if args.mesh_split and not _is_multi_value(sample_case_values.get(args.inner_radius_flag)):
        # Scalar radius value -- r2 has to come from somewhere else.
        if not args.outer_radius_flag and args.outer_radius is None:
            sys.exit(
                f"Error: --inner-radius-flag ({args.inner_radius_flag!r}) is a plain scalar, not a "
                "--full-value-params multi-token value, so an outer radius can't be derived "
                "automatically. Pass --outer-radius-flag or --outer-radius, or use "
                f"--full-value-params in generate_metadata.py to capture "
                f"{args.inner_radius_flag}'s full multi-token value instead."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snakefile_path = args.output_dir / "Snakefile"
    snakefile_path.write_text(
        render_snakefile(flag_keys, executable, build_dir, args, sample_case_values, units, unit_symbols),
        encoding="utf-8",
    )
    print(f"Wrote {snakefile_path}")

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

    print(
        f"\n{len(cases)} case(s), {len(flag_keys)} case-varying flag(s): {', '.join(flag_keys)}\n"
        f"executable = {executable}\n"
        f"build_dir  = {build_dir}"
        + ("  (derived from metadata.jsonld)" if not args.build_dir else "  (from --build-dir)")
        + (f"\nmesh-split: ON (radial={args.radial_cells_flag}, angular={args.angular_cells_flag}, "
           f"grading={args.grading_flag}, r1={args.inner_radius_flag}, "
           f"r2={args.outer_radius_flag or args.outer_radius})" if args.mesh_split else "\nmesh-split: off")
    )




def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    generate(args)


if __name__ == "__main__":
    main()
