# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
software.dumux.hooks

The DuMux extras no software.yaml setting covers -- the older flag-based
Snakefile with the rotating-cylinders --mesh-split math, used when the
description keeps DuMux's own input names (--parameter-names software,
the snakefile-only preset) or with --mesh-split. With an input mapping (the
default) the Snakefile comes from the shared command_line style
(snakefile/styles/command_line.py) and this file only passes it on.

Loaded by software.generic: add_snakefile_arguments() adds the options,
render_snakefile() replaces the style's render().
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Any

from snakefile.renderer import config_key
from software.base import SnakefileContext, SnakefileResult


def _flag_var_name(flag: str) -> str:
    """'Grid.Cells0' -> 'grid_cells0', a safe Python identifier for use as
    a local variable name in the generated Snakefile."""
    return re.sub(r"[^0-9a-zA-Z_]", "_", flag).lower()


def render_flag_snakefile(flag_keys: list[str], executable: str, build_dir: str, args: argparse.Namespace,
                      sample_case_values: dict[str, Any], units: dict[str, str],
                      unit_symbols: dict[str, str], config_names: dict[str, str] | None = None) -> str:
    """Render the generic Snakefile. `flag_keys` is the ordered list of
    "Section.Key" strings found across the metadata's ParameterSet(s).

    `config` (the parsed parameters.json) is read once at Snakefile-parse
    time -- not just inside `run:` -- so its values can also feed the
    `output:` block (needed for --zip-name-flag) the way the upstream
    OpenFOAM Snakefile reads its config up front too.

    Every `config[...]` lookup uses config_key(), NOT the bare flag name --
    see config_key()'s docstring for why (it has to match the keys
    run_benchmark.py's parameter_json_key() actually writes at runtime).
    The "-Section.Key" CLI flag names themselves stay bare regardless.
    """
    config_names = config_names or {}

    def ckey(flag: str) -> str:
        # parameters.json uses the description's parameter names -- the
        # software input name itself, or its software-neutral name when a
        # 1:1 mapping translates between them (config_names).
        return config_key(config_names.get(flag, flag), units.get(flag), unit_symbols)

    mesh_split = args.mesh_split
    special_flags = set()
    if mesh_split:
        special_flags = {
            args.radial_cells_flag, args.angular_cells_flag,
            args.grading_flag, args.inner_radius_flag,
        }
        if args.outer_radius_flag:
            special_flags.add(args.outer_radius_flag)
    special_flags |= set(args.exclude_flag or [])

    plain_flag_keys = [k for k in flag_keys if k not in special_flags and k != args.name_flag]
    # Every non-special flag gets a variable assignment, INCLUDING the name
    # flag (it still needs a value bound to reference in output_path_line
    # below) -- only the plain CLI-flags list itself excludes it, since it's
    # emitted specially rather than as a plain pass-through.
    assignment_keys = [k for k in flag_keys if k not in special_flags]

    assignments = "\n".join(
        f'        {_flag_var_name(k)} = config[{ckey(k)!r}]' for k in assignment_keys
    )

    name_var = _flag_var_name(args.name_flag) if args.name_flag in flag_keys else None
    cli_flags = [f'-{k} "{{{_flag_var_name(k)}}}"' for k in plain_flag_keys]

    if mesh_split:
        # Reproduces the rotating-cylinders benchmark's specific radial
        # mesh scheme: split the radial cell count into two equal halves,
        # mirror the grading value's sign for the second half, and combine
        # the radius values into one flag. This is NOT general-purpose
        # meshing logic -- it's this one benchmark's convention, reproduced
        # here only because it was explicitly asked for; most benchmarks
        # won't need --mesh-split.
        radial_var = _flag_var_name(args.inner_radius_flag)
        radial_sample = sample_case_values.get(args.inner_radius_flag)

        common_lines = [
            f"        {_flag_var_name(args.radial_cells_flag)} = config[{ckey(args.radial_cells_flag)!r}]",
            f"        {_flag_var_name(args.angular_cells_flag)} = config[{ckey(args.angular_cells_flag)!r}]",
            f"        {_flag_var_name(args.grading_flag)} = config[{ckey(args.grading_flag)!r}]",
            f"        _half_radial = int({_flag_var_name(args.radial_cells_flag)} / 2)",
        ]

        if isinstance(radial_sample, str) and len(radial_sample.split()) > 1:
            # Multi-token radius value, stored as a single space-joined
            # string (see generate_metadata.py's --full-value-params /
            # "has string value") -- this is now the value's ACTUAL runtime
            # representation end to end (semantic_benchmark.BenchmarkLoader
            # reads it back as a TextParameter.string_value, a plain
            # string, not a list), so it's already exactly what the CLI
            # flag needs -- no split/join/midpoint math required at all.
            radial_lines = [
                f"        _radial_values_str = config[{ckey(args.inner_radius_flag)!r}]",
            ]
            mesh_split_assignments = "\n".join(common_lines + radial_lines)
            radial_flag_line = f'-{args.inner_radius_flag} "{{_radial_values_str}}"'
        elif isinstance(radial_sample, list):
            # Fallback for a metadata.jsonld that stores the multi-value
            # radius as an actual JSON array instead of a joined string
            # (e.g. hand-edited, or from before this convention existed).
            if len(radial_sample) == 2:
                # Only r1, r2 given -- compute + insert the midpoint
                # ourselves, matching the benchmark's expected 3-value
                # "-Grid.Radial0 r1 mid r2" format.
                radial_lines = [
                    f"        {radial_var} = config[{ckey(args.inner_radius_flag)!r}]",
                    f"        _r1, _r2 = {radial_var}[0], {radial_var}[-1]",
                    f"        _midpoint = 0.5 * (_r1 + _r2)",
                    f'        _radial_values_str = f"{{_r1}} {{_midpoint}} {{_r2}}"',
                ]
            else:
                # Already has everything needed (e.g. r1, mid, r2, or more
                # points) -- pass it straight through, don't recompute.
                radial_lines = [
                    f"        {radial_var} = config[{ckey(args.inner_radius_flag)!r}]",
                    f'        _radial_values_str = " ".join(str(v) for v in {radial_var})',
                ]
            mesh_split_assignments = "\n".join(common_lines + radial_lines)
            radial_flag_line = f'-{args.inner_radius_flag} "{{_radial_values_str}}"'
        else:
            # Scalar radius value (r1 only) -- need r2 from somewhere else:
            # either another case-varying flag or a fixed constant.
            outer_radius_expr = (
                f'config[{ckey(args.outer_radius_flag)!r}]' if args.outer_radius_flag
                else repr(args.outer_radius)
            )
            radial_lines = [
                f"        {radial_var} = config[{ckey(args.inner_radius_flag)!r}]",
                f"        _outer_radius = {outer_radius_expr}",
                f"        _midpoint = 0.5 * ({radial_var} + _outer_radius)",
            ]
            mesh_split_assignments = "\n".join(common_lines + radial_lines)
            radial_flag_line = f'-{args.inner_radius_flag} "{{{radial_var}}} {{_midpoint}} {{_outer_radius}}"'

        mesh_split_flags = [
            f'-{args.radial_cells_flag} "{{_half_radial}} {{_half_radial}}"',
            f'-{args.angular_cells_flag} "{{{_flag_var_name(args.angular_cells_flag)}}}"',
            f'-{args.grading_flag} "{{{_flag_var_name(args.grading_flag)}}} -{{{_flag_var_name(args.grading_flag)}}}"',
            radial_flag_line,
        ]
        assignments = (mesh_split_assignments + "\n\n" + assignments) if assignments else mesh_split_assignments
        cli_flags = mesh_split_flags + cli_flags

    cli_flags_joined = " \\\n".join(cli_flags)
    # Indent to line up inside the shell() f-string.
    cli_flags_indented = "\n            ".join(cli_flags_joined.splitlines())

    if name_var:
        output_path_line = (
            f'-{args.name_flag} "{{container_shared_dir}}/{args.results_subdir}/'
            f'{{conf_name}}/{{{name_var}}}"'
        )
    else:
        output_path_line = None

    # Precomputed outside the f-strings below: pre-3.12 Python forbids a
    # backslash inside an f-string's {} expression part.
    tail = (" \\\n            " + output_path_line) if output_path_line else ""

    if args.zip_name_flag and args.zip_name_flag in flag_keys:
        zip_name_expr = f'f"{{config[{ckey(args.zip_name_flag)!r}]}}.zip"'
    else:
        zip_name_expr = '"results.zip"'

    return f'''\
import json
from pathlib import Path

# Use workflow.basedir to find the root relative to the Snakefile
shared_dir = Path(workflow.basedir).parent

container_image = {args.container_image!r}
container_shared_dir = {args.container_shared_dir!r}
executable = {executable!r}
build_dir = {build_dir!r}

# Read parameters up front (not just inside rule run:) -- these are exactly
# the case-varying parameters generate_metadata.py's --scenario-params
# selected (see extract_case_parameters() in generate_snakefile.py). Needed
# up here so `output:` below can name the zip file after a config value.
with open("parameters.json") as f:
    config = json.load(f)
conf_name = config.get('configuration', 'case')
zip_name = {zip_name_expr}

rule all:
    input:
        "solution_metrics.json",
        zip_name

rule run_simulation:
    input:
        rc_parameters_file = "parameters.json"
    output:
        zip = zip_name,
        metrics = "solution_metrics.json"
    resources:
        serial_run=1
    singularity:
        f"docker://{{container_image}}"
    run:
{assignments}

        shell(
            f"""
            set -euo pipefail
            cd {{build_dir}}

            # Run simulation. We use the container_shared_dir mount point to save results back to the host
            ./{{executable}} \\
            {cli_flags_indented}{tail}
            """
        )
'''


def render_snakefile(adapter, ctx: SnakefileContext, style) -> SnakefileResult:
    args, mapping = ctx.args, ctx.mapping
    args.name_flag = style.name_flag(adapter, args)
    units = dict(ctx.units)
    if args.mesh_split and args.outer_radius_flag and args.outer_radius is not None:
        sys.exit("Error: pass at most one of --outer-radius-flag / --outer-radius, not both.")

    executable = style.executable(ctx)
    build_dir = style.build_dir(adapter, ctx)

    # Any one case's values are enough to know each flag's *shape* (list vs
    # scalar) -- that's determined once, at description time, by whether
    # --full-value-params was used for it, not per-case.
    sample_case_values = next(iter(ctx.cases.values()))

    # Union of "Section.Key" flags across all cases, order-preserving. In
    # practice every case shares the same key set (same scenario-specific
    # selection was used to build the whole graph); union just guards
    # against a case that happens to be missing one.
    flag_keys: list[str] = []
    seen = set()
    for values in ctx.cases.values():
        for k in values:
            if k not in seen:
                seen.add(k)
                flag_keys.append(k)

    use_expressions = bool(mapping) and not (args.mesh_split and mapping.get("reference") is None)
    config_names: dict[str, str] = {}
    if mapping and not use_expressions:
        # 1:1 mapping (neutral names) with the flag-based renderer: flags are
        # the software inputs, parameters.json is read by the neutral name.
        neutral_to_flag = {}
        for i in mapping["inputs"]:
            params = i.get("benchmark_parameters") or []
            if len(params) == 1:
                neutral_to_flag[params[0]] = i["input"]
        flag_keys = [neutral_to_flag.get(k, k) for k in flag_keys]
        sample_case_values = {neutral_to_flag.get(k, k): v for k, v in sample_case_values.items()}
        units = {neutral_to_flag.get(k, k): v for k, v in units.items()}
        config_names = {flag: neutral for neutral, flag in neutral_to_flag.items()}

    if args.mesh_split and not use_expressions and args.inner_radius_flag not in flag_keys:
        sys.exit(
            f"Error: --mesh-split needs --inner-radius-flag ({args.inner_radius_flag!r}) to be one of the "
            f"case-varying parameters, but it's not in: {', '.join(flag_keys)}"
        )
    if args.mesh_split and not use_expressions and args.outer_radius_flag and args.outer_radius_flag not in flag_keys:
        sys.exit(
            f"Error: --outer-radius-flag ({args.outer_radius_flag!r}) is not one of the case-varying "
            f"parameters: {', '.join(flag_keys)}. Use --outer-radius for a fixed value instead."
        )

    def _is_multi_value(v: Any) -> bool:
        return (isinstance(v, str) and len(v.split()) > 1) or isinstance(v, list)

    if args.mesh_split and not use_expressions and not _is_multi_value(sample_case_values.get(args.inner_radius_flag)):
        # Scalar radius value -- r2 has to come from somewhere else.
        if not args.outer_radius_flag and args.outer_radius is None:
            sys.exit(
                f"Error: --inner-radius-flag ({args.inner_radius_flag!r}) is a plain scalar, not a "
                "--full-value-params multi-token value, so an outer radius can't be derived "
                "automatically. Pass --outer-radius-flag or --outer-radius, or use "
                f"--full-value-params in generate_metadata.py to capture "
                f"{args.inner_radius_flag}'s full multi-token value instead."
            )

    if use_expressions:
        # Inputs set from the mapping's expressions (reference mapping, or a
        # 1:1 mapping without --mesh-split).
        if args.mesh_split:
            print("note: --mesh-split is ignored with --reference-benchmark -- the mapping's expressions "
                  "already say how the mesh inputs are computed.")
        text = style.render_mapped(adapter, mapping, executable, build_dir, args, units, ctx.unit_symbols)
    else:
        # Software input names as flags; with a 1:1 mapping (neutral names),
        # parameters.json is read by the neutral name of each input.
        text = render_flag_snakefile(flag_keys, executable, build_dir, args, sample_case_values, units,
                                     ctx.unit_symbols, config_names)

    summary = (
        f"\n{len(ctx.cases)} case(s), {len(flag_keys)} case-varying flag(s): {', '.join(flag_keys)}\n"
        f"executable = {executable}\n"
        f"build_dir  = {build_dir}"
        + ("  (derived from metadata.jsonld)" if not args.build_dir else "  (from --build-dir)")
        + (f"\nmesh-split: ON (radial={args.radial_cells_flag}, angular={args.angular_cells_flag}, "
           f"grading={args.grading_flag}, r1={args.inner_radius_flag}, "
           f"r2={args.outer_radius_flag or args.outer_radius})" if args.mesh_split else "\nmesh-split: off")
    )
    return SnakefileResult(text=text, summary=[summary])


def add_snakefile_arguments(group: argparse._ArgumentGroup) -> None:
    group.add_argument("--exclude-flag", action="append", default=None,
                       help="A 'Section.Key' to drop entirely from the CLI flags passed to the executable "
                            "(repeatable), e.g. for parameters that are compile-time-only and shouldn't be "
                            "passed at runtime.")
    group.add_argument("--mesh-split", action="store_true",
                       help="Opt in to the rotating-cylinders benchmark's specific radial-mesh-splitting math: "
                            "split a cells flag in half, mirror a grading flag's sign, and combine inner/outer "
                            "radius with their midpoint into one flag. This is ONE benchmark's specific meshing "
                            "convention, not general-purpose -- leave off (default) for any other benchmark.")
    group.add_argument("--radial-cells-flag", default="Grid.Cells0",
                       help="[--mesh-split] Section.Key holding the total radial cell count to split in half "
                            "(default: Grid.Cells0).")
    group.add_argument("--angular-cells-flag", default="Grid.Cells1",
                       help="[--mesh-split] Section.Key holding the angular/azimuthal cell count, passed through "
                            "unsplit (default: Grid.Cells1).")
    group.add_argument("--grading-flag", default="Grid.Grading0",
                       help="[--mesh-split] Section.Key holding the radial grading factor, mirrored with a sign "
                            "flip for the second mesh block (default: Grid.Grading0).")
    group.add_argument("--inner-radius-flag", default="Grid.Radial0",
                       help="[--mesh-split] Section.Key holding the inner radius (r1) -- must be one of the "
                            "case-varying parameters (default: Grid.Radial0). If this parameter's value is a list "
                            "(captured via --full-value-params, e.g. 'Radial0 = 1.0 1.5 2.0'), it's used directly "
                            "and --outer-radius/--outer-radius-flag are not needed.")
    group.add_argument("--outer-radius-flag", default=None,
                       help="[--mesh-split] Section.Key holding the outer radius (r2), if it's itself case-varying "
                            "and --inner-radius-flag is a plain scalar (not a --full-value-params list). Mutually "
                            "exclusive with --outer-radius.")
    group.add_argument("--outer-radius", type=float, default=None,
                       help="[--mesh-split] Fixed outer radius (r2) value, if it's constant across all cases and "
                            "--inner-radius-flag is a plain scalar (not a --full-value-params list). Mutually "
                            "exclusive with --outer-radius-flag.")
