"""
snakefile.renderer

Renders the actual Snakefile text (CLI flag construction, optional
mesh-split math, zip-name-from-config logic), the parameters.json key
convention (mirroring run_benchmark.py's unit-suffix keys), and the
per-case preview parameters.json.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any


#: Mirrors run_benchmark.py's UNIT_SYMBOLS. MUST be kept in sync with that
#: file -- it determines the exact parameters.json keys run_benchmark.py
#: writes at runtime. If that file's UNIT_SYMBOLS differs (more units, or
#: a different symbol), pass --unit-symbol to override/extend this default
#: rather than letting the two silently drift apart.
DEFAULT_UNIT_SYMBOLS: dict[str, str] = {
    "unit:M": "m",
    "unit:PA": "Pa",
}


def config_key(flag: str, unit: str | None, unit_symbols: dict[str, str]) -> str:
    """Mirror run_benchmark.py's parameter_json_key(): append a bracketed
    unit symbol to the key when the parameter's unit is a known one (e.g.
    "Grid.Radial0" -> "Grid.Radial0[m]"), otherwise use the bare
    Section.Key. This is the key used to look the value up in `config`
    (parameters.json) -- NOT the "-Section.Key" CLI flag name passed to the
    executable, which always stays the bare label either way.
    """
    symbol = unit_symbols.get(unit) if unit else None
    return f"{flag}[{symbol}]" if symbol else flag



def _flag_var_name(flag: str) -> str:
    """'Grid.Cells0' -> 'grid_cells0', a safe Python identifier for use as
    a local variable name in the generated Snakefile."""
    return re.sub(r"[^0-9a-zA-Z_]", "_", flag).lower()


def render_snakefile(flag_keys: list[str], executable: str, build_dir: str, args: argparse.Namespace,
                      sample_case_values: dict[str, Any], units: dict[str, str],
                      unit_symbols: dict[str, str]) -> str:
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
    def ckey(flag: str) -> str:
        return config_key(flag, units.get(flag), unit_symbols)

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


def render_parameters_json(case_id: str, values: dict[str, Any], units: dict[str, str],
                            unit_symbols: dict[str, str]) -> str:
    """Render this case's preview parameters.json -- NOTE: when this
    Snakefile is actually run via run_benchmark.py, THAT script generates
    its own parameters.json per configuration (via
    create_parameter_files_from_benchmark()/create_parameter_file()) and
    this file is not read at all. It's kept as a human-readable preview /
    manual-testing aid, so its keys mirror run_benchmark.py's
    parameter_json_key() convention for consistency with what will
    actually be used at runtime.
    """
    payload = {"configuration": case_id}
    for flag, value in values.items():
        payload[config_key(flag, units.get(flag), unit_symbols)] = value
    return json.dumps(payload, indent=2) + "\n"


# ============================================================
# Entry point
# ============================================================
