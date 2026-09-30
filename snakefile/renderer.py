# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

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
    # Matches the Simulation-Benchmarks run_benchmark.py UNIT_SYMBOLS
    # ("M": "m", "RAD-PER-SEC": "rad/s"), which writes the parameters.json
    # keys this Snakefile reads.
    "unit:RAD-PER-SEC": "rad/s",
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


# ---------------------------------------------------------------------------
# Snakefiles generated from an input mapping (metadata.mapping): the
# description's software-neutral parameters are read from parameters.json,
# and each software input is set from its expression.
# ---------------------------------------------------------------------------

#: Names a benchmark parameter can't take as a Snakefile variable.
RESERVED_NAMES = {"config", "json", "Path", "shell", "workflow", "rules", "conf_name", "zip_name",
                  "container_image", "container_shared_dir", "executable", "build_dir", "shared_dir",
                  "snakefile_dir", "application", "metrics_script"}


def _parameter_assignments(names: list[str], units: dict[str, str], unit_symbols: dict[str, str]) -> str:
    clash = sorted(set(names) & RESERVED_NAMES)
    if clash:
        raise SystemExit(f"Error: benchmark parameter name(s) {clash} clash with Snakefile variables; rename them.")
    return "\n".join(f"{n} = config[{config_key(n, units.get(n), unit_symbols)!r}]" for n in names)


def render_mapped_dumux_snakefile(mapping: dict[str, Any], executable: str, build_dir: str,
                                  args: argparse.Namespace, units: dict[str, str],
                                  unit_symbols: dict[str, str]) -> str:
    """DuMux Snakefile from a mapping: every mapped input becomes
    -Section.Key "<expression>" (an f-string over the benchmark parameters),
    e.g. -Grid.Cells0 "{cells_radial // 2} {cells_radial // 2}"."""
    inputs = mapping["inputs"]
    by_flag = {i["input"]: i for i in inputs}
    names = sorted({n for i in inputs if i.get("expression") for n in i.get("benchmark_parameters", [])})

    flags = [f'-{i["input"]} "{i["expression"]}"' for i in inputs
             if i.get("expression") and i["input"] != args.name_flag]
    name_input = by_flag.get(args.name_flag)
    if name_input:
        base = name_input["expression"] or name_input.get("template_value") or "output"
        flags.append(f'-{args.name_flag} "{{container_shared_dir}}/{args.results_subdir}/{{conf_name}}/{base}"')
    flag_block = " \\\n            ".join(flags)

    zip_input = by_flag.get(args.zip_name_flag) if args.zip_name_flag else None
    if zip_input:
        zip_base = zip_input["expression"] or zip_input.get("template_value") or "results"
        zip_name_expr = f'f"{zip_base}.zip"'
    else:
        zip_name_expr = '"results.zip"'

    return f'''\
# Generated by benchmantic from the input mapping ({mapping.get("software")}):
# the benchmark's software-neutral parameters are read from parameters.json
# and each DuMux input is computed from them (see *_mapping.json).
import json
from pathlib import Path

container_image = {args.container_image!r}
container_shared_dir = {args.container_shared_dir!r}
executable = {executable!r}
build_dir = {build_dir!r}

with open("parameters.json") as f:
    config = json.load(f)
conf_name = config.get('configuration', 'case')

# benchmark parameters
{_parameter_assignments(names, units, unit_symbols)}

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
        shell(
            f"""
            set -euo pipefail
            cd {{build_dir}}
            ./{{executable}} \\
            {flag_block}
            """
        )
'''


def render_openfoam_snakefile(mapping: dict[str, Any], hints: dict[str, Any], args: argparse.Namespace,
                              units: dict[str, str], unit_symbols: dict[str, str]) -> str:
    """OpenFOAM Snakefile from a mapping. Expects the case (system/,
    constant/, 0/) in the working directory -- as run_benchmark.py sets it up
    by unpacking the case template -- then sets every mapped input:
      template placeholder {x}  -> sed into the file (X.template -> X)
      plain dictionary entry    -> foamDictionary -entry <path> -set <value>
    runs the case (Allrun, or blockMesh + the solver) and the metrics script
    that sits next to this Snakefile."""
    inputs = [i for i in mapping["inputs"] if i.get("expression")]
    names = sorted({n for i in inputs for n in i.get("benchmark_parameters", [])})

    template_cmds: dict[str, list[str]] = {}
    inplace_cmds: list[str] = []
    dict_cmds: list[str] = []
    for i in inputs:
        section, value = i["section"], i["expression"]
        if i.get("placeholder"):
            # {x} must survive two formatting passes -- the f-string, then
            # Snakemake's shell() -- so it's written as {{{{x}}}}.
            pattern = "s|" + "{{{{" + i["placeholder"] + "}}}}" + "|" + value + "|g"
            if section.endswith(".template"):
                template_cmds.setdefault(section, []).append(f'-e "{pattern}"')
            else:
                inplace_cmds.append(f'sed -i "{pattern}" {section}')
        else:
            entry = i["key"]
            dict_cmds.append(f'foamDictionary -entry "{entry}" -set "{value}" {section}')
    lines = [f'sed {" ".join(cmds)} {section} > {section[: -len(".template")]}'
             for section, cmds in template_cmds.items()]
    lines += inplace_cmds
    setup = "\n            ".join(lines) or ": # no template placeholders"
    dict_setup = "\n            ".join(dict_cmds) or ": # no dictionary entries to set"

    application = hints.get("executable_name") or "simpleFoam"
    run_case = ("chmod +x Allrun && ./Allrun" if hints.get("has_allrun")
                else f"blockMesh > log.blockMesh 2>&1 && {application} > log.{application} 2>&1")
    metrics_script = hints.get("metrics_script")
    metrics_rule = f'''
rule compute_metrics:
    input:
        sim_done = "log.{application}",
        script = ancient(str(snakefile_dir / {metrics_script!r}))
    output:
        "solution_metrics.json"
    shell:
        "python3 {{input.script}} {hints.get("metrics_args", ".")}"
''' if metrics_script else '''
# TODO: no metrics script was found -- add a rule that writes solution_metrics.json
'''

    return f'''\
# Generated by benchmantic from the input mapping (OpenFOAM): the
# benchmark's software-neutral parameters are read from parameters.json and
# written into the case's dictionaries (see *_mapping.json). Run it in a
# folder that holds the OpenFOAM case, as run_benchmark.py prepares it.
import json
from pathlib import Path

snakefile_dir = Path(workflow.snakefile).resolve().parent
container_image = {args.container_image!r}

with open("parameters.json") as f:
    config = json.load(f)
conf_name = config.get('configuration', 'case')

# benchmark parameters
{_parameter_assignments(names, units, unit_symbols)}

rule all:
    input:
        "solution_metrics.json"

rule run_simulation:
    input:
        "parameters.json"
    output:
        "log.{application}"
    resources:
        serial_run=1
    container:
        f"docker://{{container_image}}"
    run:
        shell(
            f"""
            set -euo pipefail
            # 1. benchmark parameters -> case files
            {setup}

            # 2. OpenFOAM environment, dictionary entries, run
            set +u
            for rc in /usr/lib/openfoam/openfoam*/etc/bashrc /opt/openfoam*/etc/bashrc; do
                if [ -f "$rc" ]; then . "$rc"; break; fi
            done
            set -u
            {dict_setup}
            {run_case}
            """
        )
{metrics_rule}'''
