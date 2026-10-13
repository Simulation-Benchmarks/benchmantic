# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
snakefile.styles.command_line

For software run as one executable with every input on the command line
(e.g. DuMux: ./test_ff_... -Grid.Cells0 "40 40" ...). With an input mapping
(the default), each input becomes one flag whose value is computed from the
benchmark's software-neutral parameters (see *_mapping.json); without one
(the software's own parameter names, or the snakefile-only preset), each
case-varying input is passed as read from parameters.json.

Settings in the software description (`run:`, with `type: program`):
  working_folder: "{container_root}/{repository}/build-cmake/{implementation_path}"
        where the program is run inside the container; {container_root}
        is the folder above results_folder, {repository} the repository
        name, {implementation_path} the implementation's path in it
  command: "./{executable}"
  input_argument: '-{input} "{value}"'   how one input is passed
  output_name_input: Problem.Name        the input naming the run's output
                                         (default for --name-flag)
  results_zip_name_input: Problem.Name   the input the results zip is named
                                         after (default for --zip-name-flag)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from snakefile.renderer import config_key, parameter_assignments
from software.base import SnakefileContext, SnakefileResult


def add_arguments(group) -> None:
    group.add_argument("--name-flag", default=None,
                       help="The input (e.g. 'Problem.Name') naming the run's output, so its value gets the "
                            "{container_shared_dir}/{results_subdir}/{configuration}/... prefix instead of being "
                            "passed through as-is. Default: `output_name_input` in the software description.")
    group.add_argument("--zip-name-flag", default=None,
                       help="Name the output zip '{value of this input}.zip' instead of 'results.zip' (e.g. "
                            "--zip-name-flag Problem.Name). Default: `results_zip_name_input` in the software "
                            "description, if it has one.")
    group.add_argument("--exclude-flag", action="append", default=None,
                       help="An input to leave out of the command line (repeatable), e.g. one that is "
                            "compile-time-only. Only used without an input mapping.")


def build_hints(adapter, module_dir: Path, repo_root: Path) -> dict[str, Any]:
    return {}


def repository_name(repo_url: str | None) -> str | None:
    """'https://.../benchmarks/rotating-cylinders.git' -> 'rotating-cylinders'."""
    if not repo_url:
        return None
    name = repo_url.rstrip("/").rsplit("/", 1)[-1]
    return name[:-4] if name.endswith(".git") else name


def executable(ctx: SnakefileContext) -> str:
    exe = ctx.args.executable or ctx.build_hints.get("executable_name")
    if not exe:
        hints_path = ctx.metadata_jsonld.with_name(ctx.metadata_jsonld.stem + ".build_hints.json")
        sys.exit(
            f"Error: no executable name found in {hints_path} (generate_metadata.py couldn't "
            "find/parse a CMakeLists.txt declaring it -- or that sidecar file doesn't exist at "
            "all, e.g. this metadata.jsonld predates that feature) and none was given via "
            "--executable."
        )
    return exe


def build_dir(adapter, ctx: SnakefileContext) -> str:
    """--build-dir, else `working_folder` from the software description with
    the repository name (codeRepository of the description) and the
    implementation's path (build_hints sidecar) filled in."""
    if ctx.args.build_dir:
        return ctx.args.build_dir
    template = adapter.run_settings.get("working_folder") or "{container_root}/{repository}/{implementation_path}"
    module_path = ctx.build_hints.get("module_relative_path")
    repository = repository_name((ctx.by_id.get("./") or {}).get("codeRepository"))
    if not module_path or not repository:
        sys.exit(
            "Error: could not derive a build directory (needs both a codeRepository URL on "
            "metadata.jsonld's root entity and module_relative_path in the build_hints.json "
            "sidecar) and none was given via --build-dir."
        )
    if not ctx.args.container_shared_dir:
        sys.exit("Error: no results folder in the container -- set `results_folder` under `run:` in the "
                 "software description (or pass --container-shared-dir).")
    return template.format(container_root=str(Path(ctx.args.container_shared_dir).parent),
                           repository=repository, implementation_path=module_path)


def name_flag(adapter, args) -> str | None:
    return args.name_flag or adapter.run_settings.get("output_name_input")


def zip_name_flag(adapter, args) -> str | None:
    """The input the results zip is named after: --zip-name-flag, else the
    description's `results_zip_name_input`, else none (results.zip)."""
    return args.zip_name_flag or adapter.run_settings.get("results_zip_name_input")


def render_mapped(adapter, mapping: dict[str, Any], executable_name: str, build_directory: str, args,
                  units: dict[str, str], unit_symbols: dict[str, str]) -> str:
    """Every mapped input becomes one flag whose value is an f-string over
    the benchmark parameters, e.g. -Grid.Cells0 "{cells_radial // 2} {cells_radial // 2}"."""
    settings = adapter.run_settings
    input_flag = settings.get("input_argument", '--{input} "{value}"')
    command = settings.get("command", "./{executable}")
    name_input = name_flag(adapter, args)
    inputs = mapping["inputs"]
    by_flag = {i["input"]: i for i in inputs}
    names = sorted({n for i in inputs if i.get("expression") for n in i.get("benchmark_parameters", [])})

    flags = [input_flag.format(input=i["input"], value=i["expression"]) for i in inputs
             if i.get("expression") and i["input"] != name_input]
    name_entry = by_flag.get(name_input)
    if name_entry:
        base = name_entry["expression"] or name_entry.get("template_value") or "output"
        flags.append(input_flag.format(
            input=name_input, value=f"{{container_shared_dir}}/{args.results_subdir}/{{conf_name}}/{base}"))
    flag_block = " \\\n            ".join(flags)

    zip_flag = zip_name_flag(adapter, args)
    zip_input = by_flag.get(zip_flag) if zip_flag else None
    if zip_input:
        zip_base = zip_input["expression"] or zip_input.get("template_value") or "results"
        zip_name_expr = f'f"{zip_base}.zip"'
    else:
        zip_name_expr = '"results.zip"'

    return f'''\
# Generated by benchmantic from the input mapping ({mapping.get("software")}):
# the benchmark's software-neutral parameters are read from parameters.json
# and each {adapter.name} input is computed from them (see *_mapping.json).
import json
from pathlib import Path

container_image = {args.container_image!r}
container_shared_dir = {args.container_shared_dir!r}
executable = {executable_name!r}
build_dir = {build_directory!r}

with open("parameters.json") as f:
    config = json.load(f)
conf_name = config.get('configuration', 'case')

# benchmark parameters
{parameter_assignments(names, units, unit_symbols)}

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
            {command} \\
            {flag_block}
            """
        )
'''


def _var_name(name: str) -> str:
    """'Grid.Cells0' -> 'grid_cells0', a Python identifier for the Snakefile."""
    return re.sub(r"[^0-9a-zA-Z_]", "_", name).lower()


def render_flags(adapter, flag_keys: list[str], executable_name: str, build_directory: str, args,
                 units: dict[str, str], unit_symbols: dict[str, str]) -> str:
    """Without a mapping: every case-varying input is read from
    parameters.json (by config_key(), matching run_benchmark.py's keys) and
    passed on the command line as is."""
    settings = adapter.run_settings
    input_flag = settings.get("input_argument", '--{input} "{value}"')
    command = settings.get("command", "./{executable}")
    name_input = name_flag(adapter, args)

    def ckey(flag: str) -> str:
        return config_key(flag, units.get(flag), unit_symbols)

    left_out = set(args.exclude_flag or [])
    plain_flag_keys = [k for k in flag_keys if k not in left_out and k != name_input]
    # The name input still needs a variable (for the output path below).
    assignment_keys = [k for k in flag_keys if k not in left_out]
    assignments = "\n".join(f'        {_var_name(k)} = config[{ckey(k)!r}]' for k in assignment_keys)

    name_var = _var_name(name_input) if name_input in flag_keys else None
    cli_flags = [input_flag.format(input=k, value="{" + _var_name(k) + "}") for k in plain_flag_keys]
    cli_flags_joined = " \\\n".join(cli_flags)
    # Indent to line up inside the shell() f-string.
    cli_flags_indented = "\n            ".join(cli_flags_joined.splitlines())

    output_path_line = (input_flag.format(
        input=name_input, value=f"{{container_shared_dir}}/{args.results_subdir}/{{conf_name}}/{{{name_var}}}")
        if name_var else None)
    # Precomputed outside the f-strings below: pre-3.12 Python forbids a
    # backslash inside an f-string's {} expression part.
    tail = (" \\\n            " + output_path_line) if output_path_line else ""

    zip_flag = zip_name_flag(adapter, args)
    if zip_flag and zip_flag in flag_keys:
        zip_name_expr = f'f"{{config[{ckey(zip_flag)!r}]}}.zip"'
    else:
        zip_name_expr = '"results.zip"'

    return f'''\
import json
from pathlib import Path

# Use workflow.basedir to find the root relative to the Snakefile
shared_dir = Path(workflow.basedir).parent

container_image = {args.container_image!r}
container_shared_dir = {args.container_shared_dir!r}
executable = {executable_name!r}
build_dir = {build_directory!r}

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
            {command} \\
            {cli_flags_indented}{tail}
            """
        )
'''


def render(adapter, ctx: SnakefileContext) -> SnakefileResult:
    exe = executable(ctx)
    directory = build_dir(adapter, ctx)
    if ctx.mapping:
        text = render_mapped(adapter, ctx.mapping, exe, directory, ctx.args, ctx.units, ctx.unit_symbols)
        used = [i["input"] for i in ctx.mapping["inputs"] if i.get("expression")]
        what = "mapped input(s)"
    else:
        used = []
        for values in ctx.cases.values():
            used += [k for k in values if k not in used]
        text = render_flags(adapter, used, exe, directory, ctx.args, ctx.units, ctx.unit_symbols)
        what = "case-varying input(s)"
    summary = (f"\n{len(ctx.cases)} case(s), {len(used)} {what}: {', '.join(used)}\n"
               f"executable = {exe}\n"
               f"build_dir  = {directory}"
               + ("  (from --build-dir)" if ctx.args.build_dir else "  (derived from metadata.jsonld)"))
    return SnakefileResult(text=text, summary=[summary])
