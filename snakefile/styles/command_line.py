# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
snakefile.styles.command_line

For software run as one executable with every input on the command line
(e.g. DuMux: ./test_ff_... -Grid.Cells0 "40 40" ...). Each mapped
benchmark parameter becomes one flag whose value is computed from the
benchmark's software-neutral parameters (see *_mapping.json).

Settings in software.yaml (under `snakefile:`):
  build_dir: "{container_root}/{repository}/build-cmake/{module_path}"
        where the executable is run inside the container; {container_root}
        is the folder above --container-shared-dir, {repository} the
        repository name, {module_path} the implementation's path in it
  command: "./{executable}"
  input_flag: '-{input} "{value}"'    how one input is passed
  name_input: Problem.Name            the input naming the run's output
                                      (default for --name-flag)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from software.base import SnakefileContext, SnakefileResult
from snakefile.renderer import parameter_assignments


def add_arguments(group) -> None:
    group.add_argument("--name-flag", default=None,
                       help="The input (e.g. 'Problem.Name') naming the run's output, so its value gets the "
                            "{container_shared_dir}/{results_subdir}/{configuration}/... prefix instead of being "
                            "passed through as-is. Default: the software's `name_input` in software.yaml.")
    group.add_argument("--zip-name-flag", default=None,
                       help="Name the output zip '{value of this input}.zip' instead of the fixed 'results.zip' "
                            "(e.g. --zip-name-flag Problem.Name). Off by default -- must be one of the "
                            "case-varying parameters, or this has no effect.")


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
    """--build-dir, else `build_dir` from software.yaml with the repository
    name (codeRepository of the description) and the implementation's path
    (build_hints sidecar) filled in."""
    if ctx.args.build_dir:
        return ctx.args.build_dir
    template = adapter.snakefile_settings.get("build_dir") or "{container_root}/{repository}/{module_path}"
    module_path = ctx.build_hints.get("module_relative_path")
    repository = repository_name((ctx.by_id.get("./") or {}).get("codeRepository"))
    if not module_path or not repository:
        sys.exit(
            "Error: could not derive a build directory (needs both a codeRepository URL on "
            "metadata.jsonld's root entity and module_relative_path in the build_hints.json "
            "sidecar) and none was given via --build-dir."
        )
    return template.format(container_root=str(Path(ctx.args.container_shared_dir).parent),
                           repository=repository, module_path=module_path)


def name_flag(adapter, args) -> str | None:
    return args.name_flag or adapter.snakefile_settings.get("name_input")


def render_mapped(adapter, mapping: dict[str, Any], executable_name: str, build_directory: str, args,
                  units: dict[str, str], unit_symbols: dict[str, str]) -> str:
    """Every mapped input becomes one flag whose value is an f-string over
    the benchmark parameters, e.g. -Grid.Cells0 "{cells_radial // 2} {cells_radial // 2}"."""
    settings = adapter.snakefile_settings
    input_flag = settings.get("input_flag", '--{input} "{value}"')
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

    zip_input = by_flag.get(args.zip_name_flag) if args.zip_name_flag else None
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


def render(adapter, ctx: SnakefileContext) -> SnakefileResult:
    if not ctx.mapping:
        sys.exit(
            f"Error: {adapter.name} Snakefiles are generated from the input mapping (*_mapping.json next to "
            f"{ctx.metadata_jsonld.name}), and there is none -- describe the benchmark with neutral parameter "
            "names (the default) to get one."
        )
    exe = executable(ctx)
    directory = build_dir(adapter, ctx)
    text = render_mapped(adapter, ctx.mapping, exe, directory, ctx.args, ctx.units, ctx.unit_symbols)
    mapped = [i["input"] for i in ctx.mapping["inputs"] if i.get("expression")]
    summary = (f"\n{len(ctx.cases)} case(s), {len(mapped)} mapped input(s): {', '.join(mapped)}\n"
               f"executable = {exe}\n"
               f"build_dir  = {directory}")
    return SnakefileResult(text=text, summary=[summary])
