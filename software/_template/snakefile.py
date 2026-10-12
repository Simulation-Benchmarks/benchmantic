# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
software._template.snakefile

How one case of this code is run. render() gets everything
snakefile.generator gathered (ctx: parsed options, the description graph,
build hints, per-case values and units, the input mapping) and returns the
Snakefile text; the generator writes it, the optional zip and the per-case
parameters.json previews.

With an input mapping (the default: neutral parameter names), each entry of
ctx.mapping["inputs"] has "section", "key", "input" (display name),
"expression" (an f-string body over the benchmark parameters, or None) and
"benchmark_parameters" (the names it uses). parameter_assignments() writes
the Python lines that read those parameters from parameters.json.
"""

from __future__ import annotations

from snakefile.renderer import parameter_assignments
from software.base import SnakefileContext, SnakefileResult


def render(ctx: SnakefileContext) -> SnakefileResult:
    mapping = ctx.mapping or {"inputs": []}
    inputs = [i for i in mapping["inputs"] if i.get("expression")]
    names = sorted({n for i in inputs for n in i.get("benchmark_parameters", [])})
    # TODO: turn each mapped input into a command-line argument, a line in
    # an input file, ... -- whatever this code reads.
    settings = "\n            ".join(f'# {i["input"]} = {i["expression"]}' for i in inputs)
    text = f'''\
import json

container_image = {ctx.args.container_image!r}

with open("parameters.json") as f:
    config = json.load(f)

# benchmark parameters
{parameter_assignments(names, ctx.units, ctx.unit_symbols)}

rule all:
    input:
        "solution_metrics.json"

rule run_simulation:
    input:
        "parameters.json"
    output:
        "solution_metrics.json"
    container:
        f"docker://{{container_image}}"
    run:
        shell(
            f"""
            set -euo pipefail
            {settings}
            # TODO: run the simulation and write solution_metrics.json
            """
        )
'''
    return SnakefileResult(text=text, summary=[f"\n{len(ctx.cases)} case(s), {len(inputs)} mapped input(s)"])
