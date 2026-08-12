#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
describe_benchmark.py

The only entry point for this package. Generates the semantic benchmark
description and Snakefile from a benchmark module, calling metadata.builder
and snakefile.generator directly in-process (not via subprocess), and
writes both artifacts together into outputs/<software-name>/, matching this
pattern from run-benchmark.yml:

    cd $GITHUB_WORKSPACE/outputs/<software-name>
    python3 run_benchmark.py \\
        --benchmark-file <benchmark-filename> \\
        --benchmark-zip <zip-name> \\
        --result-path ./results

Produces:
    outputs/<software-name>/<benchmark-filename>  -- metadata.builder's
                                                       RO-Crate JSON-LD,
                                                       loadable by
                                                       semantic_benchmark.BenchmarkLoader
    outputs/<software-name>/<zip-name>             -- snakefile.generator's
                                                       Snakefile, packaged
                                                       flat, ready for
                                                       --benchmark-zip

Not tied to any one simulation tool: <module_dir> just needs to be
something metadata.builder can already handle (a DUNE/DuMux-style module
with main.cc/problem.hh/params.input, or a repo checkout containing one) --
run it once per tool (DuMux, OpenFOAM, ...) with that tool's own
--software-name, --container-image, and mesh-split settings.

The output directory is always outputs/<software-name>/ (--software-name,
or if that's not given, the software name derived from the generated
benchmark file itself -- benchmark.jsonld's local:software label,
slugified, e.g. "DuMux" -> "dumux"). Since the software name only exists
after metadata.builder has run, this script generates the benchmark file
to a staging location first, derives the name from it if needed, then
moves both artifacts into outputs/<software-name>/ together.

By default, output is quiet -- one short line per step, with the full
underlying output only shown if something fails. Pass --verbose (-v) to
always see it live. Exception: if --scenario-params isn't given, the
interactive parameter-selection prompt always runs live (it needs a real
terminal for input()), regardless of --verbose.

Usage
-----
    python3 describe_benchmark.py <module_dir> \\
        --scenario-params Cells0,Cells1,Grading0,Radial0,Name,Omega1,Omega2 \\
        --full-value-params Radial0 \\
        --container-image git.iws.uni-stuttgart.de:4567/benchmarks/rotating-cylinders:3.1 \\
        --container-shared-dir /dumux/shared \\
        --zip-name rotating-cylinders.zip \\
        --mesh-split --zip-name-flag Problem.Name

--software-name isn't needed above -- it's derived automatically from the
generated benchmark file (e.g. "DuMux" -> "dumux"). Pass it explicitly only
to override that derived name.

<module_dir> can be the exact benchmark folder or a repo checkout --
metadata.builder resolves it the same way it always does.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import metadata.builder as builder  # noqa: E402
import snakefile.generator as generator  # noqa: E402
from metadata.graph import load_graph  # noqa: E402


def run_step(label: str, verbose: bool, fn, *fn_args, **fn_kwargs) -> None:
    """Call `fn` (an in-process build step). By default, stays quiet --
    prints one short line, and only dumps the step's captured stdout+stderr
    if it actually fails (raises). Pass --verbose to always show it live
    instead. Mirrors the original subprocess-based capture_output=True
    behavior now that these run in-process instead.
    """
    print(f"-> {label}...")
    if verbose:
        fn(*fn_args, **fn_kwargs)
        return

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            fn(*fn_args, **fn_kwargs)
    except SystemExit as exc:
        sys.stdout.write(buf.getvalue())
        sys.exit(f"Error: {label} failed -- {exc.code}")
    except Exception:
        sys.stdout.write(buf.getvalue())
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("module_dir", type=Path,
                     help="Path to the benchmark module (or a repo checkout containing it) -- "
                          "anything metadata.builder can already handle.")
    ap.add_argument("--software-name", default=None,
                     help="Software name -- both artifacts (benchmark file + zip) are written together to "
                          "outputs/<software-name>/, and it's passed through as snakefile.generator's "
                          "--software-name. Default: derived from the generated benchmark file itself "
                          "(benchmark.jsonld's local:software label, slugified, e.g. 'DuMux' -> 'dumux').")
    ap.add_argument("--benchmark-filename", default="benchmark.jsonld",
                     help="Filename for the generated semantic description (default: benchmark.jsonld -- "
                          "it's JSON-LD content, so this reflects that; pass --benchmark-filename benchmark.json "
                          "if your run_benchmark.py --benchmark-file argument is hardcoded to that name).")
    ap.add_argument("--zip-name", default="benchmark-workflow.zip",
                     help="Filename for the packaged Snakefile zip, written alongside the benchmark file "
                          "(default: benchmark-workflow.zip).")

    # --- metadata.builder pass-through ---
    meta = ap.add_argument_group("metadata.builder options")
    meta.add_argument("--main-cc", type=Path, default=None)
    meta.add_argument("--scenario-params", type=str, default=None)
    meta.add_argument("--full-value-params", type=str, default=None)
    meta.add_argument("--provider", type=str, default=None)
    meta.add_argument("--model", type=str, default=None)
    meta.add_argument("--clear-cache", action="store_true")
    meta.add_argument("--fallback-on-error", action="store_true")
    meta.add_argument("--skip-validation", action="store_true")

    # --- snakefile.generator pass-through ---
    snake = ap.add_argument_group("snakefile.generator options")
    snake.add_argument("--container-image", required=True)
    snake.add_argument("--container-shared-dir", required=True,
                        help="Mount point inside the container used to save results back to the host "
                             "(e.g. /dumux/shared) -- no universal default, this is tool/container-specific.")
    snake.add_argument("--executable", default=None)
    snake.add_argument("--build-dir", default=None)
    snake.add_argument("--results-subdir", default="results")
    snake.add_argument("--name-flag", default="Problem.Name")
    snake.add_argument("--zip-name-flag", default=None)
    snake.add_argument("--exclude-flag", action="append", default=None)
    snake.add_argument("--mesh-split", action="store_true",
                        help="Opt in to the rotating-cylinders-style radial-mesh-splitting math -- see "
                             "snakefile.generator --help. Only relevant to benchmarks that need it.")
    snake.add_argument("--radial-cells-flag", default="Grid.Cells0")
    snake.add_argument("--angular-cells-flag", default="Grid.Cells1")
    snake.add_argument("--grading-flag", default="Grid.Grading0")
    snake.add_argument("--inner-radius-flag", default="Grid.Radial0")
    snake.add_argument("--outer-radius-flag", default=None)
    snake.add_argument("--outer-radius", type=float, default=None)
    snake.add_argument("--unit-symbol", action="append", default=None)

    ap.add_argument("-v", "--verbose", action="store_true",
                     help="Show the full underlying build output live, instead of the default short "
                          "progress lines (still shown on failure either way).")

    return ap


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    """Run the full build (semantic description + Snakefile). Returns
    (benchmark_path, zip_path) so callers (e.g. workflow.py) can chain into
    show_description.py/verify_description.py without re-deriving the output
    location themselves.
    """

    staging_dir = Path(tempfile.mkdtemp(prefix="describe_benchmark_"))
    staged_benchmark = staging_dir / args.benchmark_filename

    # 1. Semantic description -- generated to a staging location first,
    # since the output directory's default depends on reading it back (the
    # software name lives in the graph itself, not something we know
    # beforehand unless --software-name was given explicitly).
    builder_args = argparse.Namespace(
        module_dir=args.module_dir,
        main_cc=args.main_cc,
        output=staged_benchmark,
        scenario_params=args.scenario_params,
        full_value_params=args.full_value_params,
        provider=args.provider or builder.DEFAULT_PROVIDER,
        model=args.model,
        fallback_on_error=args.fallback_on_error,
        verbose=args.verbose,
        clear_cache=args.clear_cache,
        skip_validation=args.skip_validation,
        validate_severity="REQUIRED",
    )
    # The interactive parameter-selection prompt (metadata.parameters) only
    # runs when --scenario-params wasn't supplied -- and it needs a real
    # terminal (it calls input()), so it can't be silently captured into
    # the quiet-mode buffer the way the rest of this step's output can.
    # Force live output for just this step in that case, regardless of
    # --verbose, so the prompt (and what you type) actually show up.
    needs_interactive = args.scenario_params is None
    if needs_interactive and not args.verbose:
        print("   (no --scenario-params given -- showing the interactive parameter selection live)")
    run_step("Generating semantic description", args.verbose or needs_interactive, builder.build, builder_args)

    software_name = args.software_name or generator.derive_software_name(load_graph(staged_benchmark))
    output_dir = Path("outputs") / software_name
    print(f"-> Software: {software_name!r} (output directory: {output_dir}/)")

    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = output_dir / args.benchmark_filename
    zip_path = output_dir / args.zip_name

    shutil.move(str(staged_benchmark), str(benchmark_path))
    staged_hints = staging_dir / f"{Path(args.benchmark_filename).stem}.build_hints.json"
    if staged_hints.exists():
        shutil.move(str(staged_hints), str(output_dir / staged_hints.name))
    shutil.rmtree(staging_dir, ignore_errors=True)

    # 2. Snakefile, packaged as a benchmark zip
    generator_args = argparse.Namespace(
        metadata_jsonld=benchmark_path,
        container_image=args.container_image,
        container_shared_dir=args.container_shared_dir,
        software_name=software_name,
        executable=args.executable,
        build_dir=args.build_dir,
        results_subdir=args.results_subdir,
        name_flag=args.name_flag,
        zip_name_flag=args.zip_name_flag,
        exclude_flag=args.exclude_flag,
        mesh_split=args.mesh_split,
        radial_cells_flag=args.radial_cells_flag,
        angular_cells_flag=args.angular_cells_flag,
        grading_flag=args.grading_flag,
        inner_radius_flag=args.inner_radius_flag,
        outer_radius_flag=args.outer_radius_flag,
        outer_radius=args.outer_radius,
        unit_symbol=args.unit_symbol,
        output_dir=output_dir / ".generate_snakefile_workdir",
        zip=zip_path,
    )
    run_step("Generating Snakefile", args.verbose, generator.generate, generator_args)

    # Clean up intermediates -- the raw Snakefile/preview parameters.json
    # workdir and the build_hints.json sidecar have both already served
    # their purpose (the sidecar's executable/build-dir info is now baked
    # directly into the zipped Snakefile itself), so only the two requested
    # artifacts are left behind.
    shutil.rmtree(output_dir / ".generate_snakefile_workdir", ignore_errors=True)
    hints_path = output_dir / f"{Path(args.benchmark_filename).stem}.build_hints.json"
    hints_path.unlink(missing_ok=True)

    print(
        f"\nDone -- both artifacts are in {output_dir}/:\n"
        f"  benchmark file: {benchmark_path}\n"
        f"  workflow zip:   {zip_path}\n\n"
        f"Run with:\n"
        f"  cd {output_dir} && python3 run_benchmark.py "
        f"--benchmark-file {args.benchmark_filename} --benchmark-zip {zip_path.name} --result-path ./results"
    )
    return benchmark_path, zip_path


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
