#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
describe_benchmark.py

Describes a benchmark implementation: finds its inputs and output metrics,
asks the LLM what they mean (you review the answers), and writes to
outputs/<software-name>/:

    <name>_benchmark.jsonld   the description: parameters, configurations, metrics
    <name>_dataset.jsonld     authors, license, software dependencies
    <name>_mapping.json       the benchmark's parameters -> the software's inputs
    Snakefile                 runs one configuration in the container
    review.md                 everything above as Markdown tables

--outputs picks which of these (snakefile-only: just the Snakefile, no LLM).
Every option also works as a key of a run config (--config). Without
--scenario-params you are asked which inputs differ between configurations,
and without --skip-review you review the LLM's answers -- both need a terminal.

    python3 describe_benchmark.py --config my-benchmark.yaml
    python3 describe_benchmark.py path/or/link --software DuMux --container-image IMAGE
"""

from __future__ import annotations

import argparse
import os
import contextlib
import io
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import config  # noqa: E402
import metadata.builder as builder  # noqa: E402
import show_description  # noqa: E402
import snakefile.generator as generator  # noqa: E402
import snakefile.styles as snakefile_styles  # noqa: E402
from metadata.graph import load_graph  # noqa: E402

REQUIRED_FLAGS = ("module_dir", "container_image")


def _slugify_benchmark_label(by_id: dict) -> str:
    """Slugify the root entity's "name" (the crate label, e.g. "rotating
    cylinders" -> "rotating_cylinders" -- see metadata.builder's
    GraphBuilder.add_rocrate_root()). Shared by derive_benchmark_filename()
    and derive_dataset_filename() so both files derived from the same run
    always agree on the base name.
    """
    root = by_id.get("./") or {}
    label = root.get("name") or "benchmark"
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", label).strip("_").lower()
    return slug or "benchmark"


def derive_benchmark_filename(by_id: dict) -> str:
    """"<slug>_benchmark.jsonld", used as --benchmark-filename's default
    when it isn't given explicitly. Mirrors generator.derive_software_name's
    approach for the same reason: the name only exists after
    metadata.builder has already run, not before.

    Doesn't double up the "_benchmark" suffix if the label already contains
    the word "benchmark" (e.g. a label like "rotating cylinders benchmark"
    becomes "rotating_cylinders_benchmark.jsonld", not
    "..._benchmark_benchmark.jsonld").
    """
    slug = _slugify_benchmark_label(by_id)
    if "benchmark" in slug:
        return f"{slug}.jsonld"
    return f"{slug}_benchmark.jsonld"


def mapping_filename_for(benchmark_filename: str) -> str:
    """'rotating_cylinders_benchmark.jsonld' -> 'rotating_cylinders_mapping.json'."""
    stem = Path(benchmark_filename).stem
    stem = stem[: -len("_benchmark")] if stem.endswith("_benchmark") else stem
    return f"{stem}_mapping.json"


def derive_dataset_filename(by_id: dict) -> str:
    """"<slug>_dataset.jsonld" -- the sidecar file holding author/publisher/
    dependency info (see metadata.builder's GraphBuilder.build_dataset_graph()),
    written alongside the benchmark file. Same slug and same
    don't-double-the-suffix rule as derive_benchmark_filename() above, just
    with "dataset" as the suffix/check word instead of "benchmark".
    """
    slug = _slugify_benchmark_label(by_id)
    if "dataset" in slug:
        return f"{slug}.jsonld"
    return f"{slug}_dataset.jsonld"


def run_step(label: str, verbose: bool, fn, *fn_args, **fn_kwargs):
    """Call `fn` (an in-process build step) and return whatever it returns.
    By default, stays quiet -- prints one short line, and only dumps the
    step's captured stdout+stderr if it actually fails (raises). Pass
    --verbose to always show it live instead. Mirrors the original
    subprocess-based capture_output=True behavior now that these run
    in-process instead.
    """
    print(f"-> {label}...")
    if verbose:
        return fn(*fn_args, **fn_kwargs)

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            result = fn(*fn_args, **fn_kwargs)
    except SystemExit as exc:
        sys.stdout.write(buf.getvalue())
        sys.exit(f"Error: {label} failed -- {exc.code}")
    except Exception:
        sys.stdout.write(buf.getvalue())
        raise
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    builder.add_build_arguments(ap, module_dir_optional=True)
    ap.add_argument("--config", type=Path, default=None,
                    help="Run config (YAML): its keys are the defaults of these options; the command line wins.")
    out = ap.add_argument_group("output files")
    out.add_argument("--software-name", default=None,
                     help="Write to outputs/<software-name>/ (default: the software, e.g. 'dumux').")
    out.add_argument("--benchmark-filename", default=None,
                     help="Name of the benchmark description (default: <benchmark name>_benchmark.jsonld).")
    out.add_argument("--dataset-filename", default=None,
                     help="Name of the dataset description (default: <benchmark name>_dataset.jsonld).")

    snake = ap.add_argument_group("the Snakefile")
    snake.add_argument("--container-image", default=None,
                       help="The container image the simulation runs in (required; or `container:` in --config).")
    snake.add_argument("--container-shared-dir", default=None,
                       help="Folder in the container where results are saved (default: the software description's).")
    snake.add_argument("--executable", default=None, help="The program to run (default: found in CMakeLists.txt).")
    snake.add_argument("--build-dir", default=None, help="Folder in the container to run it in.")
    snake.add_argument("--results-subdir", default="results", help="Subfolder for each configuration's results.")
    snake.add_argument("--unit-symbol", action="append", default=None, metavar="QUDT_UNIT=SYMBOL",
                       help="Unit suffix of parameters.json keys, e.g. unit:M=m (repeatable).")
    snakefile_styles.add_arguments(ap)

    return ap


def _print_intro() -> None:
    """One-time "here's what this tool does" banner shown before a real
    interactive run starts (never in CI/--skip-review runs, where it'd
    just be unwanted noise ahead of a script's own log parsing). Purely
    orientation for a first-time user -- everything after this point is
    the actual pipeline.
    """
    print(
        "\nbenchmantic\n"
        "Semantic benchmark description and execution tool\n\n"
        "This tool analyzes your simulation benchmark and produces:\n"
        "  • semantic benchmark metadata\n"
        "  • a machine-readable dataset description\n"
        "  • a Snakefile for reproducible execution\n"
        "  • a human-readable review report\n\n"
        "Let's get started."
    )
    try:
        input("\nPress Enter to continue...")
    except (EOFError, KeyboardInterrupt):
        print()


def run(args: argparse.Namespace) -> tuple[Path | None, Path | None, Path | None]:
    """Run the full build (semantic description, plus whichever of the
    dataset sidecar / Snakefile / review report were selected in the
    Outputs step -- see metadata.builder's _resolve_outputs_selection()).
    Returns (benchmark_path, dataset_path, snakefile_path); any of the
    three is None if that artifact wasn't generated this run. In
    particular, benchmark_path IS allowed to be None -- this happens when
    the resolved Outputs selection has "description": False (the
    "snakefile-only" preset/--outputs value): no benchmark.jsonld, dataset
    sidecar, or review report is generated at all in that mode, only the
    Snakefile. Callers (e.g. workflow.py) that chain into
    show_description.py/verify_description.py using benchmark_path MUST
    check for None first and skip that chain instead of passing None
    through.
    """
    if getattr(args, "llm_api_key", None):
        os.environ["LLM_API_KEY"] = args.llm_api_key
    needs_interactive = args.scenario_params is None or not args.skip_review
    if needs_interactive and not (args.verbose or args.debug):
        try:
            is_tty = sys.stdin.isatty()
        except Exception:
            is_tty = False
        if is_tty:
            _print_intro()

    staging_dir = Path(tempfile.mkdtemp(prefix="describe_benchmark_"))
    # Always stage under a fixed name -- the *final* filename may itself
    # depend on reading this file back (see derive_benchmark_filename()
    # below), so it can't be used as the staging name too.
    staged_benchmark = staging_dir / "staged.jsonld"

    # 1. Semantic description -- generated to a staging location first,
    # since the output directory's default (and, unless --benchmark-filename
    # was given explicitly, the benchmark filename itself) depends on
    # reading it back: the software name and the benchmark's own name both
    # live in the graph itself, not something known beforehand.
    builder_dests = {a.dest for a in builder.build_arg_parser()._actions if a.dest != "help"}  # noqa: SLF001
    builder_args = argparse.Namespace(**{d: getattr(args, d) for d in builder_dests if hasattr(args, d)})
    builder_args.output = staged_benchmark
    builder_args.validate_severity = "REQUIRED"
    builder_args.verbose = args.verbose or args.debug
    # Two sub-steps inside metadata.builder.build() need a real terminal
    # (they call input()), so their output can't be silently captured into
    # the quiet-mode buffer the way the rest of this step's output can:
    # the parameter-selection prompt (only when --scenario-params wasn't
    # supplied), and the interactive metadata review step (unless
    # --skip-review was passed). Force live output for the whole step
    # whenever either applies, regardless of --verbose, so prompts (and
    # what you type) actually show up. (`needs_interactive` was already
    # computed above, before the intro banner.)
    if needs_interactive and not (args.verbose or args.debug):
        print("   (interactive parameter selection and/or metadata review is enabled -- showing this step live)")
    stats = run_step(
        "Generating semantic description", args.verbose or args.debug or needs_interactive,
        builder.build, builder_args,
    )
    stats = stats or {}
    # The mapping check results matter even in a quiet run (e.g. a
    # benchmark parameter this implementation ignores) -- the builder's own
    # output is only shown live in verbose/interactive runs.
    if stats.get("mapping_notes") and not (args.verbose or args.debug or needs_interactive):
        print("\n".join(stats["mapping_notes"]))

    staged_by_id = load_graph(staged_benchmark)
    software_name = args.software_name or generator.derive_software_name(staged_by_id)
    output_dir = Path("outputs") / software_name

    # Which artifacts this run actually produces -- resolved by
    # metadata.builder's Outputs step and handed back via `stats["outputs"]`
    # (a plain dict of {"description": bool, "dataset": bool, "snakefile":
    # bool}). Falls back to "everything" if build() didn't return stats for
    # some reason (shouldn't happen on a successful run, but keeps this
    # defensive rather than crashing on a missing key).
    outputs = stats.get("outputs") or dict(builder.DEFAULT_OUTPUTS)

    if not outputs.get("description", True):
        # Snakefile-only mode: metadata.builder already skipped semantic
        # inference and wrote only a minimal, non-deliverable staging graph
        # to `staged_benchmark` (see its _generate_snakefile_only()) -- it's
        # used directly as snakefile.generator's input below and is NEVER
        # moved/renamed into output_dir as a "<name>_benchmark.jsonld"; no
        # dataset sidecar or review.md is produced either. See run()'s
        # docstring for why benchmark_path/dataset_path come back None.
        print(
            "\nSnakefile-only mode: skipping semantic inference (no Groq/OpenAI call), the "
            "benchmark description, and the review report. Parameters use bare parameters.json "
            "keys since units aren't inferred in this mode -- see README.md's Config files "
            "section if any selected parameter's real unit is meters or pascals."
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        workdir = output_dir / ".generate_snakefile_workdir"
        generator_args = generator.namespace_for(
            args, metadata_jsonld=staged_benchmark, software_name=software_name, output_dir=workdir, zip=None,
        )
        run_step("Generating Snakefile", args.verbose or args.debug, generator.generate, generator_args)
        snakefile_path = output_dir / "Snakefile"
        shutil.move(str(workdir / "Snakefile"), str(snakefile_path))
        shutil.rmtree(workdir, ignore_errors=True)
        shutil.rmtree(staging_dir, ignore_errors=True)

        benchmark_label = (staged_by_id.get("./") or {}).get("name") or software_name
        run_cmd = (
            f"# Snakefile-only mode produced no benchmark.jsonld / run_benchmark.py isn't usable here.\n"
            f"# Run the Snakefile directly (see README.md), or rerun with a different --outputs preset "
            f"for a full benchmark.jsonld + run_benchmark.py path."
        )
        _print_summary_box(benchmark_label, stats, output_dir, [snakefile_path], run_cmd)
        return None, None, snakefile_path

    benchmark_filename = args.benchmark_filename or derive_benchmark_filename(staged_by_id)
    dataset_filename = args.dataset_filename or derive_dataset_filename(staged_by_id)
    if args.verbose or args.debug:
        print(f"-> Software: {software_name!r} (output directory: {output_dir}/)")
        if not args.benchmark_filename:
            print(f"-> Benchmark filename: {benchmark_filename!r} (derived from the benchmark's name)")
        if not args.dataset_filename:
            print(f"-> Dataset filename: {dataset_filename!r} (derived from the benchmark's name)")

    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = output_dir / benchmark_filename
    dataset_path: Path | None = None

    shutil.move(str(staged_benchmark), str(benchmark_path))

    # Dataset sidecar (author, publisher, software dependencies -- see
    # metadata.builder.GraphBuilder.build_dataset_graph()) -- written by
    # metadata.builder.build() as "staged.dataset.jsonld" next to the staged
    # benchmark file, ONLY when "dataset" was selected in the Outputs step.
    # Patch in the "schema:isPartOf" link back to the benchmark file's
    # *final* name (only known now, not when that sidecar was written)
    # before moving it into place.
    staged_dataset = staging_dir / "staged.dataset.jsonld"
    if outputs.get("dataset", True) and staged_dataset.exists():
        dataset_path = output_dir / dataset_filename
        dataset_doc = json.loads(staged_dataset.read_text(encoding="utf-8"))
        for node in dataset_doc.get("@graph", []):
            if node.get("@id") == "./":
                node["schema:isPartOf"] = {"@id": benchmark_filename}
                break
        dataset_path.write_text(json.dumps(dataset_doc, indent=2), encoding="utf-8")
        if args.verbose or args.debug:
            print(f"-> Wrote {dataset_path}")

    staged_hints = staging_dir / "staged.build_hints.json"
    hints_path = output_dir / f"{Path(benchmark_filename).stem}.build_hints.json"
    if staged_hints.exists():
        shutil.move(str(staged_hints), str(hints_path))

    # Input mapping (software-neutral benchmark parameters -> this
    # software's inputs; see metadata.mapping) -- a deliverable next to the
    # description, and what the Snakefile generator reads.
    staged_mapping = staging_dir / "staged.mapping.json"
    mapping_path: Path | None = None
    if staged_mapping.exists():
        mapping_path = output_dir / mapping_filename_for(benchmark_filename)
        shutil.move(str(staged_mapping), str(mapping_path))
        if args.verbose or args.debug:
            print(f"-> Wrote {mapping_path}")
    shutil.rmtree(staging_dir, ignore_errors=True)

    # 2. Snakefile -- generated into a throwaway workdir alongside the
    # per-case preview parameters.json files (neither of those is part of
    # the deliverable), then just the Snakefile itself is moved into
    # output_dir as a plain file -- no zip packaging. Only when
    # "snakefile" was selected in the Outputs step.
    snakefile_path: Path | None = None
    if outputs.get("snakefile", True):
        workdir = output_dir / ".generate_snakefile_workdir"
        generator_args = generator.namespace_for(
            args, metadata_jsonld=benchmark_path, software_name=software_name, output_dir=workdir, zip=None,
        )
        run_step("Generating Snakefile", args.verbose or args.debug, generator.generate, generator_args)

        snakefile_path = output_dir / "Snakefile"
        shutil.move(str(workdir / "Snakefile"), str(snakefile_path))
        shutil.rmtree(workdir, ignore_errors=True)

    # build_hints.json is only ever an input to Snakefile generation (its
    # executable/build-dir info gets baked directly into the Snakefile) --
    # whether or not that step ran, it's implementation detail that
    # shouldn't be left behind as a fourth file in output_dir.
    hints_path.unlink(missing_ok=True)

    # 3. Review report -- show_description.py, called in-process the same
    # way workflow.py normally chains it after this script. Always
    # generated, same as the benchmark description itself -- not one of
    # the Outputs step's choices (only "dataset" and "snakefile" are).
    # Auto-discovers the sidecar dataset file next to benchmark_path if
    # one was generated above; gracefully omits those sections if not, so
    # a run without the dataset sidecar still gets a review report, just
    # covering the benchmark's own parameters and metrics.
    review_path: Path | None = output_dir / "review.md"
    # --quiet unconditionally (not just when non-verbose): the full
    # Markdown table is about to be saved to review.md and then listed
    # again in the "Generated files" summary this run prints at the end,
    # so dumping the whole thing to the terminal here too -- on top of
    # workflow.py's own separate show_description step potentially doing
    # the same right after -- was mostly just noise to scroll past
    # regardless of verbosity. --verbose/--debug still show run_step's own
    # captured-output-on-failure behavior and every other step's detail;
    # this table specifically just isn't the kind of thing either flag was
    # meant to surface, since it's not diagnostic info, it's the finished
    # artifact's own content, already written to disk.
    show_args = show_description.build_arg_parser().parse_args([str(benchmark_path), "--quiet"])
    run_step("Generating review report", args.verbose or args.debug, show_description.run, show_args)

    benchmark_label = (staged_by_id.get("./") or {}).get("name") or software_name
    run_cmd = (
        f"cd {output_dir} && python3 run_benchmark.py "
        f"--benchmark-file {benchmark_filename} --result-path ./results"
    )
    artifact_paths = [p for p in (benchmark_path, dataset_path, mapping_path, snakefile_path, review_path)
                      if p is not None]
    _print_summary_box(benchmark_label, stats, output_dir, artifact_paths, run_cmd)
    return benchmark_path, dataset_path, snakefile_path


def _print_summary_box(
    benchmark_label: str,
    stats: dict,
    output_dir: Path,
    artifact_paths: list[Path],
    run_cmd: str,
) -> None:
    """Final "done" state -- a boxed status summary instead of the raw
    "Wrote .../Wrote .../Done" artifact dump every prior step already
    implied. `stats` comes back from metadata.builder.build() (see its
    docstring) -- parameter/metric/case counts and the RO-Crate validation
    outcome, so this doesn't have to re-derive anything by re-parsing the
    JSON-LD just written.
    """
    banner = "✓ Benchmark generated successfully"
    width = min(78, max(len(banner) + 4, 48))
    top = "╭" + "─" * (width - 2) + "╮"
    bottom = "╰" + "─" * (width - 2) + "╯"
    print(f"\n{top}")
    print(f"│ {banner:<{width - 4}} │")
    print(bottom)

    print(f"\nBenchmark\n  {benchmark_label}")

    rocrate_passed = stats.get("rocrate_validation_passed")
    if rocrate_passed is None:
        rocrate_line = "○ RO-Crate 1.1 (skipped)"
    elif rocrate_passed:
        rocrate_line = "✓ RO-Crate 1.1"
    else:
        rocrate_line = "✗ RO-Crate 1.1 -- see --verbose for the failing checks"

    print(
        f"\nSemantic description\n"
        f"  {stats.get('parameters', '?')} parameters\n"
        f"  {stats.get('metrics', '?')} metrics\n"
        f"  {stats.get('cases', '?')} benchmark case(s)"
    )
    print(f"\nValidation\n  {rocrate_line}")

    print(f"\nGenerated files\n  {output_dir}/")
    for p in artifact_paths:
        print(f"    {p.name}")

    print(f"\nNext step:\n\n  {run_cmd}\n")


def main(argv: list[str] | None = None) -> None:
    args = config.parse_with_config(build_arg_parser(), list(argv if argv is not None else sys.argv[1:]), REQUIRED_FLAGS)
    run(args)


if __name__ == "__main__":
    main()
