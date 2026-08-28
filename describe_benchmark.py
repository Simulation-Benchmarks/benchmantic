#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
describe_benchmark.py

The only entry point for this package. Generates the semantic benchmark
description, its dataset-provenance sidecar, and a Snakefile from a
benchmark module, calling metadata.builder and snakefile.generator directly
in-process (not via subprocess), and writes all three artifacts together
into outputs/<software-name>/, matching this pattern from run-benchmark.yml:

    cd $GITHUB_WORKSPACE/outputs/<software-name>
    python3 run_benchmark.py \\
        --benchmark-file <benchmark-filename> \\
        --result-path ./results

Produces:
    outputs/<software-name>/<benchmark-filename>  -- metadata.builder's
                                                       RO-Crate JSON-LD
                                                       (parameters, metrics,
                                                       processing steps),
                                                       loadable by
                                                       semantic_benchmark.BenchmarkLoader
    outputs/<software-name>/<dataset-filename>     -- sidecar dataset-
                                                       provenance document
                                                       (author, publisher,
                                                       software
                                                       dependencies) -- see
                                                       metadata.builder's
                                                       GraphBuilder.build_dataset_graph()
    outputs/<software-name>/Snakefile              -- snakefile.generator's
                                                       Snakefile, as a plain
                                                       file (not zipped)

Not tied to any one simulation tool: <module_dir> just needs to be
something metadata.builder can already handle (a DUNE/DuMux-style module
with main.cc/problem.hh/params.input, or a repo checkout containing one) --
run it once per tool (DuMux, OpenFOAM, ...) with that tool's own
--software-name, --container-image, and mesh-split settings.

The output directory is always outputs/<software-name>/ (--software-name,
or if that's not given, the software name derived from the generated
benchmark file itself -- its local:software label, slugified, e.g. "DuMux"
-> "dumux"). Since the software name (and, unless --benchmark-filename/
--dataset-filename were given, both generated filenames) only exist after
metadata.builder has run, this script generates the benchmark file and its
dataset sidecar to a staging location first, derives the names from it if
needed, then moves all three artifacts into outputs/<software-name>/
together.

By default, output is quiet -- one short line per step, with the full
underlying output only shown if something fails. Pass --verbose (-v) to
always see it live. Exception: if --scenario-params isn't given, or unless
--skip-review is passed, the interactive parameter-selection prompt and/or
metadata review step always run live (both need a real terminal for
input()), regardless of --verbose.

After AI inference, each discovered parameter/metric is shown in a table
for review -- edit any field, loop until you confirm it's final, then the
(possibly corrected) result is what gets cached and built into
benchmark.jsonld. Pass --skip-review to accept the AI's output as-is (e.g.
for CI). Anything you correct is also remembered across benchmarks (see
ai.corrections) and offered back to the AI as guidance next time it infers
a similarly named/typed item.

Usage
-----
    python3 describe_benchmark.py <module_dir> \\
        --scenario-params Cells0,Cells1,Grading0,Radial0,Name,Omega1,Omega2 \\
        --full-value-params Radial0 \\
        --container-image git.iws.uni-stuttgart.de:4567/benchmarks/rotating-cylinders:3.1 \\
        --container-shared-dir /dumux/shared \\
        --mesh-split --zip-name-flag Problem.Name

--software-name isn't needed above -- it's derived automatically from the
generated benchmark file (e.g. "DuMux" -> "dumux"). Pass it explicitly only
to override that derived name.

<module_dir> can be the exact benchmark folder, a repo checkout, or a
GitHub/GitLab/any git URL -- metadata.builder resolves it the same way it
always does (see metadata/repo_source.py for the URL-clone case).
"""

from __future__ import annotations

import argparse
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

import metadata.builder as builder  # noqa: E402
import snakefile.generator as generator  # noqa: E402
from metadata.graph import load_graph  # noqa: E402


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
    ap.add_argument("module_dir", type=str,
                     help="Path to the benchmark module (or a repo checkout containing it) -- "
                          "anything metadata.builder can already handle. Also accepts a GitHub/GitLab/"
                          "any git-reachable URL (https://..., git://..., or an scp-like git@host:path "
                          "spec); by default it's cloned into a throwaway temp directory that's deleted "
                          "once this run finishes -- see --keep-clone/--ref/--clone-dir/--fresh-clone.")
    ap.add_argument("--ref", type=str, default=None,
                     help="Branch, tag, or commit to check out when module_dir is a git URL. Ignored for "
                          "a local module_dir. Default: the remote's default branch.")
    ap.add_argument("--keep-clone", action="store_true",
                     help="If module_dir is a git URL, clone it into a persistent local cache directory "
                          "instead of the default throwaway temp clone, so a later run against the same "
                          "URL reuses it (fetch + checkout in place) along with its inference cache, "
                          "instead of re-querying the LLM for everything every time. Implied by "
                          "--clone-dir. Ignored for a local module_dir.")
    ap.add_argument("--clone-dir", type=Path, default=None,
                     help="Explicit directory to clone module_dir into, when it's a git URL, instead of "
                          "either the default throwaway temp clone or --keep-clone's default persistent "
                          "cache location (repo_source.DEFAULT_CLONE_ROOT, override-able via the "
                          "BENCHMANTIC_REPO_CACHE env var). Implies --keep-clone. Ignored for a local "
                          "module_dir.")
    ap.add_argument("--fresh-clone", action="store_true",
                     help="With --keep-clone/--clone-dir: if a cached clone already exists at the target "
                          "location, delete it and clone from scratch instead of fetching/checking it out "
                          "in place. Has no effect on the default throwaway clone (already always fresh). "
                          "Ignored for a local module_dir.")
    ap.add_argument("--software-name", default=None,
                     help="Software name -- all three artifacts (benchmark file, dataset file, Snakefile) are "
                          "written together to outputs/<software-name>/, and it's passed through as "
                          "snakefile.generator's --software-name. Default: derived from the generated "
                          "benchmark file itself (its local:software label, slugified, e.g. 'DuMux' -> 'dumux').")
    ap.add_argument("--benchmark-filename", default=None,
                     help="Filename for the generated semantic description (parameters, metrics, processing "
                          "steps -- what semantic_benchmark.BenchmarkLoader reads). Default: derived from the "
                          "benchmark's own name (the root entity's 'name' field, e.g. 'rotating cylinders'), "
                          "slugified, as '<slug>_benchmark.jsonld' -- or just '<slug>.jsonld' if the name "
                          "already contains the word 'benchmark'. Pass this explicitly to force a specific "
                          "name instead, e.g. if your run_benchmark.py --benchmark-file argument is hardcoded "
                          "to 'benchmark.jsonld' or 'benchmark.json'.")
    ap.add_argument("--dataset-filename", default=None,
                     help="Filename for the sidecar dataset-provenance file (author, publisher, software "
                          "dependencies -- see metadata.builder's GraphBuilder.build_dataset_graph()), written "
                          "alongside the benchmark file. Default: same slug as --benchmark-filename's default, "
                          "as '<slug>_dataset.jsonld' (or '<slug>.jsonld' if the name already contains "
                          "'dataset').")

    # --- metadata.builder pass-through ---
    meta = ap.add_argument_group("metadata.builder options")
    meta.add_argument("--main-cc", type=Path, default=None)
    meta.add_argument("--scenario-params", type=str, default=None)
    meta.add_argument("--full-value-params", type=str, default=None)
    meta.add_argument("--provider", type=str, default=None)
    meta.add_argument("--model", type=str, default=None)
    meta.add_argument("--clear-cache", action="store_true")
    meta.add_argument("--fallback-on-error", action="store_true")
    meta.add_argument("--skip-review", action="store_true",
                       help="Skip the interactive review/edit step after AI inference (required for "
                            "non-interactive/CI runs -- it needs a real terminal for input()).")
    meta.add_argument("--review-confidence-threshold", type=float, default=None,
                       help="Mark parameters/metrics below this confidence with a '!' in the review table "
                            "so they're easy to spot before pressing Enter to accept (default: see "
                            "metadata.builder's own default). Doesn't block accepting on its own -- Enter "
                            "always accepts the table as shown. Has no effect with --skip-review.")
    meta.add_argument("--inference-batch-size", type=int, default=None,
                       help="Max parameters/metrics sent to the LLM in a single inference request "
                            "(default: see metadata.builder's own default). A benchmark with more "
                            "not-yet-cached items than this is split into multiple independent requests -- "
                            "lower this if you're hitting a provider's per-request token/payload limit.")
    meta.add_argument("--inference-tpm-budget", type=int, default=None,
                       help="Target tokens-per-minute budget for the inference step (default: see "
                            "metadata.builder's own default). Every request's estimated cost is kept under "
                            "this (splitting batches further if needed) and reserved against a shared budget "
                            "across the whole run, not just checked per-request. Set it well below your "
                            "account's actual TPM cap, not equal to it -- e.g. 8,000-9,000 on a 12,000 TPM "
                            "account.")
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


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Run the full build (semantic description + dataset sidecar +
    Snakefile). Returns (benchmark_path, dataset_path, snakefile_path) so
    callers (e.g. workflow.py) can chain into show_description.py/
    verify_description.py without re-deriving the output location
    themselves.
    """

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
    builder_args = argparse.Namespace(
        module_dir=args.module_dir,
        ref=args.ref,
        clone_dir=args.clone_dir,
        fresh_clone=args.fresh_clone,
        keep_clone=args.keep_clone,
        main_cc=args.main_cc,
        output=staged_benchmark,
        scenario_params=args.scenario_params,
        full_value_params=args.full_value_params,
        provider=args.provider or builder.DEFAULT_PROVIDER,
        model=args.model,
        fallback_on_error=args.fallback_on_error,
        verbose=args.verbose,
        clear_cache=args.clear_cache,
        skip_review=args.skip_review,
        review_confidence_threshold=(
            args.review_confidence_threshold
            if args.review_confidence_threshold is not None
            else builder.review.DEFAULT_CONFIDENCE_THRESHOLD
        ),
        inference_batch_size=(
            args.inference_batch_size
            if args.inference_batch_size is not None
            else builder.DEFAULT_BATCH_SIZE
        ),
        inference_tpm_budget=(
            args.inference_tpm_budget
            if args.inference_tpm_budget is not None
            else builder.DEFAULT_TPM_BUDGET
        ),
        skip_validation=args.skip_validation,
        validate_severity="REQUIRED",
    )
    # Two sub-steps inside metadata.builder.build() need a real terminal
    # (they call input()), so their output can't be silently captured into
    # the quiet-mode buffer the way the rest of this step's output can:
    # the parameter-selection prompt (only when --scenario-params wasn't
    # supplied), and the interactive metadata review step (unless
    # --skip-review was passed). Force live output for the whole step
    # whenever either applies, regardless of --verbose, so prompts (and
    # what you type) actually show up.
    needs_interactive = args.scenario_params is None or not args.skip_review
    if needs_interactive and not args.verbose:
        print("   (interactive parameter selection and/or metadata review is enabled -- showing this step live)")
    run_step("Generating semantic description", args.verbose or needs_interactive, builder.build, builder_args)

    staged_by_id = load_graph(staged_benchmark)
    software_name = args.software_name or generator.derive_software_name(staged_by_id)
    benchmark_filename = args.benchmark_filename or derive_benchmark_filename(staged_by_id)
    dataset_filename = args.dataset_filename or derive_dataset_filename(staged_by_id)
    output_dir = Path("outputs") / software_name
    print(f"-> Software: {software_name!r} (output directory: {output_dir}/)")
    if not args.benchmark_filename:
        print(f"-> Benchmark filename: {benchmark_filename!r} (derived from the benchmark's name)")
    if not args.dataset_filename:
        print(f"-> Dataset filename: {dataset_filename!r} (derived from the benchmark's name)")

    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = output_dir / benchmark_filename
    dataset_path = output_dir / dataset_filename

    shutil.move(str(staged_benchmark), str(benchmark_path))

    # Dataset sidecar (author, publisher, software dependencies -- see
    # metadata.builder.GraphBuilder.build_dataset_graph()) -- written by
    # metadata.builder.build() as "staged.dataset.jsonld" next to the staged
    # benchmark file. Patch in the "schema:isPartOf" link back to the
    # benchmark file's *final* name (only known now, not when that sidecar
    # was written) before moving it into place.
    staged_dataset = staging_dir / "staged.dataset.jsonld"
    if staged_dataset.exists():
        dataset_doc = json.loads(staged_dataset.read_text(encoding="utf-8"))
        for node in dataset_doc.get("@graph", []):
            if node.get("@id") == "./":
                node["schema:isPartOf"] = {"@id": benchmark_filename}
                break
        dataset_path.write_text(json.dumps(dataset_doc, indent=2), encoding="utf-8")
        print(f"-> Wrote {dataset_path}")

    staged_hints = staging_dir / "staged.build_hints.json"
    if staged_hints.exists():
        shutil.move(str(staged_hints), str(output_dir / f"{Path(benchmark_filename).stem}.build_hints.json"))
    shutil.rmtree(staging_dir, ignore_errors=True)

    # 2. Snakefile -- generated into a throwaway workdir alongside the
    # per-case preview parameters.json files (neither of those is part of
    # the deliverable), then just the Snakefile itself is moved into
    # output_dir as a plain file -- no zip packaging.
    workdir = output_dir / ".generate_snakefile_workdir"
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
        output_dir=workdir,
        zip=None,
    )
    run_step("Generating Snakefile", args.verbose, generator.generate, generator_args)

    snakefile_path = output_dir / "Snakefile"
    shutil.move(str(workdir / "Snakefile"), str(snakefile_path))

    # Clean up intermediates -- the workdir's per-case preview
    # parameters.json files and the build_hints.json sidecar have both
    # already served their purpose (the sidecar's executable/build-dir info
    # is now baked directly into the Snakefile itself), so only the three
    # requested artifacts are left behind.
    shutil.rmtree(workdir, ignore_errors=True)
    hints_path = output_dir / f"{Path(benchmark_filename).stem}.build_hints.json"
    hints_path.unlink(missing_ok=True)

    print(
        f"\nDone -- all artifacts are in {output_dir}/:\n"
        f"  benchmark file: {benchmark_path}\n"
        f"  dataset file:   {dataset_path}\n"
        f"  Snakefile:      {snakefile_path}\n\n"
        f"Run with:\n"
        f"  cd {output_dir} && python3 run_benchmark.py "
        f"--benchmark-file {benchmark_filename} --result-path ./results"
    )
    return benchmark_path, dataset_path, snakefile_path


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
