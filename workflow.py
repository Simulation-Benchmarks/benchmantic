#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
workflow.py

The whole check in one command: describe_benchmark.py, then
show_description.py (review.md), then verify_description.py. Its exit code is
the verification's, so `workflow.py ... && publish` is safe in CI.

    python3 workflow.py --config my-benchmark.yaml

Takes every option of describe_benchmark.py, plus:

    --skip-show / --skip-check      leave out show_description / verify_description
    --show-output PATH              where show_description saves its tables
    --semantic-benchmark-src PATH   a semantic-benchmark clone to verify against
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import config  # noqa: E402
import describe_benchmark  # noqa: E402
import verify_description  # noqa: E402
import show_description  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    argv = list(argv if argv is not None else sys.argv[1:])

    # Pull out workflow-only flags first; everything else goes to
    # describe_benchmark's own parser untouched. --config is workflow-only
    # too (peeled here, not passed through) since one config file needs to
    # supply defaults for BOTH this parser and describe_benchmark's --
    # see config.apply_config().
    extra_ap = argparse.ArgumentParser(add_help=False)
    extra_ap.add_argument("--config", type=Path, default=None)
    extra_ap.add_argument("--semantic-benchmark-src", type=Path, default=None)
    extra_ap.add_argument("--show-output", type=Path, default=None)
    extra_ap.add_argument("--skip-show", action="store_true")
    extra_ap.add_argument("--skip-check", action="store_true")

    db_parser = describe_benchmark.build_arg_parser()

    config_path = config.peek_config_path(argv)
    if config_path is not None:
        config.apply_config(config.load_config_file(config_path), extra_ap, db_parser, source=config_path)

    extra_args, remaining = extra_ap.parse_known_args(argv)

    print("=== describe_benchmark ===")
    build_args = db_parser.parse_args(remaining)
    config.check_required(build_args, describe_benchmark.REQUIRED_FLAGS)
    benchmark_path, dataset_path, snakefile_path = describe_benchmark.run(build_args)

    if benchmark_path is None:
        # Snakefile-only mode (--outputs snakefile-only / the "Snakefile
        # only" interactive preset) -- describe_benchmark.py generated only
        # a Snakefile, with no benchmark.jsonld to feed show_description.py
        # or verify_description.py, so there's nothing left for this script
        # to chain into.
        print(
            "\n(Snakefile-only mode: no benchmark.jsonld was generated, so show_description and "
            "verify_description are skipped -- only outputs/<software-name>/Snakefile was produced.)"
        )
        sys.exit(0)

    if not extra_args.skip_show:
        print("\n=== show_description ===")
        # --quiet: describe_benchmark.py already rendered and saved this
        # exact review.md moments ago (silently, same reason -- see its own
        # show_argv comment); re-printing the whole Markdown table here too
        # was just the same wall of text a second time. This call still
        # re-renders and re-saves the file (in case --show-output points
        # somewhere different from the default, or this step is ever run
        # against a benchmark.jsonld from a source other than the
        # describe_benchmark.py call right above), just without echoing it.
        show_argv = [str(benchmark_path), "--quiet"]
        if extra_args.show_output:
            show_argv += ["--output", str(extra_args.show_output)]
        show_description.run(show_description.build_arg_parser().parse_args(show_argv))

    exit_code = 0
    if not extra_args.skip_check:
        print("\n=== verify_description ===")
        check_argv = [str(benchmark_path)]
        if extra_args.semantic_benchmark_src:
            check_argv += ["--semantic-benchmark-src", str(extra_args.semantic_benchmark_src)]
        exit_code = verify_description.run(verify_description.build_arg_parser().parse_args(check_argv))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
