#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
workflow.py

Runs the full local dev loop in one command:

    describe_benchmark  -->  show_description  -->  verify_description

1. describe_benchmark generates the benchmark file, its dataset-provenance
   sidecar, and the Snakefile (a plain file, not zipped) into
   outputs/<software-name>/.
2. show_description renders the benchmark file (merged with its sibling
   dataset file, if found -- see show_description.py) as Markdown tables,
   so you can eyeball what got extracted.
3. verify_description validates it against what semantic_benchmark actually
   needs (see that script for details), and this script's own exit code
   mirrors its result -- so `workflow.py ... && echo ok` behaves the way
   you'd expect in CI.

All of describe_benchmark's own options (module_dir, --scenario-params,
--container-image, --mesh-split, etc.) are accepted here too and passed
straight through -- run `python3 describe_benchmark.py --help` for the full
list, since this script doesn't duplicate that help text. Two extra
options control the later steps:

    --semantic-benchmark-src PATH   passed through to verify_description
    --show-output PATH              passed through to show_description
                                     (saves the tables to a file instead of
                                     printing them)
    --skip-show / --skip-check      skip either later step
    --config PATH                   YAML file of flag defaults, covering
                                     both this script's own flags above and
                                     everything describe_benchmark accepts
                                     -- see config.py. A flag also given
                                     explicitly on the command line always
                                     overrides the same key in the file.

Usage
-----
    python3 workflow.py <module_dir> \\
        --scenario-params Cells0,Cells1,Grading0,Radial0,Name,Omega1,Omega2 \\
        --full-value-params Radial0 \\
        --container-image git.iws.uni-stuttgart.de:4567/benchmarks/rotating-cylinders:3.1 \\
        --container-shared-dir /dumux/shared \\
        --mesh-split --zip-name-flag Problem.Name \\
        --semantic-benchmark-src ../semantic-benchmark

    # ...or, with the same flags saved in a file (see config.py for the
    # exact key names -- same as the flags above, minus the leading "--"):
    python3 workflow.py --config rotating-cylinders.yaml

Note: -h/--help here shows describe_benchmark's help (since its parser
consumes whatever it recognizes first) -- run the three scripts' own
--help separately for their full option lists.
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
        show_argv = [str(benchmark_path)]
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
