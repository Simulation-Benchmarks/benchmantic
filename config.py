# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
config.py

Optional YAML config-file support for describe_benchmark.py and
workflow.py, so a benchmark's usual flag set doesn't have to be retyped on
every invocation. Pass --config <path.yaml>; a flag also given explicitly
on the command line always overrides the same key in the file -- the file
only supplies defaults for whatever wasn't typed that time.

File format: a flat YAML mapping from flag name (leading "--" optional,
dashes or underscores both accepted) to its value, e.g.:

    module_dir: /Users/sarbani/NFDI_Benchmark/dumux_test/rotating-cylinders
    full-value-params: Radial0
    container-image: git.iws.uni-stuttgart.de:4567/benchmarks/rotating-cylinders:3.1
    container-shared-dir: /dumux/shared
    mesh-split: true
    zip-name-flag: Problem.Name
    semantic-benchmark-src: ../semantic-benchmark
    verbose: true
    review-confidence-threshold: 0.99
    clear-cache: true

Any flag either script accepts can go in the file, including the
"required" ones (module_dir, --container-image, --container-shared-dir) --
those three are the only ones actually enforced as required, and only
after the config file has had a chance to supply them (see
check_required()).

Two caveats, both inherent to how argparse merges an explicit CLI value
with a default rather than anything special to this file:
  - A store_true flag (mesh-split, verbose, clear-cache, skip-review, ...)
    set to `true` in the config can't be forced back to `false` from the
    command line for one run -- there's no CLI syntax for "explicitly
    off" on a store_true flag. Remove it from the config for that run
    instead.
  - An append-action flag (--exclude-flag, --unit-symbol) given as a YAML
    list in the config, and *also* passed on the command line that run,
    gets both: the config's list first, then the CLI values appended
    after it -- not one replacing the other.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def load_config_file(path: Path) -> dict[str, Any]:
    """Load a YAML config file into a flat {dest_name: value} dict (keys
    normalized to underscores, matching argparse's own `dest` convention,
    so they line up 1:1 with a parser's own --flag-name -> flag_name).
    Exits with a clear message (not a traceback) on anything wrong with
    the file itself -- missing PyYAML, unreadable path, invalid YAML, or
    a YAML document that isn't a mapping.
    """
    try:
        import yaml
    except ImportError:
        sys.exit(f"Error: --config requires PyYAML (pip install pyyaml) to read {path}.")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        sys.exit(f"Error: couldn't read config file {path}: {exc}")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        sys.exit(f"Error: {path} is not valid YAML: {exc}")
    if data is None:
        data = {}
    if not isinstance(data, dict):
        sys.exit(
            f"Error: {path} must contain a YAML mapping (key: value pairs), "
            f"not a {type(data).__name__}."
        )
    return {str(k).lstrip("-").replace("-", "_"): v for k, v in data.items()}


def known_dests(*parsers: argparse.ArgumentParser) -> set[str]:
    """Every `dest` name any of the given parsers actually defines
    (excluding "help" and "config" themselves) -- used to catch a typo'd
    config key with a clear error instead of silently ignoring it.
    """
    dests: set[str] = set()
    for parser in parsers:
        for action in parser._actions:  # noqa: SLF001 -- argparse exposes no public accessor for this
            if action.dest in ("help", "config"):
                continue
            dests.add(action.dest)
    return dests


def apply_config(config: dict[str, Any], *parsers: argparse.ArgumentParser, source: Path) -> None:
    """Validate `config`'s keys against what `parsers` collectively
    accept, then set each parser's defaults from it
    (argparse.ArgumentParser.set_defaults()) -- which only takes effect
    for a flag not *also* passed on the command line; an explicit CLI
    value always wins over a default either way, so this is enough to get
    "CLI overrides config" for free rather than needing to merge by hand.
    """
    unknown = set(config) - known_dests(*parsers)
    if unknown:
        sys.exit(
            f"Error: {source}: unknown config key(s): {', '.join(sorted(unknown))}. "
            f"Run the relevant script with --help to see the accepted flag names."
        )
    for parser in parsers:
        parser_dests = {action.dest for action in parser._actions}  # noqa: SLF001
        parser.set_defaults(**{k: v for k, v in config.items() if k in parser_dests})


def peek_config_path(argv: list[str]) -> Path | None:
    """Pull just --config out of argv, ignoring everything else (and
    without erroring on flags this lightweight parser doesn't know about)
    -- so a caller can load+apply it before running its real parse_args().
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args(argv)
    return pre_args.config


def check_required(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    """Some flags (module_dir, --container-image, --container-shared-dir)
    are relaxed from argparse's own required=True/required-positional to
    optional specifically so --config can supply them instead of the
    command line. This is the other half of that: called after the real
    parse_args(), it exits with a clear message -- naming both the CLI
    flag and the config key, since either could have supplied it -- if
    any of `names` is still unset.
    """
    missing = [name for name in names if getattr(args, name, None) in (None, "")]
    if missing:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        keys = ", ".join(missing)
        sys.exit(
            f"Error: missing required value(s): {flags}. Pass them on the command line, "
            f"or set {keys} in a --config file."
        )


def parse_with_config(
    parser: argparse.ArgumentParser, argv: list[str], required: tuple[str, ...] = ()
) -> argparse.Namespace:
    """Drop-in replacement for `parser.parse_args(argv)` that also honors
    a --config file if one is given: peeks --config out of argv, loads
    and applies it (see apply_config()) as defaults before the real
    parse, then checks `required` actually ended up set either way (see
    check_required()). For a single-parser script (describe_benchmark.py
    standalone); workflow.py applies config across its two parsers
    directly instead, since it needs the same file to feed both.
    """
    config_path = peek_config_path(argv)
    if config_path is not None:
        apply_config(load_config_file(config_path), parser, source=config_path)
    args = parser.parse_args(argv)
    check_required(args, required)
    return args
