# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
config.py

The run config: a YAML file passed with --config to workflow.py or
describe_benchmark.py, so a benchmark's settings don't have to be retyped.
A value also given on the command line wins over the file.

Format -- a few plain keys, the same for every software (anything that
depends on the software is in its software description instead):

    implementation: /path/to/rotating-cylinders     # your code (folder or git URL)
    software: DuMux                                 # the simulation software's name
    container: registry.example.org/rotating-cylinders:3.1   # image it runs in
    reference: /path/to/benchmark/1.0.0/minimal-configurations.json   # optional
    llm_model: openai/gpt-oss-120b                  # optional
    review_below_confidence: 0.99                   # optional
    verbose: true                                   # optional

See run-config-template.yaml for every key. Any command-line option also
works as a key, by its name (leading "--" optional, dashes or underscores
both accepted) -- that is how older files are written.

Two caveats, both inherent to how argparse merges an explicit CLI value
with a default rather than anything special to this file:
  - A store_true flag (verbose, clear-cache, skip-review, ...)
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

#: Plain run-config keys -> the command-line option (argparse dest) they set.
PLAIN_KEYS: dict[str, str] = {
    "implementation": "module_dir",
    "software": "software",
    "container": "container_image",
    "reference": "reference_benchmark",
    "llm_model": "model",
    "llm_provider": "provider",
    "review_below_confidence": "review_confidence_threshold",
    "verbose": "verbose",
    "software_description": "software_description",
    "save_software_description": "save_software_description",
}
#: Options only workflow.py has (its later steps); describe_benchmark.py
#: skips them, so one run config works for both.
WORKFLOW_ONLY = {"semantic_benchmark_src", "show_output", "skip_show", "skip_check"}
#: The plain key to name in messages about a missing option.
PLAIN_NAME = {dest: key for key, dest in PLAIN_KEYS.items()}


def translate(data: dict[str, Any], source: Path) -> dict[str, Any]:
    """Run-config keys -> {argparse dest: value}."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        key = str(key).lstrip("-").replace("-", "_")
        if key == "reference" and str(value).lower() in ("none", "no", "false"):
            out["no_reference"] = True
            continue
        dest = PLAIN_KEYS.get(key, key)
        if dest in out and out[dest] != value:
            sys.exit(f"Error: {source}: {key} sets something already set by another key.")
        out[dest] = value
    return out


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
    return translate(data, path)


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
            f"Error: {source}: unknown key(s): {', '.join(sorted(unknown))}. The keys are "
            f"{', '.join(PLAIN_KEYS)} (see run-config-template.yaml); any command-line option also "
            f"works, by its name (python3 describe_benchmark.py --help)."
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
    """Some flags (module_dir, --container-image) are relaxed from argparse's own required=True/required-positional to
    optional specifically so --config can supply them instead of the
    command line. This is the other half of that: called after the real
    parse_args(), it exits with a clear message -- naming both the CLI
    flag and the config key, since either could have supplied it -- if
    any of `names` is still unset.
    """
    missing = [name for name in names if getattr(args, name, None) in (None, "")]
    if missing:
        keys = ", ".join(PLAIN_NAME.get(name, name) for name in missing)
        flags = ", ".join("module_dir" if name == "module_dir" else f"--{name.replace('_', '-')}"
                          for name in missing)
        sys.exit(f"Error: missing {keys} in the run config (or {flags} on the command line).")


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
        values = {k: v for k, v in load_config_file(config_path).items() if k not in WORKFLOW_ONLY}
        apply_config(values, parser, source=config_path)
    args = parser.parse_args(argv)
    check_required(args, required)
    return args
