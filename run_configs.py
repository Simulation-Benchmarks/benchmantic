#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
run_configs.py

Runs workflow.py once for every run config in a folder, one after another.
Run it in the folder with the configs (it can sit there too):

    cd /Users/sarbani/NFDI_Benchmark/configs_for_benchmantic
    python3 run_configs.py

is the same as running

    python3 workflow.py --config <folder>/<file>.yaml

for each run config there. Files that are not run configs (no
`implementation:` key, e.g. a software description kept next to them) are
skipped. A failing run doesn't stop the others; a summary at the end lists
how each went, and the exit code is non-zero if any failed.

By default every run uses this terminal, so you can answer the parameter
selection and the reviews as usual. With --logs DIR the runs don't ask
anything (--skip-review is passed) and each one's output goes to
DIR/<config name>.log instead -- for running all of them unattended.

    python3 run_configs.py <folder> --only "rotating*"    only some of them
    python3 run_configs.py <folder> --dry-run             show what would run
    python3 run_configs.py <folder> --stop-on-error       stop at the first failure
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: Where benchmantic is when this script isn't next to its workflow.py
#: (override with --workflow, or the BENCHMANTIC_HOME environment variable).
BENCHMANTIC_HOME = Path("/Users/sarbani/NFDI_Benchmark/benchmantic")


def default_workflow() -> Path:
    """benchmantic's workflow.py: next to this script, else in
    $BENCHMANTIC_HOME, else in BENCHMANTIC_HOME above."""
    import os

    for folder in (HERE, Path(os.environ.get("BENCHMANTIC_HOME", BENCHMANTIC_HOME)).expanduser()):
        if (folder / "workflow.py").is_file():
            return folder / "workflow.py"
    return BENCHMANTIC_HOME / "workflow.py"


def is_run_config(path: Path) -> bool:
    """A run config names the implementation (`implementation:`, or the
    older `module_dir:`); software descriptions next to it don't."""
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- not readable YAML: not a run config
        return False
    return isinstance(data, dict) and any(k in data for k in ("implementation", "module_dir", "module-dir"))


def find_configs(folder: Path, only: list[str]) -> tuple[list[Path], list[Path]]:
    """(run configs, other YAML files) in `folder`, sorted by name."""
    files = sorted(p for p in folder.iterdir() if p.suffix in (".yaml", ".yml") and p.is_file())
    if only:
        files = [p for p in files if any(fnmatch.fnmatch(p.name, pattern) or fnmatch.fnmatch(p.stem, pattern)
                                         for pattern in only)]
    configs = [p for p in files if is_run_config(p)]
    return configs, [p for p in files if p not in configs]


def run_one(config: Path, workflow: Path, logs: Path | None, extra: list[str]) -> tuple[int, float]:
    command = [sys.executable, str(workflow), "--config", str(config), *extra]
    start = time.monotonic()
    if logs is None:                                  # in this terminal: prompts and reviews work as usual
        code = subprocess.run(command, cwd=workflow.parent).returncode
    else:
        logs.mkdir(parents=True, exist_ok=True)
        with open(logs / f"{config.stem}.log", "w", encoding="utf-8") as log:
            code = subprocess.run([*command, "--skip-review"], cwd=workflow.parent, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT).returncode
    return code, time.monotonic() - start


def _duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, nargs="?", default=Path.cwd(),
                    help="The folder with the run configs (default: the current folder).")
    ap.add_argument("--only", action="append", default=[], metavar="PATTERN",
                    help="Run only configs whose file name matches, e.g. 'rotating*' (repeatable).")
    ap.add_argument("--logs", type=Path, default=None, metavar="DIR",
                    help="Run without questions (--skip-review) and write each run's output to DIR/<name>.log.")
    ap.add_argument("--stop-on-error", action="store_true", help="Stop at the first run that fails.")
    ap.add_argument("--dry-run", action="store_true", help="Only list the commands that would run.")
    ap.add_argument("--workflow", type=Path, default=default_workflow(),
                    help="benchmantic's workflow.py (default: next to this script, else in $BENCHMANTIC_HOME, "
                         f"else {BENCHMANTIC_HOME}).")
    args, extra = ap.parse_known_args(argv)      # anything else is passed on to workflow.py

    folder = args.folder.expanduser().resolve()
    args.workflow = args.workflow.expanduser().resolve()
    if args.logs is not None:
        args.logs = args.logs.expanduser().resolve()
    if not folder.is_dir():
        sys.exit(f"Error: {folder} is not a folder.")
    if not args.workflow.is_file():
        sys.exit(f"Error: benchmantic's workflow.py not found at {args.workflow} -- pass --workflow "
                 "/path/to/benchmantic/workflow.py, or set BENCHMANTIC_HOME to the benchmantic folder.")
    configs, skipped = find_configs(folder, args.only)
    if not configs:
        sys.exit(f"Error: no run configs (YAML files with an `implementation:` key) in {folder}"
                 + (f" matching {', '.join(args.only)}" if args.only else "") + ".")
    if skipped:
        print("Not run configs, skipped: " + ", ".join(p.name for p in skipped))
    print(f"{len(configs)} run config(s) in {folder}:")
    for config in configs:
        print(f"  python3 {args.workflow.name} --config {config}" + (" " + " ".join(extra) if extra else ""))
    if args.dry_run:
        return 0

    results: list[tuple[Path, int, float]] = []
    for number, config in enumerate(configs, start=1):
        print(f"\n{'=' * 64}\n[{number}/{len(configs)}] {config.name}\n{'=' * 64}", flush=True)
        try:
            code, seconds = run_one(config, args.workflow, args.logs, extra)
        except KeyboardInterrupt:
            print(f"\nStopped during {config.name}.")
            results.append((config, 130, 0.0))
            break
        results.append((config, code, seconds))
        if args.logs is not None:
            print(f"{'done' if code == 0 else f'FAILED (exit {code})'} in {_duration(seconds)} -- "
                  f"log: {args.logs / (config.stem + '.log')}")
        if code != 0 and args.stop_on_error:
            print("Stopping (--stop-on-error).")
            break

    print(f"\n{'=' * 64}\nSummary\n{'=' * 64}")
    for config, code, seconds in results:
        print(f"  {'ok    ' if code == 0 else 'FAILED'}  {config.name:<40} {_duration(seconds)}")
    not_run = configs[len(results):]
    for config in not_run:
        print(f"  not run {config.name}")
    failed = sum(1 for _, code, _ in results if code != 0)
    print(f"{len(results) - failed} ok, {failed} failed" + (f", {len(not_run)} not run" if not_run else ""))
    return 1 if failed or not_run else 0


if __name__ == "__main__":
    sys.exit(main())
