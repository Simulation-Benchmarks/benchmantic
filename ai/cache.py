# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
ai.cache

On-disk caching of LLM-inferred parameter/metric metadata, keyed to a
module directory, so repeated runs don't re-query the API for items
already inferred. Also persists the "which artifacts to generate"
Outputs-step selection (outputs_config_path()/load_outputs_config()/
save_outputs_config()) the same way, so that choice is remembered too.
"""

from __future__ import annotations

import json
from pathlib import Path

def cache_path(module_dir: Path) -> Path:
    return module_dir / ".parameter_metadata_cache.json"


def save_cache(module_dir: Path, metadata: list[dict]) -> None:
    cache_path(module_dir).write_text(json.dumps(metadata, indent=2))


def load_cache(module_dir: Path) -> list[dict] | None:
    path = cache_path(module_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def metric_cache_path(module_dir: Path) -> Path:
    return module_dir / ".metric_metadata_cache.json"


def save_metric_cache(module_dir: Path, metadata: list[dict]) -> None:
    metric_cache_path(module_dir).write_text(json.dumps(metadata, indent=2))


def load_metric_cache(module_dir: Path) -> list[dict] | None:
    path = metric_cache_path(module_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def outputs_config_path(module_dir: Path) -> Path:
    return module_dir / ".benchmantic_outputs.json"


def save_outputs_config(module_dir: Path, outputs: dict) -> None:
    """Persist the "which artifacts to generate" selection (dataset/
    snakefile/review booleans -- see metadata.builder's Outputs step)
    beside the module, so a later run against the same module_dir doesn't
    have to ask again. Called every time a selection is resolved --
    whether it came from an interactive prompt, a previously saved
    config, or an explicit --outputs override -- so the most recently
    used selection is always what gets remembered.
    """
    outputs_config_path(module_dir).write_text(json.dumps(outputs, indent=2))


def load_outputs_config(module_dir: Path) -> dict | None:
    path = outputs_config_path(module_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


