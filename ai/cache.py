# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
ai.cache

On-disk caching of LLM-inferred parameter/metric metadata, keyed to a
module directory, so repeated runs don't re-query the API for items
already inferred.
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


