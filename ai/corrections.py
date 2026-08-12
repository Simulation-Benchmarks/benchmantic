# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
ai.corrections

Cross-benchmark memory of human corrections made during the interactive
review step (see review.py): every time a reviewer edits an AI-inferred
parameter/metric field, the (AI-suggested -> human-corrected) pair is
appended here, keyed loosely enough -- by raw key name, matched exactly or
fuzzily -- that a SIMILAR item in a *different* benchmark module still
benefits from it, not just an exact repeat of the same key in the same
module.

This is NOT the per-module source of truth for any one run's metadata --
that's still .parameter_metadata_cache.json / .metric_metadata_cache.json
(see ai.cache), which is what actually gets reused verbatim on the next run
of the same module. This store instead feeds back into future LLM prompts
(see ai.prompts's "known corrections" section) as background knowledge, so
the model is less likely to repeat a mistake a human already fixed once,
even the first time it sees a given module.
"""

from __future__ import annotations

import difflib
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent

#: Default location -- one file shared across every benchmark module run
#: with this checkout, so a correction made on one benchmark helps the
#: next. Override with BENCHMANTIC_CORRECTIONS_PATH to point at a different
#: (e.g. shared/team-wide) location.
DEFAULT_CORRECTIONS_PATH = SCRIPT_DIR / ".benchmantic_corrections.json"


def corrections_path() -> Path:
    env = os.environ.get("BENCHMANTIC_CORRECTIONS_PATH")
    return Path(env) if env else DEFAULT_CORRECTIONS_PATH


def load_corrections(path: Path | None = None) -> list[dict]:
    """Load every stored correction record. Returns [] if the store doesn't
    exist yet, or (with a warning, not a crash) if it's unreadable -- this
    is best-effort background knowledge, never something a run should fail
    over.
    """
    p = path or corrections_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"warning: could not read corrections store {p}: {exc} -- ignoring.", file=sys.stderr)
        return []
    return data if isinstance(data, list) else []


def append_corrections(new_entries: list[dict], path: Path | None = None) -> None:
    """Append `new_entries` (as produced by review.py's interactive_review())
    to the on-disk store. No-op if there's nothing to add.
    """
    if not new_entries:
        return
    p = path or corrections_path()
    existing = load_corrections(p)
    existing.extend(new_entries)
    p.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_similar(
    key: str,
    kind: str,
    corrections: list[dict] | None = None,
    limit: int = 3,
    threshold: float = 0.5,
) -> list[dict]:
    """Return up to `limit` prior corrections of the same `kind`
    ("parameter"/"metric") whose raw key is exactly or approximately the
    same as `key` (e.g. 'Omega1' in one module vs 'Omega2' or 'omega_in' in
    another) -- most-similar first. Exact (case-insensitive) matches always
    sort ahead of fuzzy ones.
    """
    pool = corrections if corrections is not None else load_corrections()
    scored: list[tuple[float, dict]] = []
    for entry in pool:
        if entry.get("type") != kind:
            continue
        other_key = entry.get("key") or ""
        if not other_key:
            continue
        exact = other_key.lower() == key.lower()
        score = 1.0 if exact else _similarity(key, other_key)
        if exact or score >= threshold:
            scored.append((score, entry))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [entry for _, entry in scored[:limit]]


def relevant_corrections_for(
    keys: list[str],
    kind: str,
    limit_per_key: int = 2,
    total_limit: int = 6,
) -> list[dict]:
    """Convenience wrapper for a batch of candidate keys (e.g. every
    parameter/metric about to be sent to the LLM in one request): collects
    find_similar() results per key, dedupes, and caps the total so the
    prompt doesn't grow unbounded as the corrections store accumulates.
    """
    corrections = load_corrections()
    if not corrections:
        return []
    seen_ids: set[tuple] = set()
    combined: list[dict] = []
    for key in keys:
        for entry in find_similar(key, kind, corrections, limit=limit_per_key):
            entry_id = (
                entry.get("key"), entry.get("section"),
                json.dumps(entry.get("user_corrected", {}), sort_keys=True),
            )
            if entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)
            combined.append(entry)
            if len(combined) >= total_limit:
                return combined
    return combined
