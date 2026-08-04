"""
metadata.graph

Small shared helpers for reading a metadata.builder-produced benchmark.jsonld
graph: load_graph(), find_all(), find_first(), node_id(). Used across the
metadata/ai/snakefile packages, and by the standalone metadata_to_table.py /
validate_benchmark_contract.py utilities, so the "@id"-resolution /
node-lookup logic exists in exactly one place.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def node_id(ref: Any) -> str | None:
    """Pull the @id out of a {"@id": ...} reference, or return the value
    itself if it's already a plain string/None."""
    if isinstance(ref, dict):
        return ref.get("@id")
    return ref


def load_graph(path: Path) -> dict[str, dict[str, Any]]:
    """Load a metadata.jsonld's "@graph" array into an @id -> node lookup."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    graph = doc.get("@graph")
    if not isinstance(graph, list):
        sys.exit(f"Error: {path} has no top-level '@graph' array -- is this a generate_metadata.py output?")
    by_id: dict[str, dict[str, Any]] = {}
    for node in graph:
        nid = node.get("@id")
        if nid:
            by_id[nid] = node
    return by_id


def find_first(by_id: dict[str, dict[str, Any]], type_name: str) -> dict[str, Any] | None:
    """First node whose @type is (or includes) `type_name`."""
    for node in by_id.values():
        t = node.get("@type")
        if t == type_name or (isinstance(t, list) and type_name in t):
            return node
    return None


def find_all(by_id: dict[str, dict[str, Any]], type_name: str) -> list[dict[str, Any]]:
    """All nodes whose @type is (or includes) `type_name`."""
    return [
        node for node in by_id.values()
        if node.get("@type") == type_name or (isinstance(node.get("@type"), list) and type_name in node["@type"])
    ]
