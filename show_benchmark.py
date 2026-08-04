#!/usr/bin/env python3
"""
show_benchmark.py

Renders a metadata.builder-produced benchmark.jsonld RO-Crate graph as
human-readable Markdown tables for manual verification: benchmark/manifest
info, pinned dependency versions, input parameters, and output metrics.

Usage
-----
    python3 show_benchmark.py benchmark.jsonld
    python3 show_benchmark.py benchmark.jsonld --output review.md

With no --output, the tables are printed to stdout.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from metadata.graph import node_id as _id, load_graph, find_first, find_all


# ============================================================
# Small helpers
# ============================================================

def _short(uri: str | None) -> str:
    """Shorten a long QUDT/URI value to its trailing path segment for
    compact table display, e.g. 'unit:M' stays as-is, but
    'http://qudt.org/vocab/quantitykind/Length' -> 'Length'."""
    if not uri:
        return ""
    if uri.startswith("unit:"):
        return uri
    return uri.rstrip("/").rsplit("/", 1)[-1]


def _md_escape(text: Any) -> str:
    """Escape pipe characters and collapse newlines so a value is safe to
    drop into a Markdown table cell."""
    s = "" if text is None else str(text)
    s = s.replace("|", "\\|").replace("\n", " ").strip()
    return s


#: A "unit" value that actually looks like a quantityKind URI is a known
#: LLM mix-up (see ai_parameter_inference.py's _fix_unit_quantitykind_confusion) --
#: flagged here too in case this jsonld predates that fix.
QUANTITYKIND_IN_UNIT_PATTERN = re.compile(r"quantitykind[/:]", re.IGNORECASE)


def _print_table(rows: list[list[str]], headers: list[str], out: list[str]) -> None:
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(_md_escape(c) for c in row) + " |")
    out.append("")


# ============================================================
# Section builders
# ============================================================

def build_manifest_section(by_id: dict[str, dict[str, Any]], out: list[str]) -> None:
    root = by_id.get("./")
    benchmark = find_first(by_id, "m4i:Benchmark")
    software = by_id.get("local:software")
    license_node = by_id.get(_id(root.get("license"))) if root else None
    author = by_id.get(_id(root.get("author"))) if root else None
    publisher = by_id.get(_id(root.get("publisher"))) if root else None
    publication = by_id.get(_id(benchmark.get("describedAsDocumentedBy"))) if benchmark else None

    out.append("## Benchmark / Manifest metadata\n")
    rows = [
        ["Name / label", (root or {}).get("name", "")],
        ["Description", (root or {}).get("description", "")],
        ["Publication citation", (publication or {}).get("label", "")],
        ["Version", (benchmark or {}).get("schema:version", "")],
        ["Date published", (root or {}).get("datePublished", "")],
        ["License", (license_node or {}).get("name", "")],
        ["Software", (software or {}).get("label", "")],
        ["Code repository", (root or {}).get("codeRepository", "")],
    ]
    if author:
        note = f" ({author['schema:disambiguatingDescription']})" if "schema:disambiguatingDescription" in author else ""
        rows.append(["Author", f"{author.get('name', '')} [{_short(author.get('@type'))}]{note}"])
        if author.get("schema:url"):
            rows.append(["Author URL", author["schema:url"]])
    if publisher:
        note = f" ({publisher['schema:disambiguatingDescription']})" if "schema:disambiguatingDescription" in publisher else ""
        rows.append(["Publisher", f"{publisher.get('name', '')} [{_short(publisher.get('@type'))}]{note}"])
        if publisher.get("schema:url"):
            rows.append(["Publisher URL", publisher["schema:url"]])

    _print_table(rows, ["Field", "Value"], out)


def build_dependencies_section(by_id: dict[str, dict[str, Any]], out: list[str]) -> None:
    software = by_id.get("local:software")
    deps: list[str] = (software or {}).get("schema:softwareRequirements", [])
    if not deps:
        return

    out.append("## Software dependencies\n")
    rows = []
    #: Each entry looks like 'dune-istl@21c67275b17e (origin/releases/2.10, 2025-02-03 09:13:05 +0000)'
    dep_pattern = re.compile(r"^(?P<module>[\w.\-]+)@(?P<commit>[0-9a-f?]*)\s*(?:\((?P<extra>.*)\))?$")
    for dep in deps:
        m = dep_pattern.match(dep)
        if m:
            module = m.group("module")
            commit = m.group("commit") or ""
            extra = m.group("extra") or ""
            branch, _, date = extra.partition(", ")
            rows.append([module, branch, commit, date])
        else:
            rows.append([dep, "", "", ""])
    _print_table(rows, ["Module", "Branch", "Commit (short)", "Commit date"], out)


def _variable_row(var: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> tuple[list[str], bool]:
    """Build one table row for a 'numerical variable' node. Returns
    (row, has_unit_quantitykind_mixup)."""
    label = var.get("label", "")
    value = var.get("has numerical value")
    if value is None:
        value = var.get("has string value", "")
    unit = _id(var.get("has unit")) or ""
    qkind = _id(var.get("has quantity kind")) or ""
    desc = var.get("dcterms:description", "")

    # Datatype lives on the sibling Field node, found via the matching
    # 'local:field_<suffix>' id derived the same way GraphBuilder built it.
    field_id = var["@id"].replace("local:variable_", "local:field_", 1).replace("local:metric_", "local:field_", 1)
    field = by_id.get(field_id)
    datatype = _short(_id(field.get("dataType"))) if field else ""

    is_mixup = bool(QUANTITYKIND_IN_UNIT_PATTERN.search(unit))
    unit_display = f"⚠️ {_short(unit)}" if is_mixup else unit

    return [label, str(value), datatype, unit_display, _short(qkind), desc], is_mixup


def build_parameters_section(by_id: dict[str, dict[str, Any]], out: list[str]) -> None:
    param_sets = find_all(by_id, "m4i:ParameterSet")
    if not param_sets:
        return

    for pset in param_sets:
        out.append(f"## Parameters -- configuration `{pset.get('identifier', '')}` ({pset.get('label', '')})\n")
        rows = []
        any_mixup = False
        for part_ref in pset.get("has part", []):
            var = by_id.get(_id(part_ref))
            if not var:
                continue
            row, mixup = _variable_row(var, by_id)
            any_mixup = any_mixup or mixup
            rows.append(row)
        _print_table(rows, ["Parameter", "Value", "Datatype", "Unit", "Quantity Kind", "Description"], out)
        if any_mixup:
            out.append(
                "⚠️ = the 'unit' field holds a quantityKind URI instead of an actual QUDT unit "
                "(pre-existing data issue -- see ai_parameter_inference.py's unit/quantityKind fix). "
                "Re-run generate_metadata.py with --clear-cache to correct it.\n"
            )


def build_metrics_section(by_id: dict[str, dict[str, Any]], out: list[str]) -> None:
    benchmark = find_first(by_id, "m4i:Benchmark")
    if not benchmark:
        return
    metric_refs = benchmark.get("evaluates", [])
    if not metric_refs:
        return

    out.append("## Metrics (output/solution quantities)\n")
    rows = []
    any_mixup = False
    for ref in metric_refs:
        var = by_id.get(_id(ref))
        if not var:
            continue
        row, mixup = _variable_row(var, by_id)
        # metrics have no "has numerical value" -- drop that column's content
        row[1] = ""
        any_mixup = any_mixup or mixup
        rows.append(row)
    _print_table(rows, ["Metric key", "Value", "Datatype", "Unit", "Quantity Kind", "Description"], out)
    if any_mixup:
        out.append("⚠️ = unit/quantityKind mix-up detected, see note above.\n")


# ============================================================
# Entry point
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metadata_jsonld", type=Path, help="Path to a build_benchmark.py output (benchmark.jsonld).")
    ap.add_argument("--output", type=Path, default=None,
                     help="Write the Markdown tables to this file instead of printing to stdout.")
    return ap


def run(args: argparse.Namespace) -> None:
    if not args.metadata_jsonld.exists():
        sys.exit(f"Error: {args.metadata_jsonld} does not exist")

    by_id = load_graph(args.metadata_jsonld)

    out: list[str] = []
    build_manifest_section(by_id, out)
    build_dependencies_section(by_id, out)
    build_parameters_section(by_id, out)
    build_metrics_section(by_id, out)

    text = "\n".join(out)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)


def main(argv: list[str] | None = None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
