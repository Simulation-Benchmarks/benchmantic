#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
show_description.py

Renders a metadata.builder-produced benchmark.jsonld RO-Crate graph as
human-readable Markdown tables for manual verification: benchmark/manifest
info, pinned dependency versions, input parameters, and output metrics.

Author/publisher/dependency info lives in a separate sidecar file --
"<benchmark-name>_dataset.jsonld", written alongside the benchmark file by
describe_benchmark.py (see metadata.builder.GraphBuilder.build_dataset_graph())
-- rather than in the benchmark file itself. This script auto-discovers
that sibling file next to the one you pass in (or accepts --dataset-jsonld
to point at it explicitly) and merges it back in for display, so the
rendered tables look the same as when everything lived in one file. If no
sidecar is found, those rows/sections are just omitted -- not an error.

Usage
-----
    python3 show_description.py rotating_cylinders_benchmark.jsonld
    python3 show_description.py rotating_cylinders_benchmark.jsonld --output review.md
    python3 show_description.py rotating_cylinders_benchmark.jsonld --no-save
    python3 show_description.py rotating_cylinders_benchmark.jsonld \\
        --dataset-jsonld rotating_cylinders_dataset.jsonld

Always prints to stdout. Also always saves a copy as review.md right next
to the input file by default -- e.g. outputs/dumux/rotating_cylinders_benchmark.jsonld
-> outputs/dumux/review.md -- so the rendered table lives alongside the
benchmark file and the Snakefile, not just in your terminal history.
Override the saved filename/location with --output, or skip saving
entirely with --no-save.
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

def build_manifest_section(
    by_id: dict[str, dict[str, Any]],
    out: list[str],
    dataset_by_id: dict[str, dict[str, Any]] | None = None,
) -> None:
    root = by_id.get("./")
    benchmark = find_first(by_id, "m4i:Benchmark")
    software = by_id.get("local:software")
    license_node = by_id.get(_id(root.get("license"))) if root else None
    publication = by_id.get(_id(benchmark.get("describedAsDocumentedBy"))) if benchmark else None

    # Author/publisher live in the sibling "<benchmark-name>_dataset.jsonld"
    # file's own root entity now, not this one -- see this module's
    # docstring and GraphBuilder.build_dataset_graph().
    dataset_root = (dataset_by_id or {}).get("./")
    author = dataset_by_id.get(_id(dataset_root.get("author"))) if dataset_root and dataset_by_id else None
    publisher = dataset_by_id.get(_id(dataset_root.get("publisher"))) if dataset_root and dataset_by_id else None

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


def build_dependencies_section(
    by_id: dict[str, dict[str, Any]],
    out: list[str],
    dataset_by_id: dict[str, dict[str, Any]] | None = None,
) -> None:
    # Dependencies (schema:softwareRequirements) live on the sibling
    # dataset file's own root entity now, not on this file's "local:software"
    # node -- see build_manifest_section()'s comment above.
    dataset_root = (dataset_by_id or {}).get("./")
    deps: list[str] = (dataset_root or {}).get("schema:softwareRequirements", [])
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

def _find_sibling_dataset(metadata_jsonld: Path) -> Path | None:
    """Best-effort discovery of the sibling "<benchmark-name>_dataset.jsonld"
    file describe_benchmark.py writes next to the benchmark file (see
    metadata.builder.GraphBuilder.build_dataset_graph()) -- holds author/
    publisher/dependency info that's no longer part of the benchmark file
    itself. Tried in order:
      1. "<stem without a trailing '_benchmark'>_dataset.jsonld" next to it
         (the normal case -- matches describe_benchmark.py's naming).
      2. The single "*_dataset.jsonld" file in the same directory, if
         exactly one exists (covers a renamed/--benchmark-filename input).
    Returns None (not an error) if nothing matches -- the affected
    rows/sections are just omitted, same as before this sidecar file
    existed. Use --dataset-jsonld to point at one explicitly instead of
    relying on this discovery.
    """
    stem = metadata_jsonld.stem
    if stem.endswith("_benchmark"):
        candidate = metadata_jsonld.with_name(stem[: -len("_benchmark")] + "_dataset.jsonld")
        if candidate.exists():
            return candidate
    matches = sorted(metadata_jsonld.parent.glob("*_dataset.jsonld"))
    return matches[0] if len(matches) == 1 else None


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metadata_jsonld", type=Path,
                     help="Path to a describe_benchmark.py output (the '<benchmark-name>_benchmark.jsonld' file).")
    ap.add_argument("--dataset-jsonld", type=Path, default=None,
                     help="Path to the sibling '<benchmark-name>_dataset.jsonld' file (author/publisher/"
                          "dependencies) written alongside metadata_jsonld. Default: auto-discovered next to "
                          "metadata_jsonld; if none is found, those rows/sections are just omitted.")
    ap.add_argument("--output", type=Path, default=None,
                     help="Save the Markdown tables to this file instead of the default "
                          "<input directory>/review.md (e.g. outputs/dumux/rotating_cylinders_benchmark.jsonld -> "
                          "outputs/dumux/review.md).")
    ap.add_argument("--no-save", action="store_true",
                     help="Only print to stdout -- don't save a review.md (or --output) file at all.")
    return ap


def run(args: argparse.Namespace) -> None:
    if not args.metadata_jsonld.exists():
        sys.exit(f"Error: {args.metadata_jsonld} does not exist")

    by_id = load_graph(args.metadata_jsonld)

    dataset_path = args.dataset_jsonld or _find_sibling_dataset(args.metadata_jsonld)
    if args.dataset_jsonld and not args.dataset_jsonld.exists():
        sys.exit(f"Error: {args.dataset_jsonld} does not exist")
    dataset_by_id = load_graph(dataset_path) if dataset_path else None

    out: list[str] = []
    build_manifest_section(by_id, out, dataset_by_id)
    build_dependencies_section(by_id, out, dataset_by_id)
    build_parameters_section(by_id, out)
    build_metrics_section(by_id, out)

    text = "\n".join(out)
    print(text)

    if not args.no_save:
        # Default: save right next to the input (e.g. outputs/dumux/benchmark.jsonld
        # -> outputs/dumux/review.md), so the table lives alongside benchmark.jsonld
        # and the Snakefile zip -- not just in stdout/terminal scrollback.
        save_path = args.output or (args.metadata_jsonld.parent / "review.md")
        save_path.write_text(text, encoding="utf-8")
        print(f"\nSaved {save_path}")


def main(argv: list[str] | None = None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
