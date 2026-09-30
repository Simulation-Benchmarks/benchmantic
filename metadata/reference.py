# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
metadata.reference

Reads an existing benchmark description (e.g. the catalog's
benchmark/<version>/minimal-configurations.json) as the *reference* for
adding another implementation: its software-neutral parameters (name,
unit, quantity kind), its configurations (parameter values per case) and
its metrics.

Uses rdflib directly rather than semantic_benchmark.BenchmarkLoader so it
also reads parameter nodes typed m4i:NumericalVariable that carry a value
(as the rotating-cylinders reference does).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rdflib import RDF, RDFS, Graph, Literal, Namespace, URIRef

M4I = Namespace("http://w3id.org/nfdi4ing/metadata4ing#")
OBO = Namespace("http://purl.obolibrary.org/obo/")
HAS_PART = OBO.BFO_0000051
DCTERMS_DESCRIPTION = URIRef("http://purl.org/dc/terms/description")
SCHEMA_DESCRIPTION = URIRef("https://schema.org/description")


@dataclass
class ReferenceBenchmark:
    path: Path
    label: str | None
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)   # name -> unit/quantityKind/datatype/description
    configurations: list[dict[str, Any]] = field(default_factory=list)   # {"id", "label", "values": {name: value}}
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)      # label -> unit/quantityKind/description

    def summary(self) -> str:
        return (f"{len(self.parameters)} parameters, {len(self.configurations)} configuration(s), "
                f"{len(self.metrics)} metric(s)")


def _short(graph: Graph, node) -> str | None:
    if node is None:
        return None
    if isinstance(node, URIRef):
        for prefix, curie_prefix in (("http://qudt.org/vocab/unit/", "unit:"), ("https://qudt.org/vocab/unit/", "unit:")):
            if str(node).startswith(prefix):
                return curie_prefix + str(node)[len(prefix):]
        return str(node)
    return str(node)


def _description(graph: Graph, node) -> str:
    for pred in (DCTERMS_DESCRIPTION, SCHEMA_DESCRIPTION, RDFS.comment):
        value = graph.value(node, pred)
        if value is not None:
            return str(value)
    return ""


def load_reference(path: Path) -> ReferenceBenchmark:
    graph = Graph()
    try:
        graph.parse(str(path), format="json-ld")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Error: could not read the reference benchmark {path}: {exc}")

    benchmark = next(graph.subjects(RDF.type, M4I.Benchmark), None)
    if benchmark is None:
        raise SystemExit(f"Error: {path} has no m4i:Benchmark node -- is it a benchmark description?")
    ref = ReferenceBenchmark(path=path, label=str(graph.value(benchmark, RDFS.label) or "") or None)

    for pset in sorted(graph.objects(benchmark, M4I.hasParameterSet), key=str):
        values: dict[str, Any] = {}
        for part in graph.objects(pset, HAS_PART):
            name = graph.value(part, RDFS.label)
            if name is None:
                continue
            name = str(name)
            num = graph.value(part, M4I.hasNumericalValue)
            txt = graph.value(part, M4I.hasStringValue)
            if isinstance(num, Literal):
                value = num.toPython()
                datatype = "schema:Integer" if isinstance(value, int) and not isinstance(value, bool) else "schema:Float"
            elif txt is not None:
                value, datatype = str(txt), "schema:String"
            else:
                continue
            values[name] = value
            info = ref.parameters.setdefault(name, {
                "unit": _short(graph, graph.value(part, M4I.hasUnit)) or "unit:UNITLESS",
                "quantityKind": _short(graph, graph.value(part, M4I.hasKindOfQuantity)),
                "datatype": datatype,
                "description": _description(graph, part),
            })
            if info["datatype"] == "schema:Integer" and datatype == "schema:Float":
                info["datatype"] = "schema:Float"
        ident = graph.value(pset, M4I.identifier)
        ref.configurations.append({
            "id": str(ident) if ident is not None else str(pset).rsplit("/", 1)[-1],
            "label": str(graph.value(pset, RDFS.label) or ident or ""),
            "values": values,
        })

    for metric in graph.objects(benchmark, M4I.evaluates):
        name = graph.value(metric, RDFS.label)
        if name is not None:
            ref.metrics[str(name)] = {
                "unit": _short(graph, graph.value(metric, M4I.hasUnit)) or "unit:UNITLESS",
                "quantityKind": _short(graph, graph.value(metric, M4I.hasKindOfQuantity)),
                "description": _description(graph, metric),
            }

    if not ref.parameters:
        raise SystemExit(f"Error: the reference benchmark {path} has no parameter values to map onto.")
    return ref


# ---------------------------------------------------------------------------
# Finding a reference benchmark automatically
# ---------------------------------------------------------------------------

#: Folders never searched for benchmark descriptions.
SKIP_DIRS = {".git", ".benchmantic", "node_modules", "outputs", "results", "build", "build-cmake",
             "__pycache__", ".venv", "venv"}
MAX_FILE_BYTES = 5_000_000


def _version_key(path: Path) -> tuple:
    """Sort key preferring higher version folders (benchmark/1.10.0 > 1.9.0)."""
    parts = []
    for part in path.parts:
        nums = [int(n) for n in part.split(".") if n.isdigit()] if part.replace(".", "").isdigit() else []
        parts.append(tuple(nums))
    return tuple(parts)


def _looks_like_benchmark(path: Path) -> bool:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return False
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "Benchmark" in text and ("hasParameterSet" in text or "has parameter set" in text)


def _try_load(path: Path) -> ReferenceBenchmark | None:
    try:
        return load_reference(path)
    except SystemExit:
        return None


def _same_name(a: str | None, b: str | None) -> bool:
    norm = lambda s: " ".join((s or "").lower().replace("_", " ").replace("-", " ").split())  # noqa: E731
    return bool(a) and bool(b) and norm(a) == norm(b)


def repository_candidates(root: Path) -> list[Path]:
    found = []
    for path in root.rglob("*"):
        if path.suffix not in (".json", ".jsonld") or not path.is_file():
            continue
        rel = path.relative_to(root).parts
        if any(p in SKIP_DIRS or (p.startswith(".") and p != ".") for p in rel[:-1]):
            continue
        if _looks_like_benchmark(path):
            found.append(path)
    return found


def discover_reference(docs_root: Path, benchmark_label: str | None, software_slug: str,
                       outputs_root: Path | None = None) -> tuple[ReferenceBenchmark | None, str, list[Path]]:
    """Look for a reference benchmark description:

    1. in the repository (a JSON-LD file with an m4i:Benchmark and parameter
       sets, e.g. benchmark/1.0.0/minimal-configurations.json); with several,
       one whose name matches the benchmark, else the highest version folder;
    2. among benchmantic's earlier outputs for another software with the same
       benchmark name (outputs/<software>/*_benchmark.jsonld).

    Returns (reference or None, where it was found, other candidates seen).
    """
    repo = [(p, r) for p in repository_candidates(docs_root) if (r := _try_load(p))]
    if repo:
        named = [(p, r) for p, r in repo if _same_name(r.label, benchmark_label)]
        pool = named or repo
        pool.sort(key=lambda pr: (("benchmark" in pr[0].parts), _version_key(pr[0])), reverse=True)
        chosen = pool[0]
        others = [p for p, _ in repo if p != chosen[0]]
        return chosen[1], "in the repository", others

    outputs_root = outputs_root or Path("outputs")
    if outputs_root.is_dir():
        for path in sorted(outputs_root.glob("*/*_benchmark.jsonld")):
            if path.parent.name.lower() == software_slug.lower():
                continue
            ref = _try_load(path)
            if ref and _same_name(ref.label, benchmark_label):
                return ref, f"from the earlier {path.parent.name} run", []
    return None, "", []
