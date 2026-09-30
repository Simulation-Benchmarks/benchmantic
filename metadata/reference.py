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
