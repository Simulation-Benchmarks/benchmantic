# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
metadata.croissant

The benchmark's input parameters as a Croissant description, one entry per
parameter, in the style semantic-benchmark reads (see its
docs/croissant-jsonpath.md):

    cr:Field ---represents---> numerical / text variable (label, unit, values)
        \\---source---> cr:DataSource --extract--> {jsonPath: "/radius[m]"}
                                   \\--file object--> cr:FileObject "parameter.json"

The jsonPath is the parameter's key in the parameter file a benchmark run
writes for each configuration: its name, with the unit symbol when it has a
known one ("/radius[m]"), as run_benchmark.py and the generated Snakefile use it. A
parameter used with several values (across the configurations) lists them
all, as a sweep in semantic-benchmark does. Where the parameter goes in the
implementation -- the software's own input, the file it is in, line and
column, and the expression that sets it -- comes from the input mapping
(<name>_mapping.json) and is added to the field as bm:implementation. Each
of those files is a cr:FileObject too, as the Croissant spec describes files:
name, contentUrl (a link to it when the implementation came from a
repository link, else its path in the repository), encodingFormat, sha256;
a file in a zipped case has containedIn the zip's FileObject.

Built from the files benchmantic writes anyway (the benchmark description
and the mapping), so it always agrees with them.
"""

from __future__ import annotations

import json
import posixpath
import re
from pathlib import Path
from typing import Any

from snakefile.renderer import DEFAULT_UNIT_SYMBOLS, config_key
from utils import slugify

#: The benchmarks namespace (as in the Simulation-Benchmarks descriptions),
#: for what Croissant itself has no term for.
BM = "https://w3id.org/nfdi4ing/benchmarks/"
PARAMETER_TYPES = {"numerical parameter": "numerical variable", "text parameter": "text variable"}


def _ref(node: Any) -> str | None:
    return node.get("@id") if isinstance(node, dict) else None


def _values(items: list[Any]) -> Any:
    """One value, or the different values in the order they first appear."""
    unique: list[Any] = []
    for item in items:
        if item not in unique:
            unique.append(item)
    return unique[0] if len(unique) == 1 else unique


ENCODINGS = {".json": "application/json", ".ipynb": "application/x-ipynb+json", ".py": "text/x-python",
             ".yaml": "application/yaml", ".yml": "application/yaml", ".xml": "application/xml"}


def _repository_link(repository: str, ref: str | None, path: str) -> str:
    """A link to a file on the repository's web page (GitHub, GitLab)."""
    base = repository[:-4] if repository.endswith(".git") else repository.rstrip("/")
    if base.startswith("git@"):                      # git@github.com:org/repo -> https://github.com/org/repo
        base = "https://" + base[4:].replace(":", "/", 1)
    blob = "/blob/" if "github.com" in base else "/-/blob/"
    return f"{base}{blob}{ref or 'HEAD'}/{path}"


class _Files:
    """The implementation's files the parameters are found in, as
    cr:FileObjects -- one per file, made when first named: name, contentUrl
    (a link to the file in the repository it came from, else its path
    there), encodingFormat, sha256. A file in a zipped case is described
    with containedIn the zip's own FileObject, its contentUrl the path in
    the zip, as the Croissant spec does it."""

    def __init__(self, place: dict[str, Any] | None) -> None:
        self.place = place or {}
        self.nodes: dict[str, dict[str, Any]] = {}

    def _url(self, in_repository: str) -> str:
        place = self.place
        if place.get("repository"):
            return _repository_link(place["repository"], place.get("ref"), in_repository)
        return in_repository

    def _node(self, node_id: str, name: str, url: str, checksum: str | None, **more: Any) -> dict[str, str]:
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "@id": node_id, "@type": "cr:FileObject", "name": name, "contentUrl": url,
                "encodingFormat": ENCODINGS.get(posixpath.splitext(name)[1].lower(), "text/plain"),
                **({"sha256": checksum} if checksum else {}), **more,
            }
        return {"@id": node_id}

    def ref(self, file: str) -> dict[str, str]:
        place = self.place
        sums = place.get("sha256") or {}
        folder = place.get("folder", ".")
        archive = place.get("archive")
        if archive:                                  # the case was in a zip: the file is inside it
            zip_path = posixpath.normpath(posixpath.join(folder, archive))
            zip_ref = self._node(_file_id(zip_path), posixpath.basename(archive), self._url(zip_path),
                                 sums.get(archive), encodingFormat="application/zip",
                                 **{"bm:pathInRepository": zip_path})
            inner = posixpath.normpath(posixpath.join(place.get("path_in_archive", "."), file))
            return self._node(_file_id(zip_path + "#" + inner), posixpath.basename(inner), inner, None,
                              containedIn=zip_ref,
                              **{"dcterms:description": "A file of the implementation that sets input parameters."})
        in_repository = posixpath.normpath(posixpath.join(folder, place.get("case_folder", "."), file))
        return self._node(_file_id(in_repository), posixpath.basename(in_repository), self._url(in_repository),
                          sums.get(file), **{"bm:pathInRepository": in_repository,
                                             "dcterms:description": "A file of the implementation that sets "
                                                                    "input parameters."})


def _file_id(path: str) -> str:
    return "local:file_" + re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_")


def _implementation(mapping: dict[str, Any] | None, name: str, files: _Files) -> list[dict[str, Any]]:
    """The software inputs a benchmark parameter sets (from the mapping),
    each with the file it is in (a cr:FileObject), line and column."""
    found = []
    for item in (mapping or {}).get("inputs", []):
        if name not in item.get("benchmark_parameters", []):
            continue
        entry: dict[str, Any] = {"bm:input": item["input"], "bm:expression": item.get("expression")}
        location = item.get("location") or {}
        if location.get("file"):
            entry.update({"bm:file": files.ref(location["file"]), "bm:line": location.get("line"),
                          "bm:column": location.get("column")})
        found.append({k: v for k, v in entry.items() if v is not None})
    return found


def build_parameters_croissant(benchmark: dict[str, Any], mapping: dict[str, Any] | None = None,
                               unit_symbols: dict[str, str] | None = None) -> dict[str, Any]:
    """The Croissant description of the input parameters in `benchmark` (a
    benchmark description as benchmantic writes it), with `mapping` (its
    <name>_mapping.json) for where each parameter goes in the implementation.
    The jsonPath is the key in the configuration's parameter file: the name,
    with its unit symbol when it has a known one ("/radius[m]"), as the
    benchmark run writes it (see snakefile.renderer.config_key)."""
    unit_symbols = unit_symbols or DEFAULT_UNIT_SYMBOLS
    graph = benchmark.get("@graph", [])
    by_id = {n.get("@id"): n for n in graph if isinstance(n, dict)}
    file_object = by_id.get("local:parameter_file_object", {"label": "parameter.json"})

    # the parameter values, by parameter, in configuration order
    order: list[str] = []
    nodes: dict[str, dict[str, Any]] = {}
    values: dict[str, list[Any]] = {}
    configurations = [n for n in graph if n.get("@type") == "m4i:ParameterSet"]
    for configuration in configurations:
        for part in configuration.get("has part", []):
            node = by_id.get(_ref(part)) or {}
            if node.get("@type") not in PARAMETER_TYPES:
                continue
            name = node["label"]
            if name not in nodes:
                order.append(name)
                nodes[name] = node
                values[name] = []
            values[name].append(node.get("has numerical value", node.get("has string value")))

    # each parameter's field in the description (for its data type)
    field_of: dict[str, dict[str, Any]] = {}
    for field in (n for n in graph if n.get("@type") == "Field"):
        variable = by_id.get(_ref(field.get("represents"))) or {}
        name = variable.get("label")
        if variable.get("@type") not in PARAMETER_TYPES or name in field_of:
            continue
        field_of[name] = field

    out: list[dict[str, Any]] = [{
        "@id": "local:parameter_file_object", "@type": "cr:FileObject",
        "encodingFormat": "application/json", "label": file_object.get("label", "parameter.json"),
        "name": file_object.get("label", "parameter.json"),
        "dcterms:description": "The parameter file of one benchmark configuration.",
    }]
    files = _Files((mapping or {}).get("implementation"))
    for name in order:
        node, slug = nodes[name], slugify(name)
        variable = {
            "@id": f"local:variable_{slug}", "@type": PARAMETER_TYPES[node["@type"]], "label": name,
            **({"dcterms:description": node["dcterms:description"]} if node.get("dcterms:description") else {}),
            **({"schema:alternateName": node["schema:alternateName"]} if node.get("schema:alternateName") else {}),
            "has unit": node.get("has unit", {"@id": "unit:UNITLESS"}),
            **({"has quantity kind": node["has quantity kind"]} if node.get("has quantity kind") else {}),
        }
        key = "has string value" if node["@type"] == "text parameter" else "has numerical value"
        variable[key] = _values(values[name])
        field = {
            "@id": f"local:field_{slug}", "@type": "Field",
            "dataType": (field_of.get(name) or {}).get("dataType", {"@id": "schema:Float"}),
            "represents": {"@id": variable["@id"]},
            "source": {"@id": f"local:source_{slug}"},
        }
        where = _implementation(mapping, name, files)
        if where:
            field["bm:implementation"] = where
        out += [variable, field, {
            "@id": f"local:source_{slug}", "@type": "cr:DataSource",
            "extract": {"jsonPath": "/" + config_key(name, (variable["has unit"] or {}).get("@id"), unit_symbols)},
            "file object": {"@id": "local:parameter_file_object"},
        }]

    out[1:1] = list(files.nodes.values())          # the implementation's files, after the parameter file
    context = dict(benchmark.get("@context") or {})
    context.setdefault("bm", BM)
    for term in ("contentUrl", "encodingFormat"):    # Croissant's schema.org terms
        context.setdefault(term, f"schema:{term}")
    context.setdefault("sha256", "schema:sha256")
    context.setdefault("containedIn", "cr:containedIn")
    label = next((n.get("label") for n in graph if n.get("@type") == "m4i:Benchmark"), None)
    if label:
        out.insert(0, {"@id": "local:parameters", "@type": "schema:Dataset",
                       "schema:name": f"Input parameters of the {label} benchmark",
                       "schema:distribution": [{"@id": "local:parameter_file_object"}]
                       + [{"@id": i} for i in files.nodes],
                       "schema:hasPart": [{"@id": f"local:field_{slugify(n)}"} for n in order]})
    return {"@context": context, "@graph": out}


def write_parameters_croissant(benchmark_path: Path, mapping_path: Path | None, output: Path) -> int:
    """Write the description for `benchmark_path`; returns the number of parameters."""
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path and mapping_path.is_file() else None
    doc = build_parameters_croissant(benchmark, mapping)
    output.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return sum(1 for n in doc["@graph"] if n.get("@type") == "Field")
