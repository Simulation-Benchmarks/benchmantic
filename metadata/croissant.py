# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
metadata.croissant

The benchmark's input parameters as Croissant, in exactly the shape
semantic-benchmark uses (see its docs/croissant-jsonpath.md). Per parameter:

    {"@id": "local:field_radius", "@type": "Field", "dataType": ...,
     "represents": {"@id": "local:parameter_radius"},
     "source": {"@id": "local:source_radius"}},
    {"@id": "local:parameter_radius", "@type": "numerical variable",
     "label": "radius", "has unit": ..., "has quantity kind": ...},
    {"@id": "local:source_radius", "@type": "cr:DataSource",
     "extract": {"jsonPath": "/radius[m]"},
     "file object": {"@id": "local:parameter_file_object"}},

and once:

    {"@id": "local:parameter_file_object", "@type": "cr:FileObject",
     "encodingFormat": "application/json", "label": "parameter.json"}

The jsonPath is the parameter's key in the parameter file a benchmark run
writes for each configuration: its name, with the unit symbol when it has a
known one ("/radius[m]"), as run_benchmark.py and the generated Snakefile use
it. Semantic-benchmark reads only that `source`.

Where the parameter is set in the software is linked from the field with
bm:softwareSource, in the same shape -- a cr:DataSource with the path to the
input in the software's file ("/Grid/Radial0" for a key in an ini section),
its line and column, the expression that makes the input's value, and a
cr:FileObject for that file (name, contentUrl -- a link when the
implementation came from a repository link -- encodingFormat, sha256; a file
of a zipped case is containedIn the zip). It comes from the mapping
(<name>_mapping.json).

Built from the benchmark description benchmantic writes anyway, so it always
agrees with it.
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
                "@id": node_id, "@type": "cr:FileObject", "label": name, "name": name, "contentUrl": url,
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


def _json_path(name: str, unit: str | None, unit_symbols: dict[str, str]) -> str:
    """"/radius[m]": the key in the configuration's parameter file, in
    semantic-benchmark's slash-separated path syntax. That syntax splits at
    every "/", so a unit symbol with one ("rad/s") is left out:
    semantic-benchmark matches keys with or without their unit suffix
    anyway ("/omega" finds "omega[rad/s]")."""
    key = config_key(name, unit, unit_symbols)
    return "/" + (key if "/" not in key else name)


def _software_path(item: dict[str, Any]) -> str:
    """The input's place in its file, in the same slash path syntax:
    "/Grid/Radial0" for a key in an ini section, "/geom/r0" or "/radius" when
    the file itself is the section (OpenFOAM dictionaries, JSON and Python
    files)."""
    section, key = item.get("section"), item.get("key") or item.get("input", "")
    file = (item.get("location") or {}).get("file", "")
    if not section or section == file or "/" in section or "." in section:
        return "/" + key.strip("/")
    return f"/{section}/{key}".replace("//", "/")


def _software_sources(mapping: dict[str, Any] | None, name: str, files: "_Files",
                      sources: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    """The software inputs parameter `name` sets (from the mapping), each a
    cr:DataSource: the software's file it is in (a cr:FileObject), the path
    to the input in that file, line, column, and the expression that makes
    the input's value from the benchmark parameters. Inputs set from several
    parameters share one data source."""
    refs = []
    for item in (mapping or {}).get("inputs", []):
        location = item.get("location") or {}
        if name not in item.get("benchmark_parameters", []) or not location.get("file"):
            continue
        source_id = "local:software_source_" + re.sub(r"[^A-Za-z0-9]+", "_", item["input"]).strip("_")
        if source_id not in sources:
            extract = {"jsonPath": _software_path(item), "bm:line": location.get("line"),
                       "bm:column": location.get("column")}
            sources[source_id] = {
                "@id": source_id, "@type": "cr:DataSource",
                "extract": {k: v for k, v in extract.items() if v is not None},
                "file object": files.ref(location["file"]),
                "bm:input": item["input"],
                **({"bm:expression": item["expression"]} if item.get("expression") else {}),
            }
        refs.append({"@id": source_id})
    return refs


def build_parameters_croissant(benchmark: dict[str, Any], mapping: dict[str, Any] | None = None,
                               unit_symbols: dict[str, str] | None = None) -> dict[str, Any]:
    """The Croissant description of the input parameters in `benchmark` (a
    benchmark description as benchmantic writes it), in semantic-benchmark's
    shape: per parameter a Field, the variable it represents, and a
    cr:DataSource with the jsonPath into the configuration's parameter file.
    With `mapping` (its <name>_mapping.json), each field also has
    bm:softwareSource: where the parameter is set in the software, in the
    same DataSource/FileObject shape."""
    unit_symbols = unit_symbols or DEFAULT_UNIT_SYMBOLS
    graph = benchmark.get("@graph", [])
    by_id = {n.get("@id"): n for n in graph if isinstance(n, dict)}
    file_label = by_id.get("local:parameter_file_object", {}).get("label", "parameter.json")

    # the parameters, in the order the configurations name them
    order: list[str] = []
    nodes: dict[str, dict[str, Any]] = {}
    for configuration in (n for n in graph if n.get("@type") == "m4i:ParameterSet"):
        for part in configuration.get("has part", []):
            node = by_id.get(_ref(part)) or {}
            if node.get("@type") in PARAMETER_TYPES and node["label"] not in nodes:
                order.append(node["label"])
                nodes[node["label"]] = node

    # each parameter's field in the description (for its data type)
    data_type: dict[str, Any] = {}
    for field in (n for n in graph if n.get("@type") == "Field"):
        variable = by_id.get(_ref(field.get("represents"))) or {}
        if variable.get("@type") in PARAMETER_TYPES and field.get("dataType"):
            data_type.setdefault(variable.get("label"), field["dataType"])

    out: list[dict[str, Any]] = []
    files = _Files((mapping or {}).get("implementation"))
    sources: dict[str, dict[str, Any]] = {}
    for name in order:
        node, slug = nodes[name], slugify(name)
        unit = node.get("has unit") or {"@id": "unit:UNITLESS"}
        default_type = "schema:Text" if node["@type"] == "text parameter" else "schema:Float"
        variable = {"@id": f"local:parameter_{slug}", "@type": PARAMETER_TYPES[node["@type"]], "label": name,
                    "has unit": unit}
        if node.get("has quantity kind"):
            variable["has quantity kind"] = node["has quantity kind"]
        field = {"@id": f"local:field_{slug}", "@type": "Field",
                 "dataType": data_type.get(name, {"@id": default_type}),
                 "represents": {"@id": variable["@id"]}, "source": {"@id": f"local:source_{slug}"}}
        new = set(sources)
        software = _software_sources(mapping, name, files, sources)
        if software:
            field["bm:softwareSource"] = software
        out += [
            field,
            variable,
            {"@id": f"local:source_{slug}", "@type": "cr:DataSource",
             "extract": {"jsonPath": _json_path(name, unit.get("@id"), unit_symbols)},
             "file object": {"@id": "local:parameter_file_object"}},
        ] + [node for source_id, node in sources.items() if source_id not in new]
    out.append({"@id": "local:parameter_file_object", "@type": "cr:FileObject",
                "encodingFormat": "application/json", "label": file_label})
    out += list(files.nodes.values())                   # the software's files
    context = dict(benchmark.get("@context") or {})
    if sources:
        context.setdefault("bm", BM)
        for term in ("contentUrl", "encodingFormat", "sha256"):    # Croissant's schema.org terms
            context.setdefault(term, f"schema:{term}")
        context.setdefault("containedIn", "cr:containedIn")
    return {"@context": context, "@graph": out}


def write_parameters_croissant(benchmark_path: Path, mapping_path: Path | None, output: Path) -> int:
    """Write the description for `benchmark_path`; returns the number of parameters."""
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path and mapping_path.is_file() else None
    doc = build_parameters_croissant(benchmark, mapping)
    output.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return sum(1 for n in doc["@graph"] if n.get("@type") == "Field")
