# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
metadata.croissant

The benchmark's input parameters as a Croissant 1.1 description
(https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html):
where each parameter is, and how to get its value out of the file.

  distribution   one cr:FileObject per parameter file (path, sha256)
  recordSet      one cr:RecordSet per parameter file, one cr:Field per
                 parameter in it, each with a cr:source:
                   fileObject  the parameter file
                   extract     {"fileProperty": "content"} -- the file's text
                   transform   [{"regex": ...}] whose one capture group is
                               the value (plus {"separator": " "} for a
                               parameter that is a list of values)

The regex finds the value one of two ways (--croissant-anchor):

  key   (default) by section and key, e.g. [Grid] ... Radial0 = <value>;
        works on every case file, wherever the entry moved to
  line  by position: skip to the entry's line and column; exact for the
        template file, and still checks the key is on that line

The location itself is also written on each field, in benchmantic's own
terms (bm:), since Croissant has no property for a line or column:
  bm:input     the software's name for the input, e.g. "Grid.Radial0"
  bm:location  {bm:file, bm:line, bm:column, bm:endColumn} of the value
  bm:unit      the QUDT unit, when inference ran

extract_values() applies a description's extract and transform steps the
way the spec defines them; the tests use it to check that every field
gives back the value in the file.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from metadata.mapping import neutral_name
from metadata.parameters import ParameterCandidate

CROISSANT_SUFFIX = "_croissant.json"
CONFORMS_TO = "http://mlcommons.org/croissant/1.1"
BM_NAMESPACE = "https://github.com/Simulation-Benchmarks/benchmantic#"

#: Croissant 1.1's JSON-LD context (spec appendix 1), with schema.org as
#: https as mlcroissant expects, plus the bm: terms used here.
CONTEXT: dict[str, Any] = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "sc": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "dct": "http://purl.org/dc/terms/",
    "bm": BM_NAMESPACE,
    "arrayShape": "cr:arrayShape",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "containedIn": "cr:containedIn",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "excludes": "cr:excludes",
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isArray": "cr:isArray",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "readLines": "cr:readLines",
    "sdVersion": "cr:sdVersion",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
    "unArchive": "cr:unArchive",
    "value": "cr:value",
}

_DATATYPES = {"schema:Float": "sc:Float", "schema:Double": "sc:Float", "schema:Number": "sc:Float",
              "schema:Integer": "sc:Integer", "schema:Boolean": "sc:Boolean", "schema:String": "sc:Text",
              "schema:Text": "sc:Text"}
_INT = re.compile(r"^[-+]?\d+$")
_FLOAT = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")

# --- regexes ---------------------------------------------------------------------------------------
#: One value token: INI values split on whitespace (a # starts a comment);
#: OpenFOAM values also drop list parentheses and end at ";".
_INI_TOKEN = r"[^\s#]+"
_FOAM_TOKEN = r"[^\s();]+"


def _ini_key_regex(c: ParameterCandidate, index: int | None) -> str:
    head = (r"(?ms)\A.*?^[ \t]*\[" + re.escape(c.section) + r"\][ \t]*$"
            r"(?:(?!^[ \t]*\[).)*?"                     # stay inside the section
            r"^[ \t]*" + re.escape(c.key) + r"[ \t]*[=:][ \t]*")
    if index is None:
        return head + r"([^\n#]*?)[ \t]*(?:#[^\n]*)?$"
    return head + r"(?:" + _INI_TOKEN + r"[ \t]+){" + str(index) + r"}(" + _INI_TOKEN + r")"


def _foam_key_regex(c: ParameterCandidate, index: int | None) -> str:
    *blocks, last = c.key.split("/")
    head = r"(?ms)\A"
    for block in blocks:
        head += r".*?(?<![\w$])" + re.escape(block) + r"\s*\{"
    # a top-level entry starts its line; a nested one may be indented
    indent = r"[ \t]*" if blocks else ""
    head += r".*?^" + indent + re.escape(last) + r"[ \t]+"
    if index is None:
        return head + r"([^;]*?)\s*;"
    return head + r"[\s(]*(?:" + _FOAM_TOKEN + r"[\s()]+){" + str(index) + r"}(" + _FOAM_TOKEN + r")"


def _line_regex(c: ParameterCandidate, index: int | None) -> str:
    """By position: skip line-1 lines and column-1 characters, then take
    the value -- after checking the key starts that line."""
    loc = c.location
    spans = loc.token_spans
    if not spans:
        raise ValueError(f"no value position for {c.section} {c.key}")
    if index is None:   # the whole value, as written (e.g. "(0 0 1)")
        start, end = loc.column, loc.column + len(c.value.strip())
    else:
        start, end = spans[index]
    key = c.key.split("/")[-1]
    return (r"(?s)\A(?:[^\n]*\n){" + str(loc.line - 1) + r"}"
            r"(?=[ \t]*" + re.escape(key) + r"(?![\w.]))"
            r"[^\n]{" + str(start - 1) + r"}([^\n]{" + str(end - start) + r"})")


def value_regex(c: ParameterCandidate, index: int | None, fmt: str, anchor: str = "key") -> str:
    """The regex whose capture group is this parameter's value: token
    `index` of the entry, or the whole value if index is None."""
    if anchor == "line":
        return _line_regex(c, index)
    if fmt == "foam_dict":
        return _foam_key_regex(c, index)
    return _ini_key_regex(c, index)


# --- the description -------------------------------------------------------------------------------
def _datatype(spec: dict[str, Any] | None, token: str) -> str:
    if spec and spec.get("datatype") in _DATATYPES:
        return _DATATYPES[spec["datatype"]]
    if _INT.match(token):
        return "sc:Integer"
    if _FLOAT.match(token):
        return "sc:Float"
    if token.lower() in {"true", "false"}:
        return "sc:Boolean"
    return "sc:Text"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_") or "file"


def _parameters(candidates: list[ParameterCandidate], fields: dict[str, Any] | None):
    """(candidate, token index or None for the whole value, name, field
    spec) for each parameter. With `fields` (the builder's reviewed
    parameters), exactly those, under their neutral names; without,
    every candidate, named from its section and key."""
    by_key = {(c.section, c.key): c for c in candidates}
    if fields:
        for spec in fields.values():
            c = by_key.get(tuple(spec["ini"]))
            if c is None:
                continue
            index = None if spec.get("full_value") else int(spec.get("index") or 0)
            name = spec.get("label") or neutral_name(spec.get("semantic_name") or f"{c.section}_{c.key}")
            yield c, index, name, spec
        return
    for c in candidates:
        index = 0 if len(c.tokens) == 1 else None
        # INI keys repeat across sections (Grid.Name, Problem.Name); a
        # dictionary entry's path is already unique within its file
        name = neutral_name(c.key.replace("/", "_") if "/" in c.section else f"{c.section}_{c.key}")
        yield c, index, name, None


def build_croissant(adapter: Any, module_dir: Path, candidates: list[ParameterCandidate],
                    fields: dict[str, Any] | None = None, manifest: dict[str, Any] | None = None,
                    anchor: str = "key", base_dir: Path | None = None) -> dict[str, Any]:
    """The Croissant description of the parameters in the template case.
    Each file's contentUrl is relative to `base_dir` -- the folder the
    description is written to, as Croissant resolves it -- or, without
    one, to `module_dir`."""
    fmt = adapter._section("inputs").get("format", "ini") if hasattr(adapter, "_section") else "ini"
    case_dir = adapter.template_case(module_dir) if hasattr(adapter, "template_case") else module_dir
    m = manifest or {}
    label = m.get("label") or module_dir.name

    files: dict[str, dict[str, Any]] = {}
    record_sets: dict[str, dict[str, Any]] = {}
    used_ids: set[str] = set()
    for c, index, name, spec in _parameters(candidates, fields):
        if c.location is None:
            continue
        rel = c.location.file
        file_id = rel
        if file_id not in files:
            path = case_dir / rel
            if base_dir is not None:
                content_url = os.path.relpath(path.resolve(), Path(base_dir).resolve())
            else:
                content_url = str(path.relative_to(module_dir)) if path.is_relative_to(module_dir) else rel
            files[file_id] = {
                "@type": "cr:FileObject", "@id": file_id, "name": Path(rel).name,
                "description": f"{getattr(adapter, 'name', 'Simulation')} parameter file",
                "contentUrl": content_url, "encodingFormat": "text/plain",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            rs_id = _slug(rel)
            record_sets[file_id] = {"@type": "cr:RecordSet", "@id": rs_id, "name": rs_id,
                                    "description": f"The parameters set in {rel}.", "field": []}
        rs = record_sets[file_id]
        field_id = f"{rs['@id']}/{name}"
        n = 2
        while field_id in used_ids:
            field_id, n = f"{rs['@id']}/{name}_{n}", n + 1
        used_ids.add(field_id)

        tokens = c.location.token_spans
        if index is None:
            start, end = c.location.column, c.location.column + len(c.value.strip())
        elif index < len(tokens):
            start, end = tokens[index]
        else:
            start = end = c.location.column
        transform: list[dict[str, Any]] = [{"regex": value_regex(c, index, fmt, anchor)}]
        is_list = index is None and len(c.tokens) > 1 and fmt == "ini"
        if is_list:
            transform.append({"separator": " "})
        token = c.tokens[index] if index is not None and index < len(c.tokens) else (c.tokens[0] if c.tokens else "")
        dtype = _datatype(spec, token) if (index is not None or is_list) else "sc:Text"

        field: dict[str, Any] = {
            "@type": "cr:Field", "@id": field_id, "name": field_id.split("/", 1)[1],
            "description": (spec or {}).get("description") or f"{adapter.input_display(c.section, c.key)}"
                           + (f", value {index + 1} of {len(c.tokens)}" if index is not None and len(c.tokens) > 1
                              else ""),
            "dataType": dtype,
            "source": {"fileObject": {"@id": file_id}, "extract": {"fileProperty": "content"},
                       "transform": transform},
            "bm:input": adapter.input_display(c.section, c.key),
            "bm:location": {"bm:file": rel, "bm:line": c.location.line,
                            "bm:column": start, "bm:endColumn": end},
        }
        if is_list:
            field["isArray"] = True
            field["arrayShape"] = "-1"
        if index is not None and len(c.tokens) > 1:
            field["bm:tokenIndex"] = index
        if spec and spec.get("unit"):
            field["bm:unit"] = spec["unit"]
        rs["field"].append(field)

    doc: dict[str, Any] = {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": CONFORMS_TO,
        "name": _slug(label) + "_parameters",
        "description": f"Input parameters of the {label} benchmark ({getattr(adapter, 'name', '')}): "
                       "where each one is in its parameter file, and how to extract it.",
        "distribution": list(files.values()),
        "recordSet": list(record_sets.values()),
    }
    if m.get("repo_url"):
        doc["url"] = m["repo_url"]
    if m.get("license_url"):
        doc["license"] = m["license_url"]
    doc["datePublished"] = m.get("date_published") or datetime.date.today().isoformat()
    return doc


def relocate(doc: dict[str, Any], from_dir: Path, to_dir: Path) -> None:
    """Rewrite each relative contentUrl for a description moved from
    `from_dir` to `to_dir`, so it still points at the same file."""
    for f in doc.get("distribution", []):
        url = f.get("contentUrl", "")
        if url and "://" not in url and not os.path.isabs(url):
            target = os.path.normpath(os.path.join(Path(from_dir).resolve(), url))
            f["contentUrl"] = os.path.relpath(target, Path(to_dir).resolve())


def write_croissant(doc: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


# --- reading values back -----------------------------------------------------------------------------
def _cast(value: Any, dtype: str) -> Any:
    if isinstance(value, list):
        return [_cast(v, dtype) for v in value if v != ""]
    if dtype == "sc:Integer":
        return int(value)
    if dtype == "sc:Float":
        return float(value)
    if dtype == "sc:Boolean":
        return str(value).strip().lower() in {"true", "1", "yes", "on"}
    return value


def extract_values(doc: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """{field @id: value}: each field's extract and transform steps applied
    to its file under `base_dir`, as the Croissant spec defines them
    (content is the file's text; regex keeps its capture group; separator
    splits). Raises ValueError for a regex that doesn't match."""
    files = {f["@id"]: f for f in doc.get("distribution", [])}
    values: dict[str, Any] = {}
    for rs in doc.get("recordSet", []):
        for field in rs.get("field", []):
            source = field["source"]
            f = files[source["fileObject"]["@id"]]
            path = Path(os.path.normpath(Path(base_dir).resolve() / f["contentUrl"]))
            prop = source.get("extract", {}).get("fileProperty", "content")
            value: Any = {"content": lambda: path.read_text(encoding="utf-8"),
                          "filename": lambda: path.name, "fullpath": lambda: f["contentUrl"]}[prop]()
            for step in source.get("transform", []):
                if "regex" in step:
                    match = re.match(step["regex"], value)
                    if match is None:
                        raise ValueError(f"{field['@id']}: regex does not match {path}")
                    value = next(g for g in match.groups() if g is not None)
                elif "separator" in step:
                    value = value.split(step["separator"])
            values[field["@id"]] = _cast(value, field.get("dataType", "sc:Text"))
    return values
