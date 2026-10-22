# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""<name>_parameters_croissant.jsonld: the input parameters in Croissant, one
cr:Field per parameter with a cr:DataSource whose jsonPath is the key in the
configuration's parameter file, in semantic-benchmark's shape (its
docs/croissant-jsonpath.md). Checked on the DuMux example. Run with:
python3 -m pytest tests"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).parent / "fixtures"


def _answers(candidates):
    out = []
    for c in candidates:
        name = re.sub(r"[^a-z0-9]+", "_", c.key.lower()).strip("_")
        numeric = bool(c.tokens) and all(re.match(r"^[-+]?[\d.eE+-]+$", t) for t in c.tokens)
        out.append({"semantic_name": name, "ini": [c.section, c.key],
                    "datatype": "schema:Float" if numeric else "schema:String",
                    "unit": "unit:M" if c.key.startswith("Radial") else "unit:UNITLESS", "quantityKind": None,
                    "index": 0, "confidence": 0.99, "explanation": f"test answer for {c.key}"})
    return out


def _metric_answers(candidates):
    return [{"key": c.key, "semantic_name": c.key, "datatype": "schema:Float", "unit": "unit:UNITLESS",
             "quantityKind": None, "confidence": 0.99, "explanation": "test"} for c in candidates]


def test_parameters_in_croissant(tmp_path, monkeypatch):
    import describe_benchmark
    from metadata import builder

    monkeypatch.setattr(builder, "infer_parameter_metadata", lambda candidates, *a, **k: _answers(candidates))
    monkeypatch.setattr(builder, "infer_metric_metadata", lambda candidates, *a, **k: _metric_answers(candidates))
    monkeypatch.setenv("BENCHMANTIC_CORRECTIONS_PATH", str(tmp_path / "corrections.json"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda *a: "")
    remove = shutil.rmtree                       # keep the generator's case folders: their parameters.json
    monkeypatch.setattr(shutil, "rmtree", lambda path, *a, **k: None if Path(path).name ==
                        ".generate_snakefile_workdir" else remove(path, *a, **k))
    implementation = tmp_path / "dumux"
    shutil.copytree(FIXTURES / "dumux", implementation)
    describe_benchmark.main([str(implementation), "--software", "DuMux", "--container-image", "test:1",
                             "--outputs", "standard", "--no-reference", "--no-mardi-lookup", "--skip-review",
                             "--skip-validation", "--fallback-on-error", "--build-dir", "/shared/build"])

    out = tmp_path / "outputs" / "dumux"
    [croissant] = out.glob("*_parameters_croissant.jsonld")
    doc = json.loads(croissant.read_text())
    nodes = {n["@id"]: n for n in doc["@graph"]}
    fields = [n for n in doc["@graph"] if n["@type"] == "Field"]
    variables = {nodes[f["represents"]["@id"]]["label"]: nodes[f["represents"]["@id"]] for f in fields}
    assert len(fields) == len(variables) > 0                                # one field per parameter

    paths = {}
    for field in fields:
        source = nodes[field["source"]["@id"]]
        assert source["@type"] == "cr:DataSource" and source["file object"]["@id"] == "local:parameter_file_object"
        paths[nodes[field["represents"]["@id"]]["label"]] = source["extract"]["jsonPath"]
    assert nodes["local:parameter_file_object"]["encodingFormat"] == "application/json"

    # every jsonPath is a key of the parameter file a configuration gets
    for parameters in (out / ".generate_snakefile_workdir").rglob("parameters.json"):
        keys = set(json.loads(parameters.read_text()))
        assert {p.lstrip("/") for p in paths.values()} <= keys
    radius = next(name for name in paths if name.startswith("radial"))
    assert paths[radius].endswith("[m]")                                    # the unit symbol, as in the file

    # semantic-benchmark's shape: the Field's one source is the parameter file
    assert {n["@type"] for n in doc["@graph"]} <= {"Field", "numerical variable", "text variable",
                                                   "cr:DataSource", "cr:FileObject"}
    assert nodes["local:parameter_file_object"] == {"@id": "local:parameter_file_object", "@type": "cr:FileObject",
                                                    "encodingFormat": "application/json", "label": "parameter.json"}
    assert set(variables["cells0"]) <= {"@id", "@type", "label", "has unit", "has quantity kind"}

    # ... and where it is set in the software, in the same shape (bm:softwareSource)
    cells_field = next(f for f in fields if nodes[f["represents"]["@id"]] is variables["cells0"])
    [software] = [nodes[r["@id"]] for r in cells_field["bm:softwareSource"]]
    assert software["@type"] == "cr:DataSource" and software["bm:input"] == "Grid.Cells0"
    assert software["extract"]["jsonPath"] == "/Grid/Cells0" and software["extract"]["bm:line"] > 0
    file = nodes[software["file object"]["@id"]]
    assert file["@type"] == "cr:FileObject" and file["name"] == "params.input"
    assert file["bm:pathInRepository"] == "params.input" and len(file["sha256"]) == 64

    pytest.importorskip("semantic_benchmark")
    import pyshacl
    import rdflib
    from semantic_benchmark.semantics.loader import BenchmarkLoader

    graph = rdflib.Graph().parse(str(croissant), format="json-ld")
    conforms, _, report = pyshacl.validate(graph, shacl_graph=BenchmarkLoader.load_shapes(), inference="none")
    assert conforms, report


def test_one_entry_per_parameter_also_when_it_varies():
    from metadata.croissant import build_parameters_croissant

    def value(name, v, n):
        return {"@id": f"local:v_{name}_{n}", "@type": "numerical parameter", "label": name,
                "has unit": {"@id": "unit:M"}, "has numerical value": v}

    graph = [value("radius", 0.1, 1), value("radius", 0.2, 2), value("length", 1.0, 1),
             {"@id": "local:c1", "@type": "m4i:ParameterSet", "has part": [{"@id": "local:v_radius_1"},
                                                                           {"@id": "local:v_length_1"}]},
             {"@id": "local:c2", "@type": "m4i:ParameterSet", "has part": [{"@id": "local:v_radius_2"},
                                                                           {"@id": "local:v_length_1"}]}]
    doc = build_parameters_croissant({"@context": {}, "@graph": graph})
    assert [n["label"] for n in doc["@graph"] if n["@type"] == "numerical variable"] == ["radius", "length"]
    paths = [n["extract"]["jsonPath"] for n in doc["@graph"] if n["@type"] == "cr:DataSource"]
    assert paths == ["/radius[m]", "/length[m]"]


def test_every_path_is_found_by_semantic_benchmarks_own_reader(tmp_path):
    """The jsonPaths, read with semantic-benchmark's _json_path_value() from a
    parameter file as a benchmark run writes it -- also for a unit with a
    slash in it (rad/s)."""
    import types

    from metadata.croissant import build_parameters_croissant

    sys.modules.setdefault("rocrate_validator", types.SimpleNamespace(models=None, services=None))
    create = pytest.importorskip("semantic_benchmark.rocrate.create")

    def value(name, unit, v):
        return {"@id": f"local:v_{name}", "@type": "numerical parameter", "label": name,
                "has unit": {"@id": unit}, "has numerical value": v}

    graph = [value("radius", "unit:M", 0.33), value("omega", "unit:RAD-PER-SEC", 2.0), value("nu", "unit:UNITLESS", 1.0),
             {"@id": "local:c", "@type": "m4i:ParameterSet", "has part": [{"@id": "local:v_radius"},
                                                                          {"@id": "local:v_omega"},
                                                                          {"@id": "local:v_nu"}]}]
    doc = build_parameters_croissant({"@context": {}, "@graph": graph})
    paths = {n["@id"]: n["extract"]["jsonPath"] for n in doc["@graph"] if n["@type"] == "cr:DataSource"}
    assert paths == {"local:source_radius": "/radius[m]", "local:source_omega": "/omega", "local:source_nu": "/nu"}
    parameter_file = {"configuration": "c", "radius[m]": 0.33, "omega[rad/s]": 2.0, "nu": 1.0}
    assert [create._json_path_value(parameter_file, p) for p in paths.values()] == [0.33, 2.0, 1.0]


def test_software_files_inside_a_repository_and_a_zip():
    from metadata.croissant import build_parameters_croissant

    graph = [{"@id": "local:v", "@type": "numerical parameter", "label": "omega", "has unit": {"@id": "unit:UNITLESS"},
              "has numerical value": 1.0},
             {"@id": "local:c", "@type": "m4i:ParameterSet", "has part": [{"@id": "local:v"}]}]
    mapping = {"inputs": [{"input": "MRF1/omega", "section": "constant/MRFProperties", "key": "MRF1/omega",
                           "expression": "{omega}", "benchmark_parameters": ["omega"],
                           "location": {"file": "constant/MRFProperties", "line": 27, "column": 15}}],
               "implementation": {"repository": "https://github.com/org/repo", "ref": "main", "folder": "openfoam",
                                  "case_folder": ".benchmantic/output_template", "archive": "output_template.zip",
                                  "path_in_archive": "."}}
    doc = build_parameters_croissant({"@context": {}, "@graph": graph}, mapping)
    nodes = {n["@id"]: n for n in doc["@graph"]}
    [field] = [n for n in doc["@graph"] if n["@type"] == "Field"]
    software = nodes[field["bm:softwareSource"][0]["@id"]]
    assert software["extract"] == {"jsonPath": "/MRF1/omega", "bm:line": 27, "bm:column": 15}
    files = {n.get("name"): n for n in doc["@graph"] if n["@type"] == "cr:FileObject"}
    archive, file = files["output_template.zip"], files["MRFProperties"]
    assert software["file object"] == {"@id": file["@id"]}
    assert archive["contentUrl"] == "https://github.com/org/repo/blob/main/openfoam/output_template.zip"
    assert file["containedIn"] == {"@id": archive["@id"]} and file["contentUrl"] == "constant/MRFProperties"
