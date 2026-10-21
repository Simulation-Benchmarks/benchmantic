# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""Benchmarks whose program reads a JSON parameter file, one file per case
(`cases.files`, input format `json`, `run.type: parameter_file`), as in the
NFDI4Ing Model Validation Platform. Uses tests/fixtures/parameter_files/.
Run with: python3 -m pytest tests"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import software  # noqa: E402
from metadata.metrics import discover_metrics_in_python  # noqa: E402
from software.autoconfig import check  # noqa: E402
from software.formats import json_params  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "parameter_files"
DESCRIPTION = FIXTURE / "plate.yaml"
PROGRAM = "plate/program"


def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    shutil.copytree(FIXTURE / "plate", root / "plate")
    return root


def test_json_entries_with_units_and_locations():
    case = FIXTURE / "plate"
    found = {c.key: c for c in json_params.discover(case, {"files": ["parameters_1.json"]})}
    assert set(found) == {"radius", "length", "young-modulus", "element-order", "quadrature-rule"}
    assert found["young-modulus"].value == "210e9"                          # as written
    assert 'unit "Pa"' in found["young-modulus"].cpp_hint
    where = found["radius"].location
    line = (case / "parameters_1.json").read_text().splitlines()[where.line - 1]
    assert line[where.column - 1:].startswith("0.2")
    fields = {"r": {"ini": ("parameters_1.json", "radius"), "index": 0},
              "q": {"ini": ("parameters_1.json", "quadrature-rule"), "index": 0}}
    assert json_params.resolve(case, fields, {}) == {"r": 0.2, "q": "gauss"}


def test_one_case_per_file_in_the_folder_above(tmp_path):
    root = repo(tmp_path)
    spec = software.load_description(str(DESCRIPTION))
    args = argparse.Namespace(implementation_path=PROGRAM)
    assert check(spec, root, args) == []
    adapter = software.resolve(root, None, argparse.Namespace(software_description=str(DESCRIPTION),
                                                             implementation_path=PROGRAM))
    module_dir = adapter.find_module_dir(root, args)
    assert module_dir == root / PROGRAM
    cases = adapter.discover_cases(module_dir)
    assert [cid for _, cid in cases] == ["05", "1"]
    fields = {"r": {"ini": ("case_parameters.json", "radius"), "index": 0}}
    assert [adapter.resolve_case_params(d, fields)["r"] for d, _ in cases] == [0.1, 0.2]
    assert [m.key for m in adapter.discover_metrics(module_dir, root)] == ["plate_area"]


def test_metrics_dumped_to_a_file_named_by_a_variable():
    text = (FIXTURE / PROGRAM / "run_simulation.py").read_text()
    assert [m.key for m in discover_metrics_in_python(text)] == ["plate_area"]


def test_a_cloned_repositorys_description_is_kept_in_outputs(tmp_path, monkeypatch):
    from metadata import repo_source
    from software import autoconfig

    monkeypatch.chdir(tmp_path)
    clone = repo_source.EPHEMERAL_CLONE_ROOT / "test_clone_xyz"
    args = argparse.Namespace(repository_source="https://github.com/org/repo", repository_ref="main",
                              implementation_path="benchmarks/plate/fenics")
    path = autoconfig.cache_path(clone, args)
    assert path == Path("outputs/.software_descriptions/github_com_org_repo_main_benchmarks_plate_fenics.yaml")
    assert autoconfig.cache_path(tmp_path, argparse.Namespace()) == tmp_path / autoconfig.CACHE_NAME


def _answers(candidates):
    return [{"semantic_name": re.sub(r"\W+", "_", c.key.lower()), "ini": [c.section, c.key],
             "datatype": "schema:Float", "unit": "unit:M", "quantityKind": None, "index": 0,
             "confidence": 0.99, "explanation": "test"} for c in candidates]


def _metric_answers(candidates):
    return [{"key": c.key, "semantic_name": c.key, "datatype": "schema:Float", "unit": "unit:M2",
             "quantityKind": None, "confidence": 0.99, "explanation": "test"} for c in candidates]


def test_describe_and_run_a_case(tmp_path, monkeypatch):
    import describe_benchmark
    from metadata import builder

    monkeypatch.setattr(builder, "infer_parameter_metadata", lambda candidates, *a, **k: _answers(candidates))
    monkeypatch.setattr(builder, "infer_metric_metadata", lambda candidates, *a, **k: _metric_answers(candidates))
    monkeypatch.setenv("BENCHMANTIC_CORRECTIONS_PATH", str(tmp_path / "corrections.json"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda *a: "")
    remove = shutil.rmtree                       # keep the generator's case folders, to run the rule in one
    monkeypatch.setattr(shutil, "rmtree", lambda path, *a, **k: None if Path(path).name ==
                        ".generate_snakefile_workdir" else remove(path, *a, **k))
    root = repo(tmp_path)
    describe_benchmark.main([str(root / PROGRAM), "--software-description", str(DESCRIPTION),
                             "--container-image", "python:3.12", "--no-reference", "--no-mardi-lookup",
                             "--skip-review", "--skip-validation", "--fallback-on-error", "--outputs", "standard",
                             "--scenario-params", "radius,length"])
    out = tmp_path / "outputs" / "testcode"
    [description] = out.glob("*_benchmark.jsonld")
    assert "plate_area" in description.read_text()
    snakefile = (out / "Snakefile").read_text()
    assert "template_case_file = 'parameters_05.json'" in snakefile

    # the rule, as Snakemake would run it (no container), for case "1" with another radius
    case_dir = out / ".generate_snakefile_workdir" / "1"
    parameters = json.loads((case_dir / "parameters.json").read_text())
    assert parameters["configuration"] == "1"
    [radius_key] = [k for k in parameters if k.startswith("radius")]
    parameters[radius_key] = 0.3
    (case_dir / "parameters.json").write_text(json.dumps(parameters))
    shutil.copy(out / "Snakefile", case_dir / "Snakefile")
    _run_snakefile_rule(case_dir / "Snakefile", case_dir)
    assert json.loads((case_dir / "solution_metrics.json").read_text()) == {
        "plate_area": pytest.approx(1.0 - math.pi * 0.3**2)}
    written = json.loads((case_dir / "work" / "program" / "benchmantic_parameters.json").read_text())
    assert written["radius"] == {"value": 0.3, "unit": "m"}                  # the file's layout is kept
    assert written["configuration"] == "1" and written["quadrature-rule"] == "gauss"


def _run_snakefile_rule(snakefile: Path, workdir: Path) -> None:
    """The rule's Python (everything before the rules, then the `run:`
    block), with a `shell()` that calls bash -- Snakemake isn't needed."""
    text = snakefile.read_text()
    head, rules = text.split("\nrule all:", 1)
    body = textwrap.dedent(rules.split("    run:\n", 1)[1])
    namespace = {"workflow": types.SimpleNamespace(snakefile=str(snakefile)),
                 "shell": lambda cmd: subprocess.run(["bash", "-c", cmd.replace("python3 ", f"{sys.executable} ")
                                                      .replace("{{", "{").replace("}}", "}")],
                                                     check=True, capture_output=True)}
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        exec(compile(head, str(snakefile), "exec"), namespace)  # noqa: S102 -- the generated Snakefile
        exec(compile(body, str(snakefile), "exec"), namespace)  # noqa: S102
    finally:
        os.chdir(cwd)


def test_a_reference_with_lists_of_values_is_a_sweep():
    """One parameter set with several values per parameter is one
    configuration per combination, as semantic_benchmark expands it."""
    from metadata.reference import load_reference

    ref = load_reference(Path(__file__).parent / "fixtures" / "references" / "plate-with-hole-sweep.json")
    assert len(ref.configurations) == 8                     # 2 element sizes x 2 degrees x 2 cell types
    assert [c["id"] for c in ref.configurations][:2] == ["all--1", "all--2"]
    assert {c["values"]["element_size"] for c in ref.configurations} == {0.1, 0.003125}
    assert {c["values"]["cell_type"] for c in ref.configurations} == {"triangle", "quadrilateral"}
    assert ref.parameters["isoparametric_element_degree"]["datatype"] == "schema:Integer"
    try:
        from semantic_benchmark.semantics.loader import BenchmarkLoader
    except ImportError:
        return
    sets = BenchmarkLoader(Path(__file__).parent / "fixtures" / "references" / "plate-with-hole-sweep.json").load()
    theirs = {s.identifier: {p.label: (p.string_value if getattr(p, "string_value", None) is not None
                                       else p.numerical_value) for p in s.parts} for s in sets.parameter_sets}
    assert theirs == {c["id"]: c["values"] for c in ref.configurations}


def test_a_reference_link_is_read_from_its_raw_form(monkeypatch):
    from metadata import reference

    opened = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout: opened.append(url) or Response())
    path = reference.fetch_reference("https://github.com/org/repo/blob/main/benchmark/1.1.0/minimal-configurations.json")
    assert opened == ["https://raw.githubusercontent.com/org/repo/main/benchmark/1.1.0/minimal-configurations.json"]
    assert path.name == "minimal-configurations.json" and path.read_text() == "{}"
    reference.fetch_reference("https://gitlab.com/group/project/-/blob/v1/ref.json")
    assert opened[-1] == "https://gitlab.com/group/project/-/raw/v1/ref.json"
    assert reference.fetch_reference("some/local.json") == Path("some/local.json")
