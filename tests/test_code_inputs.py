# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""Implementations given as a link to a file, with the inputs written in the
code (the `python` input format) and the metrics computed by the code
(`metrics: {from: variables}`). Uses tests/fixtures/python_script/. Run
with: python3 -m pytest tests"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metadata.repo_source import split_link  # noqa: E402
from software.formats import python_code  # noqa: E402
from software.formats import python_code_runtime as runtime  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "python_script"
NOTEBOOK = "cooling.ipynb"


@pytest.mark.parametrize("link, expected", [
    ("https://github.com/jorgensd/dolfinx-tutorial/blob/main/chapter2/ns_code2.ipynb",
     ("https://github.com/jorgensd/dolfinx-tutorial", "main", "chapter2/ns_code2.ipynb")),
    ("https://github.com/org/repo/tree/v1.0/benchmarks/cylinder/",
     ("https://github.com/org/repo", "v1.0", "benchmarks/cylinder")),
    ("https://gitlab.com/group/sub/project/-/blob/dev/run.py",
     ("https://gitlab.com/group/sub/project", "dev", "run.py")),
    ("https://github.com/org/repo/tree/main", ("https://github.com/org/repo", "main", None)),
    ("https://github.com/org/repo", ("https://github.com/org/repo", None, None)),
    ("git@github.com:org/repo.git", ("git@github.com:org/repo.git", None, None)),
])
def test_links_to_a_folder_or_file(link, expected):
    assert split_link(link) == expected


def test_a_local_file_path_is_its_folder_and_name():
    assert split_link(str(FIXTURE / NOTEBOOK)) == (str(FIXTURE), None, NOTEBOOK)


CODE = """\
import numpy as np
L = 2.2
c_x = c_y = 0.2
dt = 1 / 1600  # time step
points = (0.15, 0.2)
mu = Constant(mesh, PETSc.ScalarType(0.001))  # viscosity
v_cg2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
disk = addDisk(c_x, c_y, 0, 0.05)
res_min = L / 3
name = "results"
_hidden = 3
if __name__ == "__main__":
    steps = 10
for i in range(3):
    inside_loop = 1
cells = find_cells(mesh)
front = cells.links(0)
fig = plt.figure(figsize=(25, 8))
"""


def test_constants_in_code():
    found = {c["name"]: c for c in runtime.constants(CODE)}
    assert {n: c["value"] for n, c in found.items()} == {
        "L": 2.2, "c_x": 0.2, "c_y": 0.2, "dt": 1 / 1600, "points": [0.15, 0.2], "mu": 0.001, "v_cg2": 2,
        "steps": 10}
    assert found["mu"]["wrapped"] and not found["dt"]["wrapped"]
    # computed values, strings, private names, loops, and calls mixing in other inputs are not inputs
    assert not {"res_min", "name", "_hidden", "inside_loop", "disk"} & set(found)
    # an index into something computed, and numbers in keyword options, are not inputs either
    assert not {"front", "fig"} & set(found)


def test_set_constants_keeps_the_code_around_the_number():
    new = runtime.set_constants(CODE, {"mu": "0.002", "dt": "0.001", "points": "0.1 0.3", "steps": "5"})
    assert "mu = Constant(mesh, PETSc.ScalarType(0.002))  # viscosity" in new
    assert "dt = 0.001  # time step" in new
    assert "points = (0.1, 0.3)" in new
    assert "    steps = 5" in new
    assert {c["name"]: c["value"] for c in runtime.constants(new)}["mu"] == 0.002
    with pytest.raises(KeyError):
        runtime.set_constants(CODE, {"nothing": "1"})


def test_notebook_inputs_and_metrics():
    settings = {"files": [NOTEBOOK]}
    candidates = {c.key: c for c in python_code.discover(FIXTURE, settings)}
    assert set(candidates) == {"L", "k", "T_end", "dt", "origin", "u"}      # `rate` mixes two numbers
    assert candidates["dt"].value == "0.01"
    assert "notebook cell 3" in candidates["dt"].cpp_hint
    code = python_code.read_code(FIXTURE / NOTEBOOK)
    assert "# %matplotlib inline" in code                               # magics don't break the code
    line = candidates["k"].location
    assert code.splitlines()[line.line - 1][line.column - 1:].startswith("0.5")
    fields = {"k": {"ini": (NOTEBOOK, "k"), "index": 0}, "o": {"ini": (NOTEBOOK, "origin"), "full_value": True}}
    assert python_code.resolve(FIXTURE, fields, settings) == {"k": 0.5, "o": [0.0, 0.0]}
    metrics = {key: expr for key, expr, _ in python_code.result_variables(code)}
    assert metrics == {"final_temperature": "u", "error": "error", "temperature_max": "max(temperature)",
                       "temperature_final": "temperature[-1]"}                  # `times` is left out


def test_the_prepared_copy_runs_and_writes_the_metrics(tmp_path):
    code = python_code.read_code(FIXTURE / NOTEBOOK)
    code = runtime.set_constants(code, {"k": "1.0", "dt": "0.001"})
    metrics_file = tmp_path / "solution_metrics.json"
    code += runtime.metrics_footer({"final_temperature": "u", "error": "error", "missing": "nothing"},
                                   metrics_file)
    (tmp_path / "run.py").write_text(code)
    subprocess.run([sys.executable, "run.py"], cwd=tmp_path, check=True, capture_output=True)
    values = json.loads(metrics_file.read_text())
    assert values["final_temperature"] == pytest.approx(0.1352, abs=1e-3)     # e^-2 with Euler steps
    assert values["missing"] is None


def _answers(candidates):
    return [{"semantic_name": re.sub(r"\W+", "_", c.key.lower()), "ini": [c.section, c.key],
             "datatype": "schema:Float", "unit": "unit:UNITLESS", "quantityKind": None, "index": 0,
             "full_value": len(c.tokens) > 1, "confidence": 0.99, "explanation": "test"} for c in candidates]


def _metric_answers(candidates):
    return [{"key": c.key, "semantic_name": c.key, "datatype": "schema:Float", "unit": "unit:UNITLESS",
             "quantityKind": None, "confidence": 0.99, "explanation": "test"} for c in candidates]


def test_describe_a_notebook_given_by_its_path(tmp_path, monkeypatch):
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
    implementation = tmp_path / "repo" / "examples"
    shutil.copytree(FIXTURE, implementation)
    describe_benchmark.main([str(implementation / NOTEBOOK), "--software", "FEniCSx", "--container-image",
                             "dolfinx/dolfinx:stable", "--no-reference", "--no-mardi-lookup", "--skip-review",
                             "--skip-validation", "--fallback-on-error", "--outputs", "standard"])
    out = tmp_path / "outputs" / "fenicsx"
    [description] = out.glob("*_benchmark.jsonld")
    assert description.name == "cooling_of_a_rod_benchmark.jsonld"           # from the notebook's title
    text = description.read_text()
    assert "temperature_max" in text and "final_temperature" in text
    snakefile = (out / "Snakefile").read_text()
    assert "code_script = 'cooling.ipynb'" in snakefile
    assert "'max(temperature)'" in snakefile and "def set_constants(" in snakefile
    assert re.search(r"'k': f'\{\w+\}'", snakefile)

    pytest.importorskip("semantic_benchmark")
    import verify_description
    args = verify_description.build_arg_parser().parse_args([str(description)])
    assert verify_description.run(args) == 0

    # run the Snakefile's rule as Snakemake would (no container: plain python3),
    # in the case folder the generator prepared
    [case_dir] = [p.parent for p in (out / ".generate_snakefile_workdir").rglob("parameters.json")]
    shutil.copy(out / "Snakefile", case_dir / "Snakefile")
    parameters = json.loads((case_dir / "parameters.json").read_text())
    [k_key] = [key for key in parameters if key == "k" or key.startswith("k_")]
    parameters[k_key] = 1.0                       # a case with another cooling rate than the code's 0.5
    (case_dir / "parameters.json").write_text(json.dumps(parameters))
    _run_snakefile_rule(case_dir / "Snakefile", case_dir)
    values = json.loads((case_dir / "solution_metrics.json").read_text())
    assert values["final_temperature"] == pytest.approx(0.99 ** 200)     # Euler steps with k = 1.0, dt = 0.01
    assert not (case_dir / ".implementation").exists()                       # used the local folder, no clone


def _run_snakefile_rule(snakefile: Path, workdir: Path) -> None:
    """Snakemake isn't needed to check the generated rule: its Python
    (everything before the rules, then the `run:` block) is run with a
    `shell()` that calls bash."""
    import os
    import textwrap
    import types

    text = snakefile.read_text()
    head, rules = text.split("\nrule all:", 1)
    body = textwrap.dedent(rules.split("    run:\n", 1)[1])
    namespace = {"workflow": types.SimpleNamespace(snakefile=str(snakefile)),
                 "shell": lambda cmd: subprocess.run(["bash", "-c", cmd.replace("python3 ", f"{sys.executable} ")],
                                                     check=True, capture_output=True)}
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        exec(compile(head, str(snakefile), "exec"), namespace)  # noqa: S102 -- the generated Snakefile
        exec(compile(body, str(snakefile), "exec"), namespace)  # noqa: S102
    finally:
        os.chdir(cwd)


def test_the_llm_describes_an_unknown_software_from_the_linked_file(tmp_path, monkeypatch):
    import argparse

    import yaml

    import ai.inference
    import software

    root = tmp_path / "repo"
    shutil.copytree(FIXTURE, root / "examples")
    prompts = []

    def fake_llm(**kw):
        prompts.append(kw["prompt"] if "prompt" in kw else str(kw))
        spec = yaml.safe_load((software.DESCRIPTIONS_DIR / "fenicsx.yaml").read_text())
        return kw["validator"]([dict(spec, name="MyCode", metrics={"from": "variables",
                                                                     "values": {"final_temperature": "u"}})])

    monkeypatch.setattr(ai.inference, "_query_llm_json_array", fake_llm)
    args = argparse.Namespace(implementation_path=f"examples/{NOTEBOOK}", save_software_description=False)
    adapter = software.resolve(root, "MyCode", args)
    module_dir = adapter.find_module_dir(root, args)
    assert module_dir == root / "examples" and adapter.main_file.name == NOTEBOOK
    assert "the file the user pointed at: outline" in prompts[0] and "k = 0.5" in prompts[0]
    assert [m.key for m in adapter.discover_metrics(module_dir, root)] == ["final_temperature"]


def test_a_printed_dict_is_one_metric_per_entry():
    code = "x = 1.0\nresult = {'numeric': x, 'exact': 2 * x}\nprint('result:', result)\n"
    assert [(k, e) for k, e, _ in python_code.result_variables(code)] == [
        ("result_numeric", "result['numeric']"), ("result_exact", "result['exact']")]


def test_a_partial_description_on_top_of_a_known_one(tmp_path):
    """Only `metrics:` given for one benchmark: the rest is benchmantic's own
    FEniCSx description, and no LLM is asked."""
    import argparse

    import software

    root = tmp_path / "repo"
    (root / "examples").mkdir(parents=True)
    code = "L = 1.0\nk = 0.5\nu = 1.0\nfor i in range(10):\n    u = u - 0.1 * k * u\n"
    (root / "examples" / "beam.py").write_text(code)              # prints nothing: no metrics found by itself
    partial = tmp_path / "beam.yaml"
    partial.write_text("name: FEniCSx\nmetrics:\n  from: variables\n  values:\n    final_u: u\n")
    args = argparse.Namespace(software_description=str(partial), implementation_path="examples/beam.py")
    adapter = software.resolve(root, "FEniCSx", args)
    module_dir = adapter.find_module_dir(root, args)
    assert [m.key for m in adapter.discover_metrics(module_dir, root)] == ["final_u"]
    assert adapter.run_settings["type"] == "script"                          # from software_descriptions/fenicsx.yaml
