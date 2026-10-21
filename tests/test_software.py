# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""The software descriptions and the run config, checked against the small
example implementations in tests/fixtures/<software>/. Run with:
python3 -m pytest tests"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import software  # noqa: E402
from software.autoconfig import CACHE_NAME, check  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
DESCRIPTIONS = sorted(software.DESCRIPTIONS_DIR.glob("*.yaml"))


def example_repo(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    shutil.copytree(FIXTURES / name, root)
    return root


def args(**values) -> argparse.Namespace:
    return argparse.Namespace(**values)


@pytest.mark.parametrize("path", DESCRIPTIONS, ids=lambda p: p.stem)
def test_software_name_finds_the_description_and_it_works(path, tmp_path):
    if not (FIXTURES / path.stem).is_dir():
        pytest.skip(f"no example implementation in tests/fixtures/{path.stem}/")
    root = example_repo(tmp_path, path.stem)
    name = yaml.safe_load(path.read_text())["name"]
    assert software.find_description(name) == path
    adapter = software.resolve(root, name, args())       # no LLM: the description fits
    module_dir = adapter.find_module_dir(root, None)
    assert adapter.discover_parameters(module_dir), "no inputs found"
    assert adapter.discover_metrics(module_dir, root), "no metrics found"
    cases = adapter.discover_cases(module_dir)
    assert cases, "no cases found"
    selected = adapter.default_scenario_candidates(adapter.discover_parameters(module_dir))
    fields = {f"{c.section}::{c.key}": {"ini": (c.section, c.key), "index": 0, "full_value": True}
              for c in selected}
    for case_dir, _ in cases:
        assert set(adapter.resolve_case_params(case_dir, fields)) == set(fields)


@pytest.mark.parametrize("path", DESCRIPTIONS, ids=lambda p: p.stem)
def test_check_accepts_working_and_rejects_broken_descriptions(path, tmp_path):
    if not (FIXTURES / path.stem).is_dir():
        pytest.skip(f"no example implementation in tests/fixtures/{path.stem}/")
    root = example_repo(tmp_path, path.stem)
    spec = yaml.safe_load(path.read_text())
    assert check(spec, root, None) == []
    broken = copy.deepcopy(spec)
    broken["cases"]["marker"] = "nothing-here.txt"
    assert check(broken, root, None)


def test_unknown_software_asks_the_llm_once(tmp_path, monkeypatch):
    import ai.inference

    root = example_repo(tmp_path, "dumux")
    calls = []

    def fake_llm(**kw):
        calls.append(kw["item_label"])
        spec = yaml.safe_load((software.DESCRIPTIONS_DIR / "dumux.yaml").read_text())
        return kw["validator"]([dict(spec, name="MyCode")])

    monkeypatch.setattr(ai.inference, "_query_llm_json_array", fake_llm)
    run = args(save_software_description=False)
    assert software.resolve(root, "MyCode", run).name == "MyCode"
    assert calls == ["software description"]
    assert (root / CACHE_NAME).is_file()
    software.resolve(root, "MyCode", run)                 # the cached description is reused
    assert calls == ["software description"]


def test_a_partial_description_file_keeps_its_settings(tmp_path, monkeypatch):
    import ai.inference

    root = example_repo(tmp_path, "dumux")
    partial = tmp_path / "mine.yaml"
    partial.write_text("name: DuMux\nrun:\n  command: mpirun ./{executable}\n")

    def fake_llm(**kw):
        return kw["validator"]([yaml.safe_load((software.DESCRIPTIONS_DIR / "dumux.yaml").read_text())])

    monkeypatch.setattr(ai.inference, "_query_llm_json_array", fake_llm)
    adapter = software.resolve(root, "DuMux", args(software_description=str(partial),
                                                    save_software_description=False))
    assert adapter.run_settings["command"] == "mpirun ./{executable}"      # the user's setting
    assert adapter.run_settings["input_argument"] == '-{input} "{value}"'  # filled in


def test_run_config_plain_keys(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("implementation: /x\nsoftware: DuMux\ncontainer: img:1\nreference: none\n"
                    "llm_model: m\nreview_below_confidence: 0.99\n"
                    "zip-name-flag: Problem.Name\n")      # a command-line option also works
    assert config.load_config_file(path) == {
        "module_dir": "/x", "software": "DuMux", "container_image": "img:1", "no_reference": True,
        "model": "m", "review_confidence_threshold": 0.99,
        "zip_name_flag": "Problem.Name"}


def test_results_zip_name_comes_from_the_description():
    import argparse

    from snakefile.styles import command_line
    from software.generic import GenericAdapter

    dumux = GenericAdapter.from_spec(yaml.safe_load((software.DESCRIPTIONS_DIR / "dumux.yaml").read_text()))()
    openfoam = GenericAdapter.from_spec(yaml.safe_load((software.DESCRIPTIONS_DIR / "openfoam.yaml").read_text()))()
    assert command_line.zip_name_flag(dumux, argparse.Namespace(zip_name_flag=None)) == "Problem.Name"
    assert command_line.zip_name_flag(openfoam, argparse.Namespace(zip_name_flag=None)) is None


def test_template_documents_every_key():
    template = (ROOT / "software" / "template.yaml").read_text()

    def keys(d: dict):
        for k, v in d.items():
            yield k
            if isinstance(v, dict):
                yield from keys(v)

    for path in DESCRIPTIONS:
        missing = {k for k in keys(yaml.safe_load(path.read_text())) if f"{k}:" not in template}
        assert not missing, f"{path.name}: not documented in software/template.yaml: {missing}"


def test_a_file_named_in_the_run_config_is_found_next_to_it(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "my-software.yaml").write_text("name: Mine\n")
    path = tmp_path / "configs" / "run.yaml"
    path.write_text("implementation: /x\ncontainer: img:1\nsoftware_description: my-software.yaml\n")
    monkeypatch.chdir(tmp_path)                    # run from elsewhere
    assert config.load_config_file(path)["software_description"] == str(tmp_path / "configs" / "my-software.yaml")
