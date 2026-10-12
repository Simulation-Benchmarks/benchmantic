# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""Each software in software/ against its small example in
tests/fixtures/<software>/: recognised, and inputs, metrics and cases
found. Run with: python3 -m pytest tests"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import software  # noqa: E402


@pytest.mark.parametrize("cls", software.ADAPTERS, ids=lambda a: a.slug)
def test_example(cls, tmp_path):
    example = Path(__file__).parent / "fixtures" / cls.slug
    if not example.is_dir():
        pytest.fail(f"add a small example for software/{cls.slug}/ in tests/fixtures/{cls.slug}/")
    root = tmp_path / cls.slug
    shutil.copytree(example, root)

    assert type(software.get_adapter(root)) is cls, "recognised as another software"
    adapter = cls()
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


def test_a_new_software_is_just_a_folder(tmp_path, monkeypatch):
    (tmp_path / "mycode").mkdir()
    shutil.copy(ROOT / "software" / "_template" / "software.yaml", tmp_path / "mycode" / "software.yaml")
    monkeypatch.setattr(software, "__path__", [*software.__path__, str(tmp_path)])
    assert "mycode" in [a.slug for a in software._discover()]
