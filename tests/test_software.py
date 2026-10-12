# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""Every registered software adapter against its own example under
tests/fixtures/<slug>/. A new software folder needs a fixture; these checks
then run for it automatically."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import software  # noqa: E402
from software.base import SoftwareAdapter  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
ADAPTERS = software.ADAPTERS


@pytest.fixture
def example(tmp_path, request):
    cls = request.param
    src = FIXTURES / cls.slug
    if not src.is_dir():
        pytest.fail(f"software/{cls.slug}/ has no example: add tests/fixtures/{cls.slug}/")
    dest = tmp_path / cls.slug
    shutil.copytree(src, dest)
    return cls, dest


def test_registry_lists_dumux_and_openfoam():
    assert {"dumux", "openfoam"} <= {a.slug for a in ADAPTERS}
    assert "_template" not in software.SOFTWARE_CHOICES


@pytest.mark.parametrize("example", ADAPTERS, indirect=True, ids=lambda a: a.slug)
def test_detected_by_its_own_adapter_only(example):
    cls, root = example
    assert [a.slug for a in ADAPTERS if a.detect(root)] == [cls.slug]
    assert type(software.get_adapter(root, "auto")) is cls


@pytest.mark.parametrize("example", ADAPTERS, indirect=True, ids=lambda a: a.slug)
def test_discovers_parameters_metrics_and_cases(example):
    cls, root = example
    adapter = cls()
    module_dir = adapter.find_module_dir(root, None)
    params = adapter.discover_parameters(module_dir)
    assert params, "no parameters discovered"
    selected = adapter.default_scenario_candidates(params)
    assert selected and set(map(id, selected)) <= set(map(id, params))
    assert adapter.discover_metrics(module_dir, root), "no metrics discovered"
    cases = adapter.discover_cases(module_dir)
    assert cases, "no cases discovered"
    assert len({cid for _, cid in cases}) == len(cases), "case ids must be unique"
    fields = {f"{c.section}::{c.key}": {"ini": (c.section, c.key), "index": 0, "full_value": True}
              for c in selected}
    for case_dir, _ in cases:
        values = adapter.resolve_case_params(case_dir, fields)
        assert set(values) == set(fields)


@pytest.mark.parametrize("cls", ADAPTERS, ids=lambda a: a.slug)
def test_has_prompt_notes_and_snakefile(cls):
    assert cls.name and cls.name != "unknown"
    assert cls().prompt_context(), f"software/{cls.slug}/prompt.md is missing or empty"
    assert cls.render_snakefile is not SoftwareAdapter.render_snakefile, "render_snakefile() not implemented"


def test_new_folder_is_registered_without_other_changes(tmp_path, monkeypatch):
    """Copying _template to a new folder is all it takes to register it."""
    pkg = tmp_path / "mycode"
    shutil.copytree(ROOT / "software" / "_template", pkg)
    for f in pkg.glob("*.py"):
        f.write_text(f.read_text().replace("software._template", "software.mycode"))
    monkeypatch.setattr(software, "__path__", [*software.__path__, str(tmp_path)])
    try:
        found = software._discover()
        assert "mycode" in [a.slug for a in found]
    finally:
        for name in [m for m in sys.modules if m.startswith("software.mycode")]:
            del sys.modules[name]
