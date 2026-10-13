# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""The parameter-file step: find the parameter file, then identify each
parameter and where it is in that file. Run with: python3 -m pytest tests"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import software  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def _dumux():
    return next(a for a in software.ADAPTERS if a.slug == "dumux")()


def _by_key(candidates):
    return {(c.section, c.key): c for c in candidates}


def test_ini_locations(tmp_path):
    root = tmp_path / "dumux"
    shutil.copytree(FIXTURES / "dumux", root)
    adapter = _dumux()
    params = _by_key(adapter.discover_parameters(adapter.find_module_dir(root, None)))

    loc = params[("Grid", "Radial0")].location
    assert (loc.file, loc.line, loc.column) == ("params.input", 2, 11)
    assert loc.token_spans == [(11, 14), (15, 18), (19, 22)]
    line = (root / "params.input").read_text().splitlines()[loc.line - 1]
    assert [line[a - 1:b - 1] for a, b in loc.token_spans] == ["1.0", "1.5", "2.0"]
    assert params[("Problem", "Name")].location.line == 9


def test_finds_a_renamed_parameter_file(tmp_path):
    root = tmp_path / "dumux"
    shutil.copytree(FIXTURES / "dumux", root)
    (root / "params.input").rename(root / "rotating.input")
    (root / "notes.ini").write_text("[x]\na = 1\n")          # a decoy with fewer entries
    main = root / "main.cc"
    main.write_text(main.read_text().replace("int main", '// reads "rotating.input"\nint main', 1))

    adapter = _dumux()
    module_dir = adapter.find_module_dir(root, None)
    params = _by_key(adapter.discover_parameters(module_dir))

    assert [f.path.name for f in adapter.parameter_files] == ["rotating.input", "notes.ini"]
    assert "named in the source code" in adapter.parameter_files[0].reasons
    assert params[("Grid", "Cells0")].location.file == "rotating.input"
    cases = adapter.discover_cases(module_dir)
    assert cases, "the found file also marks the case"
    fields = {"Grid::Cells1": {"ini": ("Grid", "Cells1"), "index": 0}}
    assert adapter.resolve_case_params(cases[0][0], fields) == {"Grid::Cells1": 32}


def test_openfoam_locations(tmp_path):
    root = tmp_path / "openfoam"
    shutil.copytree(FIXTURES / "openfoam", root)
    adapter = next(a for a in software.ADAPTERS if a.slug == "openfoam")()
    module_dir = adapter.find_module_dir(root, None)
    for c in adapter.discover_parameters(module_dir):
        assert c.location and c.location.file == c.section
        text = (adapter.template_case(module_dir) / c.section).read_text().splitlines()[c.location.line - 1]
        assert c.key.split("/")[-1] in text
        if c.location.token_spans:
            a, b = c.location.token_spans[0]
            assert text[a - 1:b - 1] == c.tokens[0]
