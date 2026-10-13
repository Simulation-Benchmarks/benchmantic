# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""The Croissant description of the parameters: every field's
extract/transform steps give back the value in the parameter file.
Run with: python3 -m pytest tests"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import software  # noqa: E402
from metadata.croissant import build_croissant, extract_values, write_croissant  # noqa: E402
from utils import to_number  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def _adapter(slug):
    return next(a for a in software.ADAPTERS if a.slug == slug)()


def _expected(c, index):
    if index is None:
        return [to_number(t) for t in c.tokens] if len(c.tokens) > 1 else c.value
    return to_number(c.tokens[index])


def _check(doc, module_dir, candidates):
    """Every field's value equals the value discovered from the file."""
    by_input = {(c.section, c.key): c for c in candidates}
    values = extract_values(doc, module_dir)
    assert values
    for rs in doc["recordSet"]:
        for field in rs["field"]:
            c = next(c for (s, k), c in by_input.items() if field["bm:input"].endswith(k)
                     and field["bm:location"]["bm:file"] in (s, Path(s).name, c.location.file))
            index = field.get("bm:tokenIndex", 0 if len(c.tokens) == 1 else None)
            got, want = values[field["@id"]], _expected(c, index)
            if isinstance(want, str) and not isinstance(got, str):
                want = to_number(want)
            if isinstance(got, bool):
                want = str(want).lower() == "true"
            assert got == want, field["@id"]
            # the location points at that value in the file
            loc = field["bm:location"]
            text = (module_dir / doc["distribution"][0]["contentUrl"]).parent  # case folder
            line = (text / loc["bm:file"]).read_text().splitlines()[loc["bm:line"] - 1] \
                if (text / loc["bm:file"]).exists() else None
            if line is not None and index is not None:
                assert line[loc["bm:column"] - 1:loc["bm:endColumn"] - 1] == c.tokens[index]


@pytest.mark.parametrize("anchor", ["key", "line"])
@pytest.mark.parametrize("slug", ["dumux", "openfoam"])
def test_fields_read_back_their_values(slug, anchor, tmp_path):
    root = tmp_path / slug
    shutil.copytree(FIXTURES / slug, root)
    adapter = _adapter(slug)
    module_dir = adapter.find_module_dir(root, None)
    candidates = adapter.discover_parameters(module_dir)
    doc = build_croissant(adapter, module_dir, candidates, anchor=anchor)
    assert doc["conformsTo"] == "http://mlcommons.org/croissant/1.1"
    assert len([f for rs in doc["recordSet"] for f in rs["field"]]) == len(candidates)
    _check(doc, module_dir, candidates)


def test_nested_openfoam_entries(tmp_path):
    root = tmp_path / "openfoam"
    shutil.copytree(FIXTURES / "openfoam", root)
    header = (root / "case/constant/transportProperties").read_text().split("transportModel")[0]
    (root / "case/constant/MRFProperties").write_text(header.replace("transportProperties", "MRFProperties") + """
MRF1
{
    cellZone    rotor;
    active      yes;
    origin      (0 0 0);
    axis        (0 0 1);
    omega       {omega};
}
nu  2e-05;
""")
    adapter = _adapter("openfoam")
    module_dir = adapter.find_module_dir(root, None)
    candidates = adapter.discover_parameters(module_dir)
    keys = {c.key for c in candidates if c.section == "constant/MRFProperties"}
    assert {"MRF1/omega", "MRF1/axis"} <= keys
    for anchor in ("key", "line"):
        doc = build_croissant(adapter, module_dir, candidates, anchor=anchor)
        values = extract_values(doc, module_dir)
        assert values["constant_MRFProperties/mrf1_omega"] == "{omega}"
        assert values["constant_MRFProperties/mrf1_axis"] == "(0 0 1)"
        # a top-level nu in another file is not confused with this one
        assert values["constant_MRFProperties/nu"] == 2e-05
        assert values["constant_transportProperties/nu"] == 1e-05


def test_key_anchor_survives_moved_lines(tmp_path):
    """A case file with the same entries on other lines: the key-anchored
    description still reads it, the line-anchored one notices."""
    root = tmp_path / "dumux"
    shutil.copytree(FIXTURES / "dumux", root)
    adapter = _adapter("dumux")
    module_dir = adapter.find_module_dir(root, None)
    candidates = adapter.discover_parameters(module_dir)
    by_key = build_croissant(adapter, module_dir, candidates, anchor="key")
    by_line = build_croissant(adapter, module_dir, candidates, anchor="line")

    params = root / "params.input"
    params.write_text("# a finer case\n\n" + params.read_text().replace("Cells1 = 32", "Cells1 = 64"))
    assert extract_values(by_key, module_dir)["params_input/grid_cells1"] == 64
    with pytest.raises(ValueError):
        extract_values(by_line, module_dir)


def test_with_reviewed_fields(tmp_path):
    """As describe_benchmark.py writes it: the reviewed parameters only,
    under their neutral names, with units, one token per field."""
    root = tmp_path / "dumux"
    shutil.copytree(FIXTURES / "dumux", root)
    adapter = _adapter("dumux")
    module_dir = adapter.find_module_dir(root, None)
    candidates = adapter.discover_parameters(module_dir)
    fields = {
        "Grid::Radial0": {"ini": ("Grid", "Radial0"), "index": 2, "label": "outer_radius",
                          "unit": "unit:M", "datatype": "schema:Double", "description": "Outer radius."},
        "Grid::Cells0": {"ini": ("Grid", "Cells0"), "full_value": True, "label": "cells_radial",
                         "unit": "unit:UNITLESS", "datatype": "schema:Integer"},
        "Problem::Omega1": {"ini": ("Problem", "Omega1"), "index": 0, "label": "angular_velocity",
                            "unit": "unit:RAD-PER-SEC", "datatype": "schema:Double"},
    }
    doc = build_croissant(adapter, module_dir, candidates, fields, {"label": "rotating cylinders"})
    rs = doc["recordSet"][0]
    by_name = {f["name"]: f for f in rs["field"]}
    assert set(by_name) == {"outer_radius", "cells_radial", "angular_velocity"}
    assert by_name["outer_radius"]["bm:unit"] == "unit:M"
    assert by_name["outer_radius"]["bm:location"] == {"bm:file": "params.input", "bm:line": 2,
                                                       "bm:column": 19, "bm:endColumn": 22}
    assert by_name["cells_radial"]["isArray"] is True
    assert extract_values(doc, module_dir) == {"params_input/outer_radius": 2.0,
                                               "params_input/cells_radial": [8, 8],
                                               "params_input/angular_velocity": 1.0}


def test_valid_for_mlcroissant(tmp_path):
    mlc = pytest.importorskip("mlcroissant")
    root = tmp_path / "dumux"
    shutil.copytree(FIXTURES / "dumux", root)
    adapter = _adapter("dumux")
    module_dir = adapter.find_module_dir(root, None)
    doc = build_croissant(adapter, module_dir, adapter.discover_parameters(module_dir))
    path = module_dir / "parameters_croissant.json"
    write_croissant(doc, path)
    dataset = mlc.Dataset(jsonld=str(path))
    assert not dataset.metadata.ctx.issues.errors
    assert json.loads(path.read_text())["recordSet"][0]["field"][0]["source"]["extract"] == {"fileProperty": "content"}


def test_written_elsewhere_still_finds_the_file(tmp_path):
    """Written to outputs/<software>/, or moved there: contentUrl is
    relative to the description, so the values still read back."""
    from metadata.croissant import relocate
    root = tmp_path / "dumux"
    shutil.copytree(FIXTURES / "dumux", root)
    adapter = _adapter("dumux")
    module_dir = adapter.find_module_dir(root, None)
    candidates = adapter.discover_parameters(module_dir)
    out = tmp_path / "benchmantic" / "outputs" / "dumux"
    doc = build_croissant(adapter, module_dir, candidates, base_dir=out)
    assert doc["distribution"][0]["contentUrl"] == "../../../dumux/params.input"
    assert extract_values(doc, out)["params_input/grid_cells1"] == 32
    staging = tmp_path / "staging"
    doc = build_croissant(adapter, module_dir, candidates, base_dir=staging)
    relocate(doc, staging, out)
    assert extract_values(doc, out)["params_input/grid_cells1"] == 32


def test_command_writes_to_outputs(tmp_path, monkeypatch):
    import croissant_description
    root = tmp_path / "dumux"
    shutil.copytree(FIXTURES / "dumux", root)
    monkeypatch.setattr(croissant_description, "HERE", tmp_path / "benchmantic")
    monkeypatch.chdir(root)
    assert croissant_description.main([str(root), "--check"]) == 0
    assert (tmp_path / "benchmantic" / "outputs" / "dumux" / "parameters_croissant.json").is_file()
    assert not (root / "parameters_croissant.json").exists()
