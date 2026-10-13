# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""benchmantic's output against the SHACL shapes of the installed
semantic-benchmark package (BenchmarkLoader.load_shapes()). When a new
semantic-benchmark version changes the shapes, this test shows whether
benchmantic still writes valid descriptions. Run with:
python3 -m pytest tests  (after pip install -U semantic-benchmark)"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("semantic_benchmark")

FIXTURES = Path(__file__).parent / "fixtures"


def _answers(candidates):
    """Fixed LLM answers: a plausible name, datatype and unit per input."""
    out = []
    for c in candidates:
        name = re.sub(r"[^a-z0-9]+", "_", c.key.split("/")[-1].lower()).strip("_")
        numeric = bool(c.tokens) and all(re.match(r"^[-+]?[\d.eE+-]+$", t) for t in c.tokens)
        out.append({"semantic_name": name, "ini": [c.section, c.key],
                    "datatype": "schema:Float" if numeric else "schema:String",
                    "unit": "unit:UNITLESS", "quantityKind": None, "index": 0,
                    "confidence": 0.99, "explanation": f"test answer for {c.key}"})
    return out


def _metric_answers(candidates):
    return [{"key": c.key, "semantic_name": c.key, "datatype": "schema:Float", "unit": "unit:UNITLESS",
             "quantityKind": None, "confidence": 0.99, "explanation": f"test answer for {c.key}"}
            for c in candidates]


@pytest.mark.parametrize("software", ["DuMux", "OpenFOAM"])
def test_description_conforms_to_the_installed_shapes(software, tmp_path, monkeypatch):
    import describe_benchmark
    import verify_description
    from metadata import builder

    monkeypatch.setattr(builder, "infer_parameter_metadata", lambda candidates, *a, **k: _answers(candidates))
    monkeypatch.setattr(builder, "infer_metric_metadata", lambda candidates, *a, **k: _metric_answers(candidates))
    monkeypatch.setattr(builder.mardi, "lookup", lambda *a, **k: None, raising=False)
    monkeypatch.setenv("BENCHMANTIC_CORRECTIONS_PATH", str(tmp_path / "corrections.json"))
    monkeypatch.delenv("SEMANTIC_BENCHMARK_SRC", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda *a: "")   # Enter: accept the suggested selection

    implementation = tmp_path / software.lower()
    shutil.copytree(FIXTURES / software.lower(), implementation)
    describe_benchmark.main([str(implementation), "--software", software, "--container-image", "test:1",
                             "--outputs", "description-only", "--no-reference", "--no-mardi-lookup",
                             "--skip-review", "--skip-validation", "--fallback-on-error"])

    [description] = (tmp_path / "outputs" / software.lower()).glob("*benchmark.jsonld")
    args = verify_description.build_arg_parser().parse_args([str(description)])
    assert verify_description.run(args) == 0, "the description does not conform to the installed shapes"
