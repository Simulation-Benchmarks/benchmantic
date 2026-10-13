#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
croissant_description.py

Finds a benchmark's parameter file, identifies its parameters and writes
them as a Croissant 1.1 description: each parameter a cr:Field with its
location in the file and the extract/transform steps that read its value
(see metadata/croissant.py). No LLM call: parameters are named from their
section and key. describe_benchmark.py writes the same file, with the
reviewed names, units and types, as <name>_croissant.json.

    python3 croissant_description.py <module_dir> [-o outputs/dumux/parameters_croissant.json]
                                     [--software dumux] [--anchor key|line] [--check]

--check reads every value back through its extract/transform steps and
prints them next to their location, as a quick test of the description.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import software  # noqa: E402
from metadata.croissant import build_croissant, extract_values, write_croissant  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("module_dir", type=Path, help="The benchmark implementation (or a folder above it).")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Where to write the description (default: outputs/<software>/parameters_croissant.json "
                         "in this benchmantic folder, next to describe_benchmark.py's outputs).")
    ap.add_argument("--software", default="auto", help="Simulation software (default: detected).")
    ap.add_argument("--anchor", choices=["key", "line"], default="key",
                    help="Find each value by section and key (default) or by line and column.")
    ap.add_argument("--check", action="store_true",
                    help="Read every value back through its extract/transform steps and print it.")
    args = ap.parse_args(argv)

    root = args.module_dir.resolve()
    adapter = software.get_adapter(root, args.software)
    module_dir = adapter.find_module_dir(root, args)
    candidates = adapter.discover_parameters(module_dir)
    if not candidates:
        sys.exit(f"Error: no parameters found in {adapter.input_label} under {module_dir}")

    software_dir = re.sub(r"[^0-9a-zA-Z]+", "-", adapter.name).strip("-").lower() or "benchmark"
    output = (args.output or HERE / "outputs" / software_dir / "parameters_croissant.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = build_croissant(adapter, module_dir, candidates, anchor=args.anchor, base_dir=output.parent,
                          manifest={"label": adapter.benchmark_label(module_dir, root, adapter.source_texts(module_dir))})
    write_croissant(doc, output)
    n_fields = sum(len(rs["field"]) for rs in doc["recordSet"])
    files = ", ".join(f["contentUrl"] for f in doc["distribution"])
    print(f"Wrote {output}: {n_fields} parameters from {files}")

    if args.check:
        values = extract_values(doc, output.parent)
        for rs in doc["recordSet"]:
            for field in rs["field"]:
                loc = field["bm:location"]
                where = f"{loc['bm:file']}:{loc['bm:line']}:{loc['bm:column']}"
                print(f"  {field['@id']:<45} {where:<28} {values[field['@id']]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
