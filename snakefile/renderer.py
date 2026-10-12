# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
snakefile.renderer

Snakefile pieces shared by every software: the parameters.json key
convention (mirroring run_benchmark.py's unit-suffix keys), the per-case
preview parameters.json, and the benchmark-parameter assignments a
mapping-based Snakefile starts with. The Snakefile templates themselves
live with each software (software/<name>/snakefile.py).
"""

from __future__ import annotations

import json
from typing import Any


#: Mirrors run_benchmark.py's UNIT_SYMBOLS. MUST be kept in sync with that
#: file -- it determines the exact parameters.json keys run_benchmark.py
#: writes at runtime. If that file's UNIT_SYMBOLS differs (more units, or
#: a different symbol), pass --unit-symbol to override/extend this default
#: rather than letting the two silently drift apart.
DEFAULT_UNIT_SYMBOLS: dict[str, str] = {
    "unit:M": "m",
    "unit:PA": "Pa",
    # Matches the Simulation-Benchmarks run_benchmark.py UNIT_SYMBOLS
    # ("M": "m", "RAD-PER-SEC": "rad/s"), which writes the parameters.json
    # keys this Snakefile reads.
    "unit:RAD-PER-SEC": "rad/s",
}


def config_key(flag: str, unit: str | None, unit_symbols: dict[str, str]) -> str:
    """Mirror run_benchmark.py's parameter_json_key(): append a bracketed
    unit symbol to the key when the parameter's unit is a known one (e.g.
    "Grid.Radial0" -> "Grid.Radial0[m]"), otherwise use the bare
    Section.Key. This is the key used to look the value up in `config`
    (parameters.json) -- NOT the "-Section.Key" CLI flag name passed to the
    executable, which always stays the bare label either way.
    """
    symbol = unit_symbols.get(unit) if unit else None
    return f"{flag}[{symbol}]" if symbol else flag


def render_parameters_json(case_id: str, values: dict[str, Any], units: dict[str, str],
                            unit_symbols: dict[str, str]) -> str:
    """Render this case's preview parameters.json -- NOTE: when this
    Snakefile is actually run via run_benchmark.py, THAT script generates
    its own parameters.json per configuration (via
    create_parameter_files_from_benchmark()/create_parameter_file()) and
    this file is not read at all. It's kept as a human-readable preview /
    manual-testing aid, so its keys mirror run_benchmark.py's
    parameter_json_key() convention for consistency with what will
    actually be used at runtime.
    """
    payload = {"configuration": case_id}
    for flag, value in values.items():
        payload[config_key(flag, units.get(flag), unit_symbols)] = value
    return json.dumps(payload, indent=2) + "\n"


# ---------------------------------------------------------------------------
# Snakefiles generated from an input mapping (metadata.mapping): the
# description's software-neutral parameters are read from parameters.json,
# and each software input is set from its expression.
# ---------------------------------------------------------------------------

#: Names a benchmark parameter can't take as a Snakefile variable.
RESERVED_NAMES = {"config", "json", "Path", "shell", "workflow", "rules", "conf_name", "zip_name",
                  "container_image", "container_shared_dir", "executable", "build_dir", "shared_dir",
                  "snakefile_dir", "application", "metrics_script"}


#: Same helper as metadata.mapping.tokens(), for expressions like
#: {tokens(radial_coordinates)[0]} (one value of a multi-value parameter).
TOKENS_HELPER = '''def tokens(value):
    items = value if isinstance(value, (list, tuple)) else str(value).replace("(", " ").replace(")", " ").split()
    out = []
    for item in items:
        try:
            num = float(item)
            out.append(int(num) if num.is_integer() and "." not in str(item) else num)
        except (TypeError, ValueError):
            out.append(item)
    return out
'''


def parameter_assignments(names: list[str], units: dict[str, str], unit_symbols: dict[str, str]) -> str:
    clash = sorted(set(names) & (RESERVED_NAMES | {"tokens"}))
    if clash:
        raise SystemExit(f"Error: benchmark parameter name(s) {clash} clash with Snakefile variables; rename them.")
    lines = [f"{n} = config[{config_key(n, units.get(n), unit_symbols)!r}]" for n in names]
    return TOKENS_HELPER + "\n" + "\n".join(lines)
