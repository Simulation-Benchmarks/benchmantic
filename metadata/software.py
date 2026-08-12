# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
metadata.software

Detects which simulation software a benchmark module uses (DuMux,
OpenFOAM, deal.II, FEniCS, DUNE, ...) from telltale #include/namespace
signatures in its source.
"""

from __future__ import annotations

import re

SOFTWARE_SIGNATURES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"#include\s*<dumux/", re.IGNORECASE), "DuMux"),
    (re.compile(r"\bnamespace\s+Dumux\b"), "DuMux"),
    (re.compile(r"OpenFOAM", re.IGNORECASE), "OpenFOAM"),
    (re.compile(r"#include\s*<deal\.II/", re.IGNORECASE), "deal.II"),
    (re.compile(r"\bFEniCS\b", re.IGNORECASE), "FEniCS"),
    (re.compile(r"#include\s*<dune/", re.IGNORECASE), "DUNE"),
]


def detect_software_label(*source_texts: str) -> str | None:
    for text in source_texts:
        if not text:
            continue
        for pattern, label in SOFTWARE_SIGNATURES:
            if pattern.search(text):
                return label
    return None
