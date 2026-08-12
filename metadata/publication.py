# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""
metadata.publication

Pulls a literature citation out of a benchmark's Doxygen-style description
comment, if one is present.
"""

from __future__ import annotations

import re


def extract_publication_citation(benchmark_description: str) -> str | None:
    """Pull the citation block out of a doc-comment like:
    "\\brief ...\n\nBenchmark case from\n  Turek, Schaefer et al (1996) ...\n  https://doi.org/..."
    Returns the citation text (without the doi URL line) if a
    year-in-parens pattern is found, signaling an actual reference rather
    than just descriptive prose.
    """
    if not benchmark_description:
        return None
    lines = [ln.strip() for ln in benchmark_description.splitlines() if ln.strip()]
    citation_lines = [ln for ln in lines if not ln.startswith("\\") and "doi.org" not in ln.lower()]
    citation = " ".join(citation_lines).strip()
    if citation and re.search(r"\(\d{4}\)", citation):
        return citation
    return None
