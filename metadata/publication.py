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


DOI_URL_PATTERN = re.compile(r"(?:https?://(?:dx\.)?doi\.org/|doi:\s*)(10\.\d{4,9}/\S+)", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\(\d{4}\)")


def extract_publication_citation(benchmark_description: str) -> str | None:
    """Pull the citation out of a doc-comment like:

        "\\brief Test problem for ...\n\nBenchmark case from\n
          Taylor, G. I. (1923) Stability of ...\n  223, 289-343. https://doi.org/..."

    The citation starts at the first line containing a year in parentheses
    (the signal that this is a real reference, not prose) and runs to the
    end. DOI URLs are removed from the text but the rest of their line is
    kept -- e.g. the page numbers in "223, 289-343. https://doi.org/..."
    (dropping whole lines that mentioned doi.org used to cut the citation
    off mid-sentence). The DOI itself is returned by extract_publication_doi().
    """
    if not benchmark_description:
        return None
    lines = [ln.strip() for ln in benchmark_description.splitlines() if ln.strip()]
    start = next((i for i, ln in enumerate(lines) if YEAR_PATTERN.search(ln)), None)
    if start is None:
        return None
    text = " ".join(DOI_URL_PATTERN.sub("", ln).strip() for ln in lines[start:])
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def extract_publication_doi(benchmark_description: str) -> str | None:
    """The first DOI in the doc-comment, as a https://doi.org/... URL."""
    if not benchmark_description:
        return None
    m = DOI_URL_PATTERN.search(benchmark_description)
    return f"https://doi.org/{m.group(1).rstrip('.,;)')}" if m else None
