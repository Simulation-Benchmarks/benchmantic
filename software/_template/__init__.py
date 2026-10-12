# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""Template for a new simulation code. Copy this folder to
software/<name>/ (e.g. software/fenics/) -- the folder name becomes the
--software value -- then fill in adapter.py, snakefile.py and prompt.md.
Folders starting with "_" are ignored by the registry, so this one is never
loaded itself."""

from software._template.adapter import TemplateAdapter as Adapter

__all__ = ["Adapter"]
