# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""OpenFOAM support -- see adapter.py. Registered automatically by
software/__init__.py through `Adapter`."""

from software.openfoam.adapter import OpenFOAMAdapter as Adapter

__all__ = ["Adapter"]
