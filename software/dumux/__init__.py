# SPDX-FileCopyrightText: 2026 Simulation-Benchmarks
#
# SPDX-License-Identifier: MIT

"""DuMux / DUNE support -- see adapter.py. Registered automatically by
software/__init__.py through `Adapter`."""

from software.dumux.adapter import DumuxAdapter as Adapter

__all__ = ["Adapter"]
