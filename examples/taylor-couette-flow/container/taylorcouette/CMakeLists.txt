# SPDX-FileCopyrightText: Copyright © DuMux Project contributors, see AUTHORS.md in root folder
# SPDX-FileCopyrightText: 2026 The taylor-couette-flow authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Taylor-Couette benchmark, built in place of DuMux's
# test/freeflow/navierstokes/taylorcouette (see ../Dockerfile and ../build_dumux.sh).
# The coarsest configuration is the default input file of the executable;
# the benchmark workflow passes every input on the command line.
configure_file(16x128/params.input params.input COPYONLY)

dumux_add_test(NAME test_ff_navierstokes_taylorcouette
               SOURCES main.cc
               LABELS freeflow navierstokes
               CMAKE_GUARD "( HAVE_UMFPACK AND dune-uggrid_FOUND )"
               COMMAND ./test_ff_navierstokes_taylorcouette
               CMD_ARGS params.input)
