#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The taylor-couette-flow authors
# SPDX-License-Identifier: MIT
#
# Build the Taylor-Couette benchmark: DUNE 2.10 core modules, dune-uggrid and
# DuMux from source (pinned commits, GitHub mirrors), with the benchmark sources
# in place of DuMux's test/freeflow/navierstokes/taylorcouette. Needs SuiteSparse
# (UMFPACK) installed.
#
# Usage: build_dumux.sh <install root> <benchmark source folder>
# The executable ends up in
#   <install root>/dumux/build-cmake/test/freeflow/navierstokes/taylorcouette/
set -euo pipefail

ROOT="$(realpath -m "${1:?install root}")"
SOURCES="$(realpath "${2:?benchmark source folder}")"
JOBS="${JOBS:-$(nproc)}"

# DUNE releases/2.10 (core + dune-uggrid) and DuMux master (3.11-dev) as of 2026-10-19
declare -A COMMITS=(
  [dune-common]=2988ac1d85f59c949a73524be60631ffdb26d2e3
  [dune-geometry]=5673e95ac364ad3498aed9eaf65e0d224384d15a
  [dune-grid]=954436b88247e904628ec4d7c8bb7b2eaac08900
  [dune-localfunctions]=ddbf693b5f9c867b2d58d418fe130bbe92c06a99
  [dune-istl]=21c67275b17e93918365177f93f42e4aaa9afd23
  [dune-uggrid]=cf2513efb6497dc95744649e0658cedef2980bff
  [dumux]=83f2aa840eebbd63956294fe3abcacf86f1211f4
)
declare -A URLS=(
  [dune-common]=https://github.com/dune-project/dune-common
  [dune-geometry]=https://github.com/dune-project/dune-geometry
  [dune-grid]=https://github.com/dune-project/dune-grid
  [dune-localfunctions]=https://github.com/dune-project/dune-localfunctions
  [dune-istl]=https://github.com/dune-project/dune-istl
  [dune-uggrid]=https://github.com/dune-project/dune-uggrid
  [dumux]=https://github.com/dumux/dumux
)

mkdir -p "$ROOT"
cd "$ROOT"
for module in dune-common dune-geometry dune-uggrid dune-grid dune-localfunctions dune-istl dumux; do
  if [ ! -d "$module/.git" ]; then
    git init -q "$module"
    git -C "$module" remote add origin "${URLS[$module]}"
  fi
  git -C "$module" fetch -q --depth 1 origin "${COMMITS[$module]}"
  git -C "$module" checkout -q --force FETCH_HEAD
done

# the benchmark sources replace the upstream test
TEST_DIR="$ROOT/dumux/test/freeflow/navierstokes/taylorcouette"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"
cp -r "$SOURCES"/. "$TEST_DIR/"

cat > "$ROOT/release.opts" <<'EOF'
CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS='-O2 -g0' \
 -DDUNE_ENABLE_PYTHONBINDINGS=OFF -DDUNE_PYTHON_VIRTUALENV_SETUP=OFF \
 -DCMAKE_DISABLE_FIND_PACKAGE_MPI=TRUE -DCMAKE_DISABLE_FIND_PACKAGE_Doxygen=TRUE \
 -DCMAKE_DISABLE_FIND_PACKAGE_LATEX=TRUE"
MAKE_FLAGS="-j__JOBS__"
EOF
sed -i "s/__JOBS__/$JOBS/" "$ROOT/release.opts"

# configure DuMux and every DUNE module it depends on, build the DUNE
# libraries, then only the benchmark executable
./dune-common/bin/dunecontrol --opts=release.opts --module=dumux configure
for module in dune-common dune-geometry dune-uggrid dune-grid dune-localfunctions dune-istl; do
  make -C "$module/build-cmake" -j"$JOBS"
done
make -C dumux/build-cmake/test/freeflow/navierstokes/taylorcouette -j"$JOBS" test_ff_navierstokes_taylorcouette
echo "Built $TEST_DIR -> dumux/build-cmake/test/freeflow/navierstokes/taylorcouette/test_ff_navierstokes_taylorcouette"
