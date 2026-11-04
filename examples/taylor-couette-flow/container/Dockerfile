# SPDX-FileCopyrightText: 2026 The taylor-couette-flow authors
# SPDX-License-Identifier: MIT
#
# DuMux (master, pinned) with the Taylor-Couette benchmark built in place of
# test/freeflow/navierstokes/taylorcouette. The benchmark workflow mounts its
# results folder at /dumux/shared and runs
#   /dumux/dumux/build-cmake/test/freeflow/navierstokes/taylorcouette/test_ff_navierstokes_taylorcouette
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake git libsuitesparse-dev pkg-config python3 zip \
 && rm -rf /var/lib/apt/lists/*

COPY build_dumux.sh /opt/benchmark/build_dumux.sh
COPY taylorcouette /opt/benchmark/taylorcouette
RUN /opt/benchmark/build_dumux.sh /dumux /opt/benchmark/taylorcouette \
 && rm -rf /dumux/*/.git \
 && mkdir -p /dumux/shared

WORKDIR /dumux/dumux/build-cmake/test/freeflow/navierstokes/taylorcouette
