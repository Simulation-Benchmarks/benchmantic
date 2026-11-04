// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
//
// SPDX-FileCopyrightText: Copyright © DuMux Project contributors, see AUTHORS.md in root folder
// SPDX-FileCopyrightText: 2026 The taylor-couette-flow authors
// SPDX-License-Identifier: GPL-3.0-or-later
//
/*!
 * \file
 * \ingroup NavierStokesTests
 * \brief Taylor-Couette flow between two concentric rotating cylinders
 *
 * Adapted from DuMux test/freeflow/navierstokes/taylorcouette: in addition to the
 * upstream test, this writes the discrete L2 errors of pressure and velocity
 * (absolute and relative to the analytical solution) to solution_metrics.json
 * next to the output named by Problem.Name, and packs the VTK output into
 * <Problem.Name>.zip.
 */
#include <config.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <dune/common/parallel/mpihelper.hh>

#include <dumux/common/initialize.hh>
#include <dumux/common/properties.hh>
#include <dumux/common/parameters.hh>

#include <dumux/io/grid/cakegridmanager.hh>
#include <dumux/io/vtkoutputmodule.hh>

#include <dumux/linear/istlsolvers.hh>
#include <dumux/linear/linearsolvertraits.hh>
#include <dumux/linear/linearalgebratraits.hh>

#include <dumux/multidomain/assembler.hh>
#include <dumux/multidomain/traits.hh>
#include <dumux/multidomain/newtonsolver.hh>

#include <dumux/freeflow/navierstokes/momentum/velocityoutput.hh>
#include <test/freeflow/navierstokes/analyticalsolutionvectors.hh>
#include <test/freeflow/navierstokes/errors.hh>

#include "properties.hh"

int main(int argc, char** argv)
{
    using namespace Dumux;

    using MomentumTypeTag = Properties::TTag::TaylorCouetteTestMomentum;
    using MassTypeTag = Properties::TTag::TaylorCouetteTestMass;

    initialize(argc, argv);
    const auto& mpiHelper = Dune::MPIHelper::instance();
    Parameters::init(argc, argv);

    CakeGridManager<GetPropType<MassTypeTag, Properties::Grid>> gridManager;
    gridManager.init();
    // the new vertices of a global refinement lie on the edges of the coarse grid,
    // so the cylinders remain the polygons of the coarse grid
    gridManager.grid().globalRefine(getParam<int>("Grid.Refinement", 0));
    const auto& gridView = gridManager.grid().leafGridView();

    using MomentumGridGeometry = GetPropType<MomentumTypeTag, Properties::GridGeometry>;
    auto momentumGridGeometry = std::make_shared<MomentumGridGeometry>(gridView);
    using MassGridGeometry = GetPropType<MassTypeTag, Properties::GridGeometry>;
    auto massGridGeometry = std::make_shared<MassGridGeometry>(gridView);

    using CouplingManager = GetPropType<MomentumTypeTag, Properties::CouplingManager>;
    auto couplingManager = std::make_shared<CouplingManager>();

    using MomentumProblem = GetPropType<MomentumTypeTag, Properties::Problem>;
    auto momentumProblem = std::make_shared<MomentumProblem>(momentumGridGeometry, couplingManager);
    using MassProblem = GetPropType<MassTypeTag, Properties::Problem>;
    auto massProblem = std::make_shared<MassProblem>(massGridGeometry, couplingManager);

    constexpr auto momentumIdx = CouplingManager::freeFlowMomentumIndex;
    constexpr auto massIdx = CouplingManager::freeFlowMassIndex;
    using Traits = MultiDomainTraits<MomentumTypeTag, MassTypeTag>;
    using SolutionVector = typename Traits::SolutionVector;
    SolutionVector x;
    x[momentumIdx].resize(momentumGridGeometry->numDofs());
    x[massIdx].resize(massGridGeometry->numDofs());
    x = 0.0;

    using MomentumGridVariables = GetPropType<MomentumTypeTag, Properties::GridVariables>;
    auto momentumGridVariables = std::make_shared<MomentumGridVariables>(momentumProblem, momentumGridGeometry);
    using MassGridVariables = GetPropType<MassTypeTag, Properties::GridVariables>;
    auto massGridVariables = std::make_shared<MassGridVariables>(massProblem, massGridGeometry);

    couplingManager->init(momentumProblem, massProblem, std::make_tuple(momentumGridVariables, massGridVariables), x);
    massGridVariables->init(x[massIdx]);
    momentumGridVariables->init(x[momentumIdx]);

    using Assembler = Experimental::MultiDomainAssembler<Traits, CouplingManager, DiffMethod::numeric>;
    auto assembler = std::make_shared<Assembler>(std::make_tuple(momentumProblem, massProblem),
                                                 std::make_tuple(momentumGridGeometry, massGridGeometry),
                                                 std::make_tuple(momentumGridVariables, massGridVariables),
                                                 couplingManager);

    using IOFields = GetPropType<MassTypeTag, Properties::IOFields>;
    VtkOutputModule vtkWriter(*massGridVariables, x[massIdx], massProblem->name());
    IOFields::initOutputModule(vtkWriter);
    vtkWriter.addVelocityOutput(std::make_shared<NavierStokesVelocityOutput<MassGridVariables>>());
    NavierStokesTest::AnalyticalSolutionVectors analyticalSolVectors(momentumProblem, massProblem);
    vtkWriter.addField(analyticalSolVectors.analyticalPressureSolution(), "pressureExact");
    vtkWriter.addField(analyticalSolVectors.analyticalVelocitySolution(), "velocityExact");

    using LinearSolver = UMFPackIstlSolver<SeqLinearSolverTraits, LinearAlgebraTraitsFromAssembler<Assembler>>;
    auto linearSolver = std::make_shared<LinearSolver>();

    using NewtonSolver = MultiDomainNewtonSolver<Assembler, LinearSolver, CouplingManager>;
    NewtonSolver nonLinearSolver(assembler, linearSolver, couplingManager);
    nonLinearSolver.solve(x);

    // the pressure is fixed in one cell; match its volume average to the
    // analytical solution so the error of that cell does not shift the whole pressure field
    const auto& pressureExact = analyticalSolVectors.analyticalPressureSolution();
    double pressureDifference = 0.0;
    double volume = 0.0;
    for (const auto& element : elements(gridView))
    {
        const auto eIdx = massGridGeometry->elementMapper().index(element);
        const auto elementVolume = element.geometry().volume();
        pressureDifference += (x[massIdx][eIdx][0] - pressureExact[eIdx])*elementVolume;
        volume += elementVolume;
    }
    for (auto& p : x[massIdx])
        p -= pressureDifference/volume;
    massGridVariables->update(x[massIdx]);

    vtkWriter.write(1.0);

    NavierStokesTest::Errors errors(momentumProblem, massProblem, x);
    NavierStokesTest::ErrorCSVWriter(momentumProblem, massProblem).printErrors(errors);

    if (mpiHelper.rank() == 0)
    {
        // benchmark metrics: discrete L2 errors of pressure and velocity. The velocity error
        // combines both components, e = (|e_x|^2 + |e_y|^2)^(1/2); its relative error divides
        // by the discrete L2 norm of the analytical velocity, recovered from the per-component
        // absolute and relative errors (rel_i = abs_i / ref_i).
        const auto& abs = errors.l2Absolute();
        const auto& rel = errors.l2Relative();
        const double pressureAbs = abs[0];
        const double pressureRel = rel[0];
        double errorSquared = 0.0, referenceSquared = 0.0;
        for (int i = 1; i < abs.size(); ++i)
        {
            errorSquared += abs[i]*abs[i];
            if (rel[i] > 0.0)
                referenceSquared += (abs[i]/rel[i])*(abs[i]/rel[i]);
        }
        const double velocityAbs = std::sqrt(errorSquared);
        const double velocityRel = referenceSquared > 0.0 ? std::sqrt(errorSquared/referenceSquared) : 0.0;

        const std::filesystem::path outputName = massProblem->name();
        const auto outputDir = outputName.has_parent_path() ? outputName.parent_path() : std::filesystem::path(".");
        std::filesystem::create_directories(outputDir);
        std::ofstream metrics(outputDir / "solution_metrics.json");
        metrics << Fmt::format(
            "{{\n"
            "  \"l2_error_pressure_rel\": {:.10e},\n"
            "  \"l2_error_velocity_rel\": {:.10e},\n"
            "  \"l2_error_pressure\": {:.10e},\n"
            "  \"l2_error_velocity\": {:.10e}\n"
            "}}\n",
            pressureRel, velocityRel, pressureAbs, velocityAbs);
        metrics.close();
        std::cout << "Wrote " << (outputDir / "solution_metrics.json").string() << std::endl;

        // pack the VTK output of this run into <Problem.Name>.zip
        const auto base = outputName.string();
        const auto zipCommand = "zip -q -j '" + base + ".zip' '" + base + "'-*.vtu";
        if (std::system(zipCommand.c_str()) != 0)
            DUNE_THROW(Dune::IOError, "Could not create " << base << ".zip");
    }

    if (mpiHelper.rank() == 0)
        Parameters::print();

    return 0;
}
