// SPDX-FileCopyrightText: Copyright © DuMux Project contributors, see AUTHORS.md in root folder
// SPDX-License-Identifier: GPL-3.0-or-later
#include <dumux/common/parameters.hh>
namespace Dumux {
/*!
 * \brief Rotating cylinders (Taylor-Couette) test problem.
 * Two concentric cylinders rotate with prescribed angular velocities;
 * the analytical velocity profile is compared to the numerical solution.
 */
template<class TypeTag>
class RotatingCylindersProblem : public NavierStokesProblem<TypeTag>
{
public:
    RotatingCylindersProblem()
    {
        omega1_ = getParam<Scalar>("Problem.Omega1");
        omega2_ = getParam<Scalar>("Problem.Omega2");
        density_ = getParam<Scalar>("Component.LiquidDensity");
    }
private:
    Scalar omega1_, omega2_, density_;
};
} // end namespace Dumux
