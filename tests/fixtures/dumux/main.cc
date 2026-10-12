// SPDX-FileCopyrightText: Copyright © DuMux Project contributors, see AUTHORS.md in root folder
// SPDX-License-Identifier: GPL-3.0-or-later
#include <dumux/common/initialize.hh>
#include "problem.hh"
int main(int argc, char** argv)
{
    const auto radii = getParam<std::vector<double>>("Grid.Radial0");
    const auto cells = getParam<std::vector<int>>("Grid.Cells0");
    std::ofstream out(getParam<std::string>("Problem.Name") + "_metrics.json");
    out << "{\n";
    out << "  \"l2_error_velocity\": " << l2v << ",\n";
    out << "  \"l2_error_pressure\": " << l2p << "\n";
    out << "}\n";
    return 0;
}
