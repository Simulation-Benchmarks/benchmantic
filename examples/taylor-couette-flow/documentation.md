# Taylor-Couette Flow

Steady viscous flow between two concentric rotating cylinders.

## Problem

An incompressible fluid with density $\rho$ and dynamic viscosity $\mu$
fills the gap between two long, concentric cylinders. The inner cylinder of
radius $R_1$ rotates with angular velocity $\Omega_1$, the outer cylinder of
radius $R_2$ with $\Omega_2$. If the flow does not vary along the axis, the
steady Navier-Stokes equations

$$
\rho (\mathbf{u} \cdot \nabla) \mathbf{u} = -\nabla p + \mu \nabla^2 \mathbf{u}, \qquad \nabla \cdot \mathbf{u} = 0
$$

are solved in the plane perpendicular to the axis.

| Parameter | Symbol | Value | Unit |
|---|---|---|---|
| Inner cylinder radius | $R_1$ | 1 | m |
| Outer cylinder radius | $R_2$ | 2 | m |
| Inner cylinder angular velocity | $\Omega_1$ | 100 | rad/s |
| Outer cylinder angular velocity | $\Omega_2$ | 0 | rad/s |
| Fluid density | $\rho$ | 1 | kg/m³ |
| Fluid dynamic viscosity | $\mu$ | 1 | Pa s |

With these values the Reynolds number $Re = \Omega_1 R_1 (R_2 - R_1)\rho/\mu = 100$.

## Analytical solution

The flow is circular Couette flow, $u_r = 0$ and

$$
u_\theta(r) = A r + \frac{B}{r}, \qquad
A = \frac{\Omega_2 R_2^2 - \Omega_1 R_1^2}{R_2^2 - R_1^2}, \qquad
B = \frac{(\Omega_1 - \Omega_2) R_1^2 R_2^2}{R_2^2 - R_1^2}.
$$

The radial momentum balance $\partial p/\partial r = \rho\, u_\theta^2 / r$ gives

$$
p(r) = \rho \left( \frac{A^2 r^2}{2} + 2AB \ln r - \frac{B^2}{2 r^2} \right) + C,
$$

where the constant $C$ is arbitrary because the velocity is prescribed on the
whole boundary. For comparison, the pressure difference across the gap is
$p(R_2) - p(R_1) \approx 2172$ Pa and the inner wall speed is
$\Omega_1 R_1 = 100$ m/s.

In three dimensions this flow becomes unstable to axisymmetric Taylor vortices
above a critical Reynolds number (Taylor 1923); for $R_1/R_2 = 0.5$ and
$\Omega_2 = 0$, $Re_c \approx 68.2$. The two-dimensional model has no axial
direction, so at $Re = 100$ it computes the circular Couette flow above.

## Configurations

The annulus is meshed as a structured grid of quadrilaterals with 16 cells
across the gap and 128 around it. Global refinement halves the cell size in
both directions; the new vertices lie on the edges of the coarse grid, so the
cylinders stay polygons with 128 edges. The velocity on both cylinders is set
to the analytical solution evaluated on the grid's boundary, so the errors
measure the discretization and not the polygonal approximation of the
cylinders.

| Configuration | Refinement | Cells (radial × angular) |
|---|---:|---|
| `16x128` | 0 | 16 × 128 |
| `32x256` | 1 | 32 × 256 |
| `64x512` | 2 | 64 × 512 |

## Metrics

Discrete $L^2$ errors

$$
\| e \| = \left( \frac{\sum_i |V_i|\, |e_i|^2}{\sum_i |V_i|} \right)^{1/2},
$$

where $e_i$ is the error at degree of freedom $i$ with control volume $V_i$:
cell centres for the pressure, face centres for the velocity. Before the
pressure error is computed, the numerical pressure is shifted so that its
volume average equals that of the analytical solution. The velocity error
combines both components. The relative errors divide by the discrete $L^2$
norm of the analytical solution, computed the same way.

| Metric | Unit | Description |
|---|---|---|
| `l2_error_pressure_rel` | - | relative $L^2$ error of the pressure |
| `l2_error_velocity_rel` | - | relative $L^2$ error of the velocity |
| `l2_error_pressure` | Pa | $L^2$ error of the pressure |
| `l2_error_velocity` | m/s | $L^2$ error of the velocity |

## Reference results (DuMux)

DuMux (face-centred diamond scheme for momentum, cell-centred TPFA for mass,
central weighting of the advective flux) gives the values below, published on
the [DuMux benchmark page](https://dumux.org/docs/doxygen/master/benchmark-taylor-couette.html).
Both errors converge with second order.

| Grid | $L^2$ error pressure (Pa) | rate | $L^2$ error velocity (m/s) | rate |
|---|---:|---:|---:|---:|
| 16 × 128 | 1.519e+02 | - | 6.727e+00 | - |
| 32 × 256 | 3.368e+01 | 2.17 | 1.534e+00 | 2.13 |
| 64 × 512 | 8.143e+00 | 2.05 | 3.736e-01 | 2.04 |

## References

- G. I. Taylor (1923). Stability of a viscous liquid contained between two rotating cylinders. *Philosophical Transactions of the Royal Society A* 223, 289–343.
- DuMux: [Taylor-Couette flow benchmark](https://dumux.org/docs/doxygen/master/benchmark-taylor-couette.html), `test/freeflow/navierstokes/taylorcouette`.
