Parameters are entries of a DUNE/DuMux params.input INI file: "ini":
[section, key] is the INI [Section] and the key (e.g. ["Grid", "Radial0"]).

"cpp_hint" is the getParam<Type>("Section.Key") call site in the C++ source
(e.g. the exact C++ variable name and type it is assigned to, such as
`density_` of type `Scalar`); "code_context" is the C++ source around that
call site. Variable names imply units: `density_`/`rho_` implies kg/m3, a
variable read as `omega_`/an angular velocity implies rad/s, `viscosity_`
(dynamic) implies Pa*s, `radius_`/`length_` implies m.
