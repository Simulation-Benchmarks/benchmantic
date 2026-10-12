Parameters are entries of OpenFOAM case dictionaries: "ini": [section, key]
is the dictionary file relative to the case (e.g. "constant/MRFProperties")
and the entry path inside it (e.g. "MRF1/omega").

"cpp_hint" names the dictionary and entry the value lives in; "code_context"
shows the entry and the lines that use it. In OpenFOAM, `nu` is a kinematic
viscosity (m2/s) and an MRF `omega` an angular velocity (rad/s).
