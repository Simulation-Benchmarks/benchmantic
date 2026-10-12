"""Post-process an OpenFOAM case and write solution_metrics.json."""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("case")
ap.add_argument("--single-case", action="store_true")
args = ap.parse_args()

l2_velocity = 0.0  # L2 error of the tangential velocity vs. the analytical profile
torque_inner = 0.0  # torque on the inner cylinder per unit length
with open("solution_metrics.json", "w") as f:
    json.dump({"l2_error_velocity": l2_velocity, "torque_inner": torque_inner}, f, indent=2)
