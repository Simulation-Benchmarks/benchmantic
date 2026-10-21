"""Plate with a hole (test program): the remaining area of a square plate."""

import json
import math
from argparse import ArgumentParser


def run(parameter_file: str, metrics_file: str) -> None:
    with open(parameter_file) as f:
        parameters = json.load(f)
    radius = parameters["radius"]["value"]
    length = parameters["length"]["value"]
    metrics = {
        "plate_area": length**2 - math.pi * radius**2,
    }
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_parameter_file", required=True)
    parser.add_argument("--output_metrics_file", required=True)
    args = parser.parse_args()
    run(args.input_parameter_file, args.output_metrics_file)
