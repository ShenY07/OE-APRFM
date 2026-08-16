"""Interpolate level A onto level B and record reference refinement errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def relative_error(approximation, reference, weights=None):
    difference = approximation - reference
    if weights is None:
        return float(np.linalg.norm(difference) / np.linalg.norm(reference))
    return float(np.sqrt(np.sum(weights * difference**2) / np.sum(weights * reference**2)))


def compare_p2(a, b):
    points = np.stack(np.meshgrid(b["x"], b["velocity"], indexing="ij"), axis=-1)
    interpolated_f = RegularGridInterpolator(
        (a["x"], a["velocity"]), a["f"], bounds_error=False, fill_value=None
    )(points)
    interpolated_rho = np.interp(b["x"], a["x"], a["rho"])
    return relative_error(interpolated_f, b["f"], b["weights"][None, :]), relative_error(interpolated_rho, b["rho"])


def sorted_periodic(data):
    order = np.argsort(data["theta"])
    theta = data["theta"][order]
    field = data["f"][:, :, order]
    theta = np.concatenate((theta[-1:] - 2.0 * np.pi, theta, theta[:1] + 2.0 * np.pi))
    field = np.concatenate((field[:, :, -1:], field, field[:, :, :1]), axis=2)
    return theta, field


def compare_p5(a, b):
    theta_a, field_a = sorted_periodic(a)
    order_b = np.argsort(b["theta"])
    theta_b = b["theta"][order_b]
    field_b = b["f"][:, :, order_b]
    mesh = np.stack(np.meshgrid(b["x"], b["y"], theta_b, indexing="ij"), axis=-1)
    interpolated_f = RegularGridInterpolator(
        (a["x"], a["y"], theta_a), field_a, bounds_error=False, fill_value=None
    )(mesh)
    density_mesh = np.stack(np.meshgrid(b["x"], b["y"], indexing="ij"), axis=-1)
    interpolated_rho = RegularGridInterpolator(
        (a["x"], a["y"]), a["rho"], bounds_error=False, fill_value=None
    )(density_mesh)
    return relative_error(interpolated_f, field_b), relative_error(interpolated_rho, b["rho"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=("p2", "p5"), required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--directory", type=Path, default=Path("results/references"))
    args = parser.parse_args()
    prefix = f"{args.problem}_parity_ref_eps_{args.epsilon:.0e}_level_"
    with np.load(args.directory / f"{prefix}A.npz") as source:
        a = {key: source[key] for key in source.files}
    with np.load(args.directory / f"{prefix}B.npz") as source:
        b = {key: source[key] for key in source.files}
    error_f, error_rho = compare_p2(a, b) if args.problem == "p2" else compare_p5(a, b)
    record = {"problem": args.problem, "epsilon": args.epsilon, "relative_difference_f_BA": error_f, "relative_difference_rho_BA": error_rho}
    output = args.directory / f"{args.problem}_parity_ref_eps_{args.epsilon:.0e}_refinement.json"
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
