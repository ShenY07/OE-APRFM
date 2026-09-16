#!/usr/bin/env python3
"""Run the notebook-derived OE-SI-DSA/Krylov solver on P1--P5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from configuration.p1_manufactured_1d import get_config as p1_config
from configuration.p2_heterogeneous_1d import get_config as p2_config
from configuration.p3_manufactured_2d import get_config as p3_config
from configuration.p4_circular_hole_2d import get_config as p4_config
from configuration.p5_heterogeneous_2d import get_config as p5_config
from numerical.parity_reference import solve_parity_gmres_1d, solve_parity_gmres_2d, solve_parity_si_dsa_1d


CONFIGS = {"p1": p1_config, "p2": p2_config, "p3": p3_config, "p4": p4_config, "p5": p5_config}


def _relative_error(numerical, reference, mask=None):
    if mask is not None:
        numerical, reference = numerical[mask], reference[mask]
    denominator = np.linalg.norm(reference)
    return float(np.linalg.norm(numerical - reference) / denominator) if denominator else float("nan")


def _reference(problem, epsilon, result, config):
    if problem == "p2":
        path = ROOT / "results/references" / f"p2_parity_ref_eps_{epsilon:.0e}_level_B.npz"
        with np.load(path) as data:
            rho = np.interp(result["x"], data["x"], data["rho"])
        return None, rho, np.ones(rho.shape, dtype=bool)
    if problem in ("p1",):
        xx, vv = np.meshgrid(result["x"], result["velocity"], indexing="ij")
        exact = np.asarray(config.model.exact_solution(xx, vv), float)
        return np.broadcast_to(exact, result["f"].shape), exact[:, 0], np.ones(exact.shape[:1], dtype=bool)
    xx, yy = np.meshgrid(result["x"], result["y"], indexing="ij")
    exact_rho = np.asarray(config.model.exact_solution(xx, yy, 0.0), float)
    exact_f = np.broadcast_to(exact_rho[:, :, None], result["f"].shape)
    mask = xx**2 + yy**2 >= 0.25 if problem == "p4" else np.ones(xx.shape, dtype=bool)
    return exact_f, exact_rho, mask


def run(problem, epsilon, output_dir, grid_1d=(256, 128), grid_2d=(24, 24, 6), tol=1e-9, max_iter=500, method_label=None, dsa_relaxation=1e-2, solver_1d="auto"):
    config = CONFIGS[problem](epsilon)
    if int(config.model.eqn_type[-2]) == 1:
        if solver_1d == "dsa" or (solver_1d == "auto" and epsilon >= 1e-2):
            result = solve_parity_si_dsa_1d(config, grid=grid_1d, tol=tol, max_iter=max_iter, dsa_relaxation=dsa_relaxation)
            method = "oe_si_dsa"
        else:
            result = solve_parity_gmres_1d(config, grid=grid_1d, tol=tol, max_iter=max_iter)
            method = "oe_sn_krylov"
    else:
        result = solve_parity_gmres_2d(config, grid=grid_2d, tol=tol, max_iter=max_iter)
        method = "oe_si_dsa_krylov"
    if method_label is not None:
        method = method_label
    reference_f, reference_rho, mask = _reference(problem, epsilon, result, config)
    phase_mask = np.broadcast_to(mask[..., None], result["f"].shape)
    error_f = None if reference_f is None else _relative_error(result["f"], reference_f, phase_mask)
    error_rho = _relative_error(result["rho"], reference_rho, mask)
    record = {
        "problem": problem, "method": method, "epsilon": epsilon,
        "grid": list(grid_1d if problem in ("p1", "p2") else grid_2d),
        "angular_quadrature": "Gauss-Legendre",
        "tolerance": tol, "max_iterations": max_iter,
        "dsa_relaxation": dsa_relaxation if problem in ("p1", "p2") else None,
        "stopping_criterion": (
            "||u^(k)-u^(k-1)||_2 < tolerance" if method == "oe_si_dsa"
            else "preconditioned relative GMRES residual < tolerance"
        ),
        "iterations": int(result["iterations"]), "converged": bool(result["converged"]),
        "relative_l2_f": error_f, "relative_l2_rho": error_rho,
        "runtime_seconds": float(result["runtime_seconds"]),
    }
    if "final_difference" in result:
        record["final_difference"] = float(result["final_difference"])
    if "relative_residual" in result:
        record["relative_residual"] = float(result["relative_residual"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{problem}_{method}_eps_{epsilon:.0e}"
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    arrays.update(reference_rho=reference_rho, mask=mask)
    if reference_f is not None:
        arrays["reference_f"] = reference_f
    np.savez_compressed(output_dir / f"{stem}.npz", **arrays)
    (output_dir / f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--grid-1d", type=int, nargs=2, default=(256, 128))
    parser.add_argument("--grid-2d", type=int, nargs=3, default=(24, 24, 6))
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/baselines/oe_si_dsa")
    parser.add_argument("--method-label", default=None)
    parser.add_argument("--dsa-relaxation", type=float, default=1e-2)
    parser.add_argument("--solver-1d", choices=("auto","dsa","krylov"), default="auto")
    args = parser.parse_args()
    run(args.problem, args.epsilon, args.output_dir, tuple(args.grid_1d), tuple(args.grid_2d), args.tol, args.max_iter, args.method_label, args.dsa_relaxation, args.solver_1d)
