#!/usr/bin/env python3
"""Train a 1D or 2D odd-even asymptotic-preserving neural network."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from ap_eqn import build_solver
from config import get_config
from data_pipeline import CollocationDataset
from networks import build_networks


def _mean_square(residuals):
    return torch.stack([torch.mean(value.square()) for value in residuals.values()]).sum()


def evaluate(config, nets, solver, output_dir: Path, batch_size: int = 2048):
    """Evaluate on the same independent grids/references used by the comparison."""
    problem = config.problem.name
    epsilon = float(config.rte.kn)
    reference_dir = Path("results/references")
    if problem == "p1":
        reference_path = reference_dir / "p1_exact.npz"
    elif problem == "p2":
        reference_path = reference_dir / f"p2_parity_ref_eps_{epsilon:.0e}_level_B.npz"
    elif problem in ("p3", "p4"):
        reference_path = None
    else:
        reference_path = None

    if reference_path is not None:
        with np.load(reference_path) as data:
            arrays = {name: data[name] for name in data.files}
        x = arrays["x"]
        reference_f, reference_rho = arrays["f"], arrays["rho"]
        if int(config.problem.dimension) == 1:
            velocity = arrays["velocity"]
        else:
            y, theta = arrays["y"], arrays["theta"]
            mask = arrays.get("domain_mask", np.ones(reference_rho.shape, dtype=bool)).astype(bool)
    else:
        x = np.linspace(-1.0, 1.0, 65)
        y = np.linspace(-1.0, 1.0, 65)
        theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        xx, yy, _ = np.meshgrid(x, y, theta, indexing="ij")
        if problem == "p3":
            reference_rho = np.exp(-xx[:, :, 0] - yy[:, :, 0])
        elif problem == "p4":
            reference_rho = 1.0 / (1.0 + xx[:, :, 0] ** 2 + yy[:, :, 0] ** 2)
        else:
            reference_rho = np.exp(-xx[:, :, 0] - yy[:, :, 0])
        reference_f = np.broadcast_to(reference_rho[:, :, None], xx.shape)
        mask = (
            xx[:, :, 0] ** 2 + yy[:, :, 0] ** 2 >= 0.25
            if problem == "p4"
            else ((np.abs(xx[:, :, 0]) > 1/3) | (np.abs(yy[:, :, 0]) > 1/3)
                  if problem == "p5" else np.ones(reference_rho.shape, dtype=bool))
        )

    for net in nets.values():
        net.eval()
    if int(config.problem.dimension) == 1:
        xx, vv = np.meshgrid(x, velocity, indexing="ij")
        flat = (xx.reshape(-1), vv.reshape(-1))
    else:
        xx, yy, tt = np.meshgrid(x, y, theta, indexing="ij")
        flat = (xx.reshape(-1), yy.reshape(-1), tt.reshape(-1))
    values = []
    with torch.no_grad():
        for begin in range(0, flat[0].size, batch_size):
            arguments = [
                torch.as_tensor(value[begin : begin + batch_size], dtype=torch.get_default_dtype(), device=solver.device).reshape(-1, 1)
                for value in flat
            ]
            values.append(solver.reconstruct(nets, *arguments).cpu().numpy().reshape(-1))
    numerical = np.concatenate(values).reshape(reference_f.shape)
    if int(config.problem.dimension) == 1:
        if "weights" in arrays:
            weights = arrays["weights"]
            numerical_rho = 0.5 * (numerical @ weights)
        else:
            numerical_rho = np.trapz(numerical, velocity, axis=1) / (velocity[-1] - velocity[0])
        phase_mask = np.ones(reference_f.shape, dtype=bool)
        mask = np.ones(reference_rho.shape, dtype=bool)
        save_axes = dict(x=x, velocity=velocity)
    else:
        numerical_rho = numerical.mean(axis=2)
        phase_mask = np.broadcast_to(mask[:, :, None], reference_f.shape)
        save_axes = dict(x=x, y=y, theta=theta, mask=mask)
    error_f = float(np.linalg.norm((numerical - reference_f)[phase_mask]) / np.linalg.norm(reference_f[phase_mask]))
    error_rho = float(np.linalg.norm((numerical_rho - reference_rho)[mask]) / np.linalg.norm(reference_rho[mask]))
    np.savez_compressed(
        output_dir / "solution.npz", **save_axes, f=numerical, rho=numerical_rho,
        reference_f=reference_f, reference_rho=reference_rho,
        error_f=np.abs(numerical-reference_f), error_rho=np.abs(numerical_rho-reference_rho),
    )
    return {"relative_l2_f": error_f, "relative_l2_rho": error_rho}


def train(config, output_dir: Path, log_every: int = 100, evaluate_solution: bool = True):
    seed = int(config.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    solver = build_solver(config)
    nets = {name: net.to(solver.device) for name, net in build_networks(config).items()}
    parameters = [parameter for net in nets.values() for parameter in net.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=float(config.model.Adam.lr))
    data = iter(CollocationDataset(config, solver.device))
    history = []
    started = time.perf_counter()
    steps = int(config.model.iteration_steps)
    for step in range(1, steps + 1):
        batch = next(data)
        equation_loss = _mean_square(solver.residual(nets, batch["interior"]))
        boundary_loss = _mean_square(solver.boundary_residual(nets, batch["boundary"]))
        loss = (
            float(config.model.regularizers.equation) * equation_loss
            + float(config.model.regularizers.boundary) * boundary_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == steps:
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "equation_loss": float(equation_loss.detach()),
                "boundary_loss": float(boundary_loss.detach()),
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(record)
            print(json.dumps(record))

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, net in nets.items():
        torch.save(net.state_dict(), output_dir / f"{name}.pt")
    evaluation_started = time.perf_counter()
    errors = evaluate(config, nets, solver, output_dir) if evaluate_solution else {}
    summary = {
        "method": "OE-APNN",
        "dimension": int(config.problem.dimension),
        "epsilon": float(config.rte.kn),
        "seed": seed,
        "steps": steps,
        "interior_samples": int(config.model.dataset.interior_samples),
        "boundary_samples": int(config.model.dataset.boundary_samples),
        "quadrature_points": int(config.rte.num_vquads),
        "elapsed_seconds": time.perf_counter() - started,
        "evaluation_seconds": time.perf_counter() - evaluation_started if evaluate_solution else 0.0,
        **errors,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return nets, solver, summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("p1", "p2", "p3", "p4", "p5"), default="p3")
    parser.add_argument("--epsilon", type=float, default=1.0e-3)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--interior-samples", type=int, default=None)
    parser.add_argument("--boundary-samples", type=int, default=None)
    parser.add_argument("--quadrature-points", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config(args.problem, args.epsilon)
    config.seed = args.seed
    if args.steps is not None:
        config.model.iteration_steps = args.steps
    if args.interior_samples is not None:
        config.model.dataset.interior_samples = args.interior_samples
    if args.boundary_samples is not None:
        config.model.dataset.boundary_samples = args.boundary_samples
    if args.quadrature_points is not None:
        config.rte.num_vquads = args.quadrature_points
    output_dir = args.output_dir or Path("results") / f"{args.problem}_eps{args.epsilon:g}"
    train(config, output_dir, args.log_every, not args.skip_evaluation)


if __name__ == "__main__":
    main()
