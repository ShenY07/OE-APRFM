#!/usr/bin/env python3
"""Train a 1D or 2D odd-even asymptotic-preserving neural network."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from ap_eqn import build_solver
from config import get_config
from data_pipeline import CollocationDataset
from networks import build_networks, parameter_count


def _squared_losses(residuals):
    return {
        name: torch.mean(value.square()) for name, value in residuals.items()
    }


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=Path(__file__).resolve().parents[2],
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def evaluate(config, nets, solver, output_dir: Path, batch_size: int = 2048):
    """Evaluate on the same independent grids/references used by the comparison."""
    problem = config.problem.name
    epsilon = float(config.rte.kn)
    reference_dir = Path("results/references")
    if problem == "p1":
        reference_path = None
    elif problem == "p2":
        reference_path = (
            reference_dir / f"p2_parity_ref_eps_{epsilon:.0e}_level_B.npz"
        )
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
            mask = arrays.get(
                "domain_mask", np.ones(reference_rho.shape, dtype=bool)
            ).astype(bool)
    elif problem == "p1":
        x = np.linspace(0.0, 1.0, 257)
        velocity = np.linspace(-1.0, 1.0, 128)
        reference_f = 1.0 - x[:, None] + np.zeros((1, velocity.size))
        reference_rho = reference_f.mean(axis=1)
        mask = np.ones(reference_rho.shape, dtype=bool)
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
            else (
                (np.abs(xx[:, :, 0]) > 1 / 3) | (np.abs(yy[:, :, 0]) > 1 / 3)
                if problem == "p5"
                else np.ones(reference_rho.shape, dtype=bool)
            )
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
                torch.as_tensor(
                    value[begin : begin + batch_size],
                    dtype=torch.get_default_dtype(),
                    device=solver.device,
                ).reshape(-1, 1)
                for value in flat
            ]
            values.append(
                solver.reconstruct(nets, *arguments).cpu().numpy().reshape(-1)
            )
    numerical = np.concatenate(values).reshape(reference_f.shape)
    if int(config.problem.dimension) == 1:
        if reference_path is not None and "weights" in arrays:
            weights = arrays["weights"]
            numerical_rho = 0.5 * (numerical @ weights)
        else:
            numerical_rho = np.trapezoid(numerical, velocity, axis=1) / (
                velocity[-1] - velocity[0]
            )
        phase_mask = np.ones(reference_f.shape, dtype=bool)
        mask = np.ones(reference_rho.shape, dtype=bool)
        save_axes = dict(x=x, velocity=velocity)
    else:
        numerical_rho = numerical.mean(axis=2)
        phase_mask = np.broadcast_to(mask[:, :, None], reference_f.shape)
        save_axes = dict(x=x, y=y, theta=theta, mask=mask)
    error_f = float(
        np.linalg.norm((numerical - reference_f)[phase_mask])
        / np.linalg.norm(reference_f[phase_mask])
    )
    error_rho = float(
        np.linalg.norm((numerical_rho - reference_rho)[mask])
        / np.linalg.norm(reference_rho[mask])
    )
    np.savez_compressed(
        output_dir / "solution.npz",
        **save_axes,
        f=numerical,
        rho=numerical_rho,
        reference_f=reference_f,
        reference_rho=reference_rho,
        error_f=np.abs(numerical - reference_f),
        error_rho=np.abs(numerical_rho - reference_rho),
    )
    return {"relative_l2_f": error_f, "relative_l2_rho": error_rho}


def train(
    config,
    output_dir: Path,
    log_every: int = 100,
    evaluate_solution: bool = True,
):
    seed = int(config.seed)
    torch.set_default_dtype(torch.float64)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    solver = build_solver(config)
    nets = {
        name: net.to(solver.device)
        for name, net in build_networks(config).items()
    }
    count = parameter_count(nets)
    expected = 25474 if int(config.problem.dimension) == 1 else 25730
    if count != expected:
        raise RuntimeError(
            f"parameter count {count} does not match locked value {expected}"
        )
    parameters = [
        parameter for net in nets.values() for parameter in net.parameters()
    ]
    optimizer = torch.optim.Adam(parameters, lr=float(config.model.Adam.lr))
    data = iter(CollocationDataset(config, solver.device))
    # A fixed validation batch is independent of the resampled training stream.
    cpu_state, numpy_state, python_state = (
        torch.get_rng_state(),
        np.random.get_state(),
        random.getstate(),
    )
    cuda_state = (
        torch.cuda.get_rng_state(solver.device)
        if solver.device.type == "cuda"
        else None
    )
    torch.manual_seed(seed + 1000)
    np.random.seed(seed + 1000)
    random.seed(seed + 1000)
    if solver.device.type == "cuda":
        torch.cuda.manual_seed(seed + 1000)
    validation = next(iter(CollocationDataset(config, solver.device)))
    torch.set_rng_state(cpu_state)
    np.random.set_state(numpy_state)
    random.setstate(python_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, solver.device)
    history = {
        key: []
        for key in (
            "step",
            "learning_rate",
            "total_loss",
            "loss_macro",
            "loss_even",
            "loss_odd",
            "loss_boundary",
            "elapsed_seconds",
        )
    }
    best_loss, best_step, best_record = float("inf"), 0, None
    output_dir.mkdir(parents=True, exist_ok=True)
    if solver.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(solver.device)
    _synchronize(solver.device)
    started = time.perf_counter()
    steps = int(config.model.iteration_steps)
    for step in range(1, steps + 1):
        batch = next(data)
        equation = _squared_losses(
            solver.phase_residual(nets, batch["interior"])
        )
        boundary_loss = _squared_losses(
            solver.boundary_residual(nets, batch["boundary"])
        )["inflow"]
        loss = (
            sum(
                float(config.model.regularizers[name]) * equation[name]
                for name in ("macro", "even", "odd")
            )
            + float(config.model.regularizers.boundary) * boundary_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (
            step == 1
            or step % int(config.model.validation_interval) == 0
            or step == steps
        ):
            validation_equation = _squared_losses(
                solver.phase_residual(nets, validation["interior"])
            )
            validation_boundary = _squared_losses(
                solver.boundary_residual(nets, validation["boundary"])
            )["inflow"]
            validation_loss = (
                sum(validation_equation.values()) + validation_boundary
            )
            _synchronize(solver.device)
            record = {
                "step": step,
                "total_loss": float(validation_loss.detach()),
                "loss_macro": float(validation_equation["macro"].detach()),
                "loss_even": float(validation_equation["even"].detach()),
                "loss_odd": float(validation_equation["odd"].detach()),
                "loss_boundary": float(validation_boundary.detach()),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - started,
            }
            for key in history:
                history[key].append(record[key])
            if record["total_loss"] < best_loss:
                best_loss, best_step, best_record = (
                    record["total_loss"],
                    step,
                    record.copy(),
                )
                for name, net in nets.items():
                    torch.save(net.state_dict(), output_dir / f"{name}.pt")
            if step == 1 or step % log_every == 0 or step == steps:
                print(json.dumps(record), flush=True)
    _synchronize(solver.device)
    train_seconds = time.perf_counter() - started
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(solver.device) / 2**20
        if solver.device.type == "cuda"
        else 0.0
    )
    for name, net in nets.items():
        net.load_state_dict(
            torch.load(
                output_dir / f"{name}.pt",
                map_location=solver.device,
                weights_only=True,
            )
        )
    np.savez(
        output_dir / "history.npz",
        **{key: np.asarray(value) for key, value in history.items()},
    )
    evaluation_started = time.perf_counter()
    errors = (
        evaluate(config, nets, solver, output_dir) if evaluate_solution else {}
    )
    relative_f, relative_rho = errors.get("relative_l2_f"), errors.get(
        "relative_l2_rho"
    )
    summary = {
        "protocol": str(config.protocol.name),
        "parity_projection": str(config.protocol.parity_projection),
        "method": "OE-APNN",
        "problem": str(config.problem.name).upper(),
        "dimension": int(config.problem.dimension),
        "epsilon": float(config.rte.kn),
        "seed": seed,
        "steps": steps,
        "interior_samples": int(config.model.dataset.interior_samples),
        "boundary_samples": int(config.model.dataset.boundary_samples),
        "quadrature_points": int(config.rte.num_vquads),
        "status": "success",
        "evaluation_status": "complete" if evaluate_solution else "deferred",
        "best_step": best_step,
        "train_time_s": train_seconds,
        "parameter_count": count,
        "peak_gpu_memory_mb": peak_memory_mb,
        "n_int_per_step": int(config.model.dataset.interior_samples),
        "n_bdy_per_step": int(config.model.dataset.boundary_samples),
        "n_int_total": int(config.model.dataset.interior_samples) * steps,
        "n_bdy_total": int(config.model.dataset.boundary_samples) * steps,
        "quadrature_base_nodes": int(config.rte.num_vquads),
        "dtype": "float64",
        "device": str(solver.device),
        "git_commit": _git_commit(),
        "E_f": relative_f,
        "E_rho": relative_rho,
        "elapsed_seconds": train_seconds,
        "evaluation_seconds": (
            time.perf_counter() - evaluation_started
            if evaluate_solution
            else 0.0
        ),
        **errors,
        "loss_best": {
            "total": best_loss,
            "macro": best_record["loss_macro"],
            "even": best_record["loss_even"],
            "odd": best_record["loss_odd"],
            "boundary": best_record["loss_boundary"],
        },
    }
    config_summary = {
        "protocol": str(config.protocol.name),
        "problem": str(config.problem.name).upper(),
        "epsilon": float(config.rte.kn),
        "seed": seed,
        "dtype": "float64",
        "network": {
            "parity_projection": str(config.protocol.parity_projection),
            "hidden_width": 64,
            "hidden_transforms": 4,
            "separate_r_j": True,
        },
        "optimizer": {
            "name": "Adam",
            "learning_rate": float(config.model.Adam.lr),
            "scheduler": None,
        },
        "training": {
            "steps": steps,
            "interior_phase_samples": int(
                config.model.dataset.interior_samples
            ),
            "boundary_phase_samples": int(
                config.model.dataset.boundary_samples
            ),
            "validation_interval": int(config.model.validation_interval),
        },
        "parameter_count": count,
        "device": str(solver.device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "cpu_threads": torch.get_num_threads(),
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("ap_eqn.py", "config.py", "networks.py", "data_pipeline.py", "train.py")
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return nets, solver, summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem", choices=("p1", "p2", "p3", "p4", "p5"), default="p3"
    )
    parser.add_argument("--epsilon", type=float, default=1.0e-3)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--interior-samples", type=int, default=None)
    parser.add_argument("--boundary-samples", type=int, default=None)
    parser.add_argument("--quadrature-points", type=int, default=None)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.require_cuda:
        if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
            raise RuntimeError(f"Requested cuda:{args.device} is unavailable; refusing CPU fallback")
        torch.cuda.set_device(args.device)
    config = get_config(args.problem, args.epsilon)
    config.seed = args.seed
    config.model.device_ids = [args.device]
    if args.steps is not None:
        config.model.iteration_steps = args.steps
    if args.interior_samples is not None:
        config.model.dataset.interior_samples = args.interior_samples
    if args.boundary_samples is not None:
        config.model.dataset.boundary_samples = args.boundary_samples
    if args.quadrature_points is not None:
        config.rte.num_vquads = args.quadrature_points
    epsilon_tag = "1e0" if args.epsilon == 1.0 else "1e-3"
    output_dir = (
        args.output_dir
        or Path("results/baselines/oe_apnn")
        / args.problem.upper()
        / f"eps_{epsilon_tag}"
        / f"seed_{args.seed}"
    )
    train(config, output_dir, args.log_every, not args.skip_evaluation)


if __name__ == "__main__":
    main()
