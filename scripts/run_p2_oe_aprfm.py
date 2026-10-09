"""Run one P2 OE-APRFM realization against the accepted level-B reference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
import numpy as np
from jax import random, vmap

import modules.function_space as function_space
from configuration.p2_heterogeneous_1d import get_config
from constraints.continuous_1d_odd_even import (
    OddEvenDecompositionPointwiseBoundaryConstraint1D,
    OddEvenDecompositionPointwiseInteriorConstraint1D,
)
from modules.collocation_sampling import Sample1D
from modules.solution_construction import OddEvenDecompositionConstructor1D
from solver.least_squares import solve

jax.config.update("jax_enable_x64", True)


def run(
    epsilon: float,
    seed: int,
    output_dir: Path,
    reference_dir: Path,
    *,
    partitions: tuple[int, int] = (2, 2),
    features: int = 64,
    scale: float = 1.0,
    rcond: float = 1.0e-12,
    block_weights: tuple[float, float, float, float] | None = None,
    collocation: tuple[int, int] | None = None,
    tag: str = "",
    reference_level: str = "B",
    evaluation_stride: int = 1,
):
    config = get_config(epsilon)
    model = config.model
    model.Jn = {"j": features, "r": features, "f": features}
    model.Mp = {
        "j": partitions[0] * partitions[1],
        "r": partitions[0] * partitions[1],
        "f": partitions[0] * partitions[1],
    }
    model.num_unknowns = {
        "j/r": 2 * features * partitions[0] * partitions[1],
        "f": features * partitions[0] * partitions[1],
    }
    model.scale = scale
    if collocation is not None:
        model.collocation_sizes = {
            "interior": tuple(collocation),
            "boundary": int(max(collocation)),
        }
    config.mesh.strides = {"x": 1.0 / partitions[0], "v": 1.0 / partitions[1]}
    domain, strides = dict(config.mesh.domain), dict(config.mesh.strides)
    key = random.key(seed)
    function_space.seedXV = seed
    sampler = Sample1D(
        domain, dict(model.collocation_sizes), model.sample_mode
    )
    common = dict(
        domain=domain,
        strides=strides,
        Jn=dict(model.Jn),
        scale=float(model.scale),
        init_rng=key,
        kn=epsilon,
        activation=model.activation,
    )
    started = perf_counter()
    interior = OddEvenDecompositionPointwiseInteriorConstraint1D(
        **common, num_quads=int(model.num_quads), coeff_fns=dict(model.coeff)
    )
    boundary = OddEvenDecompositionPointwiseBoundaryConstraint1D(**common)
    dummy = (jnp.zeros((1,)), jnp.ones((1,)))
    ip = interior.init(key, *dummy)
    bp = boundary.init(key, *dummy)
    interior_fn = jax.jit(vmap(lambda x, v: interior.apply(ip, x, v)))
    boundary_fn = jax.jit(vmap(lambda x, v: boundary.apply(bp, x, v)))
    feature_seconds = perf_counter() - started
    started = perf_counter()
    lx, lv = sampler.pts_left
    rx, rv = sampler.pts_right
    boundary_matrix = np.concatenate(
        (np.asarray(boundary_fn(lx, lv)), np.asarray(boundary_fn(rx, rv)))
    )
    boundary_rhs = np.concatenate(
        (
            np.asarray(vmap(model.bdy_cond.f_l)(lv)).reshape(-1),
            np.asarray(vmap(model.bdy_cond.f_r)(rv)).reshape(-1),
        )
    )
    ix, iv = sampler.pts_int
    interior_matrix = np.asarray(interior_fn(ix, iv)).reshape(
        -1, int(model.num_unknowns["j/r"])
    )
    q_plus = np.asarray(vmap(model.source)(ix, iv)).reshape(-1)
    q_minus = np.asarray(vmap(model.source)(ix, -iv)).reshape(-1)
    q_even, q_odd = 0.5 * (q_plus + q_minus), 0.5 * (q_plus - q_minus)
    interior_rhs = np.column_stack(
        (np.zeros_like(q_even), epsilon**2 * q_even, epsilon * q_odd)
    ).reshape(-1)
    matrix = np.vstack((boundary_matrix, interior_matrix))
    rhs = np.concatenate((boundary_rhs, interior_rhs))
    assembly_seconds = perf_counter() - started
    started = perf_counter()
    coefficients, diagnostics = solve(
        matrix,
        rhs,
        n_boundary=boundary_matrix.shape[0] if block_weights else None,
        block_weights=block_weights,
        rcond=rcond,
        return_diagnostics=True,
    )
    solve_seconds = perf_counter() - started
    constructor = OddEvenDecompositionConstructor1D(
        **common, coefficients=jnp.asarray(coefficients.reshape(-1))
    )
    cp = constructor.init(key, *dummy)
    approximation = jax.jit(vmap(lambda x, v: constructor.apply(cp, x, v)))
    ref_path = (
        reference_dir
        / f"p2_parity_ref_eps_{epsilon:.0e}_level_{reference_level}.npz"
    )
    with np.load(ref_path) as ref:
        x, velocity, reference_f, weights = (
            ref["x"],
            ref["velocity"],
            ref["f"],
            ref["weights"],
        )
        reference_rho = ref["rho"]
    if evaluation_stride < 1:
        raise ValueError("evaluation_stride must be positive")
    if evaluation_stride > 1:
        x = x[::evaluation_stride]
        velocity = velocity[::evaluation_stride]
        reference_f = reference_f[::evaluation_stride, ::evaluation_stride]
        weights = weights[::evaluation_stride]
        weights = weights * (2.0 / weights.sum())
        reference_rho = 0.5 * (reference_f @ weights)
    started = perf_counter()
    numerical = np.empty_like(reference_f)
    for begin in range(0, x.size, 32):
        stop = min(begin + 32, x.size)
        xx, vv = np.meshgrid(x[begin:stop], velocity, indexing="ij")
        numerical[begin:stop] = np.asarray(
            approximation(
                jnp.asarray(xx.reshape(-1, 1)), jnp.asarray(vv.reshape(-1, 1))
            )
        ).reshape(stop - begin, velocity.size)
    numerical_rho = 0.5 * (numerical @ weights)
    error_f = float(
        np.sqrt(
            np.sum(weights[None, :] * (numerical - reference_f) ** 2)
            / np.sum(weights[None, :] * reference_f**2)
        )
    )
    error_rho = float(
        np.linalg.norm(numerical_rho - reference_rho)
        / np.linalg.norm(reference_rho)
    )
    evaluation_seconds = perf_counter() - started
    record = {
        "problem": "p2",
        "method": "oe_aprfm",
        "epsilon": epsilon,
        "seed": seed,
        "relative_l2_f": error_f,
        "relative_l2_rho": error_rho,
        "residual_half": diagnostics["normalized_residual_rms"],
        "empirical_stability_ratio": np.hypot(error_f, error_rho)
        / diagnostics["normalized_residual_rms"],
        "condition_number": diagnostics["condition_number"],
        "rank": diagnostics["rank"],
        "largest_singular_value": diagnostics["largest_singular_value"],
        "smallest_effective_singular_value": diagnostics[
            "smallest_effective_singular_value"
        ],
        "singular_value_threshold": diagnostics["singular_value_threshold"],
        "num_rows": int(matrix.shape[0]),
        "num_columns": int(matrix.shape[1]),
        "oversampling_ratio": float(matrix.shape[0] / matrix.shape[1]),
        "partitions": partitions,
        "strides": dict(config.mesh.strides),
        "features_per_patch": features,
        "scale": scale,
        "rcond": rcond,
        "block_weights": block_weights,
        "collocation": dict(model.collocation_sizes),
        "reference_level": reference_level,
        "evaluation_stride": evaluation_stride,
        "evaluation_grid": [int(x.size), int(velocity.size)],
        "feature_seconds": feature_seconds,
        "assembly_seconds": assembly_seconds,
        "solve_seconds": solve_seconds,
        "evaluation_seconds": evaluation_seconds,
        "total_seconds": feature_seconds
        + assembly_seconds
        + solve_seconds
        + evaluation_seconds,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    stem = f"p2_oe_aprfm_eps_{epsilon:.0e}_seed_{seed}{suffix}"
    (output_dir / f"{stem}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    np.savez_compressed(
        output_dir / f"{stem}.npz",
        x=x,
        velocity=velocity,
        f=numerical,
        rho=numerical_rho,
        reference_f=reference_f,
        reference_rho=reference_rho,
        error_rho=np.abs(numerical_rho - reference_rho),
    )
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/raw"))
    parser.add_argument(
        "--reference-dir", type=Path, default=Path("results/references")
    )
    parser.add_argument("--partitions", type=int, nargs=2, default=(2, 2))
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--rcond", type=float, default=1.0e-12)
    parser.add_argument("--block-weights", type=float, nargs=4)
    parser.add_argument("--collocation", type=int, nargs=2)
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--reference-level", choices=("A", "B", "C", "D"), default="B"
    )
    parser.add_argument("--evaluation-stride", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.epsilon,
                args.seed,
                args.output_dir,
                args.reference_dir,
                partitions=tuple(args.partitions),
                features=args.features,
                scale=args.scale,
                rcond=args.rcond,
                block_weights=(
                    None
                    if args.block_weights is None
                    else tuple(args.block_weights)
                ),
                collocation=(
                    None
                    if args.collocation is None
                    else tuple(args.collocation)
                ),
                tag=args.tag,
                reference_level=args.reference_level,
                evaluation_stride=args.evaluation_stride,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
