"""Run one reproducible P1 OE-APRFM realization and save machine-readable results."""

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
from configuration.p1_manufactured_1d import get_config
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
    *,
    partitions: tuple[int, int] = (1, 1),
    features: int = 64,
    scale: float = 1.0,
    rcond: float = 1.0e-12,
    block_weights: tuple[float, float, float, float] | None = None,
    collocation: tuple[int, int] | None = None,
    tag: str = "",
) -> dict[str, object]:
    config = get_config(epsilon)
    model = config.model
    patch_count = partitions[0] * partitions[1]
    model.Jn = {"j": features, "r": features, "f": features}
    model.Mp = {"j": patch_count, "r": patch_count, "f": patch_count}
    model.num_unknowns = {"j/r": 2 * features * patch_count, "f": features * patch_count}
    model.scale = scale
    if collocation is not None:
        model.collocation_sizes = {"interior": tuple(collocation), "boundary": int(max(collocation))}
    config.mesh.strides = {"x": 1.0 / partitions[0], "v": 1.0 / partitions[1]}
    domain, strides = dict(config.mesh.domain), dict(config.mesh.strides)
    key = random.key(seed)
    # The retained feature module historically used a module seed. Set it before
    # constructing every Flax module so the five declared seeds are effective.
    function_space.seedXV = seed
    sampler = Sample1D(domain, dict(model.collocation_sizes), model.sample_mode)
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
        **common,
        num_quads=int(model.num_quads),
        coeff_fns=dict(model.coeff),
    )
    boundary = OddEvenDecompositionPointwiseBoundaryConstraint1D(**common)
    dummy = (jnp.zeros((1,)), jnp.ones((1,)))
    interior_params = interior.init(key, *dummy)
    boundary_params = boundary.init(key, *dummy)
    interior_fn = vmap(lambda x, v: interior.apply(interior_params, x, v))
    boundary_fn = vmap(lambda x, v: boundary.apply(boundary_params, x, v))
    feature_seconds = perf_counter() - started

    started = perf_counter()
    left_x, left_v = sampler.pts_left
    right_x, right_v = sampler.pts_right
    boundary_matrix = np.concatenate(
        (
            np.asarray(boundary_fn(left_x, left_v)),
            np.asarray(boundary_fn(right_x, right_v)),
        )
    )
    boundary_rhs = np.concatenate(
        (
            np.asarray(vmap(model.bdy_cond.f_l)(left_v)).reshape(-1),
            np.asarray(vmap(model.bdy_cond.f_r)(right_v)).reshape(-1),
        )
    )
    int_x, int_v = sampler.pts_int
    interior_matrix = np.asarray(interior_fn(int_x, int_v)).reshape(
        -1, int(model.num_unknowns["j/r"])
    )
    q_plus = np.asarray(vmap(model.source)(int_x, int_v)).reshape(-1)
    q_minus = np.asarray(vmap(model.source)(int_x, -int_v)).reshape(-1)
    q_even = 0.5 * (q_plus + q_minus)
    q_odd = 0.5 * (q_plus - q_minus)
    # The three AP equations are ordered exactly as returned by the constraint.
    interior_rhs = np.column_stack(
        (
            np.zeros_like(q_even),
            epsilon**2 * q_even,
            epsilon * q_odd,
        )
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

    started = perf_counter()
    constructor = OddEvenDecompositionConstructor1D(
        **common, coefficients=jnp.asarray(coefficients.reshape(-1))
    )
    constructor_params = constructor.init(key, *dummy)
    approximation = vmap(
        lambda x, v: constructor.apply(constructor_params, x, v)
    )
    nx, nv = map(int, config.protocol.evaluation_grid)
    x = jnp.linspace(domain["x"][0], domain["x"][1], nx)
    v = jnp.linspace(-1.0, 1.0, nv)
    xx, vv = jnp.meshgrid(x, v, indexing="ij")
    numerical = np.asarray(approximation(xx.reshape(-1, 1), vv.reshape(-1, 1))).reshape(nx, nv)
    exact = 1.0 - np.asarray(x)[:, None] + np.zeros((1, nv))
    relative_l2_f = float(np.linalg.norm(numerical - exact) / np.linalg.norm(exact))
    numerical_rho = np.trapezoid(numerical, np.asarray(v), axis=1) / 2.0
    exact_rho = 1.0 - np.asarray(x)
    relative_l2_rho = float(np.linalg.norm(numerical_rho - exact_rho) / np.linalg.norm(exact_rho))
    evaluation_seconds = perf_counter() - started

    record = {
        "problem": "p1",
        "method": "oe_aprfm",
        "epsilon": epsilon,
        "seed": seed,
        "relative_l2_f": relative_l2_f,
        "relative_l2_rho": relative_l2_rho,
        "residual_half": diagnostics["normalized_residual_rms"],
        "empirical_stability_ratio": diagnostics["normalized_residual_rms"] / (relative_l2_f + relative_l2_rho),
        "condition_number": diagnostics["condition_number"],
        "rank": diagnostics["rank"],
        "coefficient_norm": diagnostics["coefficient_norm"],
        "num_rows": int(matrix.shape[0]),
        "num_columns": int(matrix.shape[1]),
        "feature_seconds": feature_seconds,
        "assembly_seconds": assembly_seconds,
        "solve_seconds": solve_seconds,
        "evaluation_seconds": evaluation_seconds,
        "total_seconds": feature_seconds + assembly_seconds + solve_seconds + evaluation_seconds,
        "rcond": rcond,
        "partitions": partitions,
        "strides": dict(config.mesh.strides),
        "features_per_patch": features,
        "scale": scale,
        "block_weights": block_weights,
        "collocation": dict(model.collocation_sizes),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    stem = f"p1_oe_aprfm_eps_{epsilon:.0e}_seed_{seed}{suffix}"
    (output_dir / f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
    np.savez_compressed(output_dir / f"{stem}.npz", x=np.asarray(x), v=np.asarray(v), f=numerical, rho=numerical_rho, exact=exact, exact_rho=exact_rho)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--partitions", type=int, nargs=2, default=(1, 1))
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--rcond", type=float, default=1.0e-12)
    parser.add_argument("--block-weights", type=float, nargs=4)
    parser.add_argument("--collocation", type=int, nargs=2)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    print(json.dumps(run(args.epsilon, args.seed, args.output_dir, partitions=tuple(args.partitions), features=args.features, scale=args.scale, rcond=args.rcond, block_weights=None if args.block_weights is None else tuple(args.block_weights), collocation=None if args.collocation is None else tuple(args.collocation), tag=args.tag), indent=2))


if __name__ == "__main__":
    main()
