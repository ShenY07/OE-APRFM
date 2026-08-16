"""Run matched 1-D MM-APRFM baselines using the vendored implementation."""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("JAX_ENABLE_X64", "true")

ROOT = Path(__file__).resolve().parents[1]
CURRENT_SRC = ROOT / "src"
VENDORED_SRC = ROOT / "baselines" / "mm_oerfm" / "original" / "src"
sys.path.insert(0, str(CURRENT_SRC))
from configuration.p1_manufactured_1d import get_config as get_p1_config
from configuration.p2_heterogeneous_1d import get_config as get_p2_config

# Keep the original MM implementation isolated from the OE implementation.
sys.path.insert(0, str(VENDORED_SRC))
import jax
import jax.numpy as jnp
from jax import jit, random, vmap
import numpy as np
import hashlib
from scipy.linalg import lstsq, svdvals

from constraints.continuous1d import (
    MicroMacroPointwiseBoundaryConstraint1D,
    MicroMacroPointwiseInteriorConstraint1D,
)
from modules.generator import Sample1D
from modules.solution import MicroMacroConstructor1D
import modules.func_space as mm_function_space
from utils.integrate import leggauss

jax.config.update("jax_enable_x64", True)


def integral(fn, quadratures, argnum):
    points, weights = quadratures

    def apply(*args):
        args = list(args)
        axes = [None] * len(args)
        args.insert(argnum, points)
        axes.insert(argnum, 0)
        values = vmap(fn, in_axes=axes, out_axes=-1)(*args)
        return jnp.asarray(jnp.dot(values, weights).squeeze())

    return apply


def run(problem, epsilon, seed, output_dir, *, partitions, rho_features, g_features, rcond, collocation):
    config = (get_p1_config if problem == "p1" else get_p2_config)(epsilon)
    model = config.model
    px, pv = partitions
    domain = {"x": tuple(config.mesh.domain.x), "v": (-1.0, 1.0)}
    strides = {
        "x": (domain["x"][1] - domain["x"][0]) / px,
        "v": 2.0 / pv,
    }
    jn = {"rho": rho_features, "g": g_features}
    unknowns = rho_features * px + g_features * px * pv
    key = random.key(seed)
    # The vendored feature spaces retain legacy module-level RNG seeds.
    # Set both explicitly so independent realizations are actually distinct.
    mm_function_space.seedX = seed
    mm_function_space.seedXV = seed
    source = (
        (lambda x, v: -v)
        if problem == "p1"
        else (lambda x, v: jnp.zeros_like(v))
    )
    common = dict(
        domain=domain,
        strides=strides,
        Jn=jn,
        scale=1.0,
        init_rng=key,
        kn=epsilon,
        activation=jnp.tanh,
    )
    started = perf_counter()
    equation = MicroMacroPointwiseInteriorConstraint1D(
        **common, num_quads=8, coeff_fns=dict(model.coeff)
    )
    boundary = MicroMacroPointwiseBoundaryConstraint1D(**common)
    params = {}
    eq_fn = jit(vmap(lambda x, v: equation.apply(params, x, v)))
    bc_fn = jit(vmap(lambda x, v: boundary.apply(params, x, v)))
    quadrature = leggauss(8)

    def rhs(x, v):
        q = source(x, v).squeeze()
        average = integral(source, quadrature, 1)(x)
        return jnp.array([average, q - average, 0.0])

    rhs_fn = jit(vmap(rhs))
    feature_seconds = perf_counter() - started
    collocation_sizes = {"interior": tuple(collocation), "boundary": 64}
    sample = Sample1D(domain, collocation_sizes, "uniform")
    started = perf_counter()
    left = np.asarray(bc_fn(*sample.pts_left))
    right = np.asarray(bc_fn(*sample.pts_right))
    interior = np.asarray(eq_fn(*sample.pts_int)).reshape(-1, unknowns)
    rhs_interior = np.asarray(rhs_fn(*sample.pts_int)).reshape(-1)
    matrix = np.concatenate((left, right, interior), axis=0)
    vector = np.empty(matrix.shape[0])
    boundary_count = left.shape[0]
    if problem == "p1":
        vector[:boundary_count] = 1.0
        vector[boundary_count : 2 * boundary_count] = 0.0
    else:
        vector[:boundary_count] = 0.5
        vector[boundary_count : 2 * boundary_count] = 0.0
    vector[2 * boundary_count :] = rhs_interior
    norms = np.linalg.norm(matrix, axis=1)
    keep = norms > np.finfo(float).eps
    scaled_matrix = matrix[keep] / norms[keep, None]
    scaled_vector = vector[keep] / norms[keep]
    assembly_seconds = perf_counter() - started
    started = perf_counter()
    coefficients, _, rank, singular_values = lstsq(
        scaled_matrix, scaled_vector, cond=rcond, lapack_driver="gelsd"
    )
    solve_seconds = perf_counter() - started
    retained = singular_values[singular_values > rcond * singular_values[0]]
    condition_number = float(retained[0] / retained[-1])
    constructor = MicroMacroConstructor1D(
        **common, coefficients=jnp.asarray(coefficients)
    )
    approx = jit(vmap(lambda x, v: constructor.apply(params, x, v)))
    started = perf_counter()
    if problem == "p1":
        x = np.linspace(0.0, 1.0, 257)
        velocity = np.linspace(-1.0, 1.0, 128)
        reference = 1.0 - x[:, None] + np.zeros((1, velocity.size))
        weights = None
    else:
        reference_path = ROOT / "results" / "references" / f"p2_parity_ref_eps_{epsilon:.0e}_level_B.npz"
        with np.load(reference_path) as data:
            x, velocity, reference, weights = data["x"], data["velocity"], data["f"], data["weights"]
    numerical = np.empty_like(reference)
    for begin in range(0, x.size, 64):
        xx, vv = np.meshgrid(x[begin : begin + 64], velocity, indexing="ij")
        numerical[begin : begin + 64] = np.asarray(
            approx(jnp.asarray(xx.reshape(-1, 1)), jnp.asarray(vv.reshape(-1, 1)))
        ).reshape(xx.shape)
    rho = np.trapezoid(numerical, velocity, axis=1) / 2.0
    reference_rho = np.trapezoid(reference, velocity, axis=1) / 2.0
    if weights is None:
        error_f = float(np.linalg.norm(numerical - reference) / np.linalg.norm(reference))
    else:
        error_f = float(np.sqrt(np.sum(weights[None, :] * (numerical-reference)**2) / np.sum(weights[None, :] * reference**2)))
    error_rho = float(np.linalg.norm(rho - reference_rho) / np.linalg.norm(reference_rho))
    evaluation_seconds = perf_counter() - started
    record = {
        "problem": problem,
        "method": "mm_aprfm",
        "epsilon": epsilon,
        "seed": seed,
        "relative_l2_f": error_f,
        "relative_l2_rho": error_rho,
        "condition_number": condition_number,
        "rank": int(rank),
        "num_rows": int(scaled_matrix.shape[0]),
        "num_columns": int(unknowns),
        "oversampling_ratio": float(scaled_matrix.shape[0] / unknowns),
        "feature_matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
        "partitions": list(partitions),
        "rho_features_per_patch": rho_features,
        "g_features_per_patch": g_features,
        "total_features": unknowns,
        "rcond": rcond,
        "collocation": collocation_sizes,
        "feature_seconds": feature_seconds,
        "assembly_seconds": assembly_seconds,
        "solve_seconds": solve_seconds,
        "evaluation_seconds": evaluation_seconds,
        "total_seconds": feature_seconds + assembly_seconds + solve_seconds + evaluation_seconds,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{problem}_mm_aprfm_eps_{epsilon:.0e}_seed_{seed}"
    (output_dir / f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
    np.savez_compressed(
        output_dir / f"{stem}.npz",
        x=x,
        velocity=velocity,
        f=numerical,
        rho=rho,
        reference_f=reference,
        reference_rho=reference_rho,
        error_f=np.abs(numerical - reference),
        error_rho=np.abs(rho - reference_rho),
    )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=("p1", "p2"), required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--partitions", type=int, nargs=2, default=(1, 1))
    parser.add_argument("--features", type=int, default=64, help="Set both rho and g features")
    parser.add_argument("--rho-features", type=int)
    parser.add_argument("--g-features", type=int)
    parser.add_argument("--rcond", type=float, default=1e-12)
    parser.add_argument("--collocation", type=int, nargs=2, default=(32, 34))
    parser.add_argument("--output-dir", type=Path, default=Path("results/baselines/mm_aprfm"))
    args = parser.parse_args()
    print(json.dumps(run(args.problem, args.epsilon, args.seed, args.output_dir, partitions=tuple(args.partitions), rho_features=args.rho_features or args.features, g_features=args.g_features or args.features, rcond=args.rcond, collocation=tuple(args.collocation)), indent=2))
