"""Run matched square-domain 2-D MM-APRFM baselines (P3 and P5)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("JAX_ENABLE_X64", "true")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from configuration.p3_manufactured_2d import get_config as get_p3_config
from configuration.p4_circular_hole_2d import get_config as get_p4_config
from configuration.p5_heterogeneous_2d import get_config as get_p5_config

sys.path.insert(0, str(ROOT / "baselines" / "mm_oerfm" / "original" / "src"))
import jax
import jax.numpy as jnp
from jax import jit, random, vmap
import numpy as np
import hashlib
from scipy.linalg import lstsq

from constraints.continuous2d import (
    MicroMacroPointwiseBoundaryConstraint2D,
    MicroMacroPointwiseInteriorConstraint2D,
)
from modules.generator import Sample2D
from modules.solution import MicroMacroConstructor2D
import modules.func_space as mm_function_space
from utils.integrate import leggauss

jax.config.update("jax_enable_x64", True)


def run(problem, epsilon, seed, output_dir, *, partitions, features, rcond, collocation):
    config = {"p3": get_p3_config, "p4": get_p4_config, "p5": get_p5_config}[problem](epsilon)
    model = config.model
    px, py, pt = partitions
    domain = {"x": tuple(config.mesh.domain.x), "y": tuple(config.mesh.domain.y), "theta": (0.0, 2.0 * np.pi)}
    strides = {"x": 2.0 / px, "y": 2.0 / py, "theta": 2.0 * np.pi / pt}
    spatial_patches, phase_patches = px * py, px * py * pt
    unknowns = features * (spatial_patches + phase_patches)
    jn = {"rho": features, "g": features}
    key = random.key(seed)
    mm_function_space.seedX = seed
    mm_function_space.seedXV = seed
    common = dict(domain=domain, strides=strides, Jn=jn, scale=1.0, init_rng=key, kn=epsilon, activation=jnp.tanh)
    started = perf_counter()
    equation = MicroMacroPointwiseInteriorConstraint2D(**common, num_quads=16, coeff_fns=dict(model.coeff))
    boundary = MicroMacroPointwiseBoundaryConstraint2D(**common)
    params = {}
    eq_fn = jit(vmap(lambda x, y, t: equation.apply(params, x, y, t)))
    bc_fn = jit(vmap(lambda x, y, t: boundary.apply(params, x, y, t)))
    q_points, q_weights = leggauss(16, (0.0, 2.0 * jnp.pi))

    def rhs(x, y, theta):
        q = model.source(x, y, theta).squeeze()
        average = (vmap(model.source, (None, None, 0))(x, y, q_points).squeeze() @ q_weights / (2.0 * jnp.pi)).squeeze()
        return jnp.array([average, (epsilon * (q - average)).squeeze(), 0.0])

    rhs_fn = jit(vmap(rhs))
    feature_seconds = perf_counter() - started
    sizes = {"interior": tuple(collocation), "boundary": tuple(collocation)}
    sample = Sample2D(domain, sizes, "uniform")
    started = perf_counter()
    outer_points = (sample.pts_left, sample.pts_right, sample.pts_lower, sample.pts_upper)
    boundary_blocks = [np.asarray(bc_fn(*points)) for points in outer_points]
    interior_points = sample.pts_int
    if problem == "p4":
        keep_interior = (interior_points[0].reshape(-1) ** 2 + interior_points[1].reshape(-1) ** 2) >= 0.25
        interior_points = tuple(values[keep_interior] for values in interior_points)
        normal = jnp.linspace(0.0, 2.0 * jnp.pi, 64, endpoint=False)[:, None]
        circle_x = 0.5 * jnp.cos(normal)
        circle_y = 0.5 * jnp.sin(normal)
        offsets = jnp.linspace(-0.49 * jnp.pi, 0.49 * jnp.pi, 32)
        circle_points = (
            jnp.repeat(circle_x, offsets.size, axis=0),
            jnp.repeat(circle_y, offsets.size, axis=0),
            (jnp.repeat(normal, offsets.size, axis=0) + jnp.tile(offsets, normal.shape[0])[:, None]) % (2.0 * jnp.pi),
        )
        boundary_blocks.append(np.asarray(bc_fn(*circle_points)))
    interior = np.asarray(eq_fn(*interior_points)).reshape(-1, unknowns)
    rhs_interior = np.asarray(rhs_fn(*interior_points)).reshape(-1)
    matrix = np.concatenate((*boundary_blocks, interior), axis=0)
    values = (model.bdy_cond.f_l, model.bdy_cond.f_r, model.bdy_cond.f_b, model.bdy_cond.f_t)
    coordinate_indices = (1, 1, 0, 0)
    boundary_values = [np.asarray(vmap(fn)(points[index])).reshape(-1) for points, fn, index in zip(outer_points, values, coordinate_indices)]
    if problem == "p4":
        boundary_values.append(np.asarray(vmap(model.bdy_cond.f_circle)(circle_points[0], circle_points[1])).reshape(-1))
    vector = np.concatenate(boundary_values + [rhs_interior])
    norms = np.linalg.norm(matrix, axis=1)
    keep = norms > np.finfo(float).eps
    matrix = matrix[keep] / norms[keep, None]
    vector = vector[keep] / norms[keep]
    assembly_seconds = perf_counter() - started
    started = perf_counter()
    coefficients, _, rank, singular_values = lstsq(matrix, vector, cond=rcond, lapack_driver="gelsd")
    solve_seconds = perf_counter() - started
    retained = singular_values[singular_values > rcond * singular_values[0]]
    condition_number = float(retained[0] / retained[-1])
    constructor = MicroMacroConstructor2D(**common, coefficients=jnp.asarray(coefficients))
    approx = jit(vmap(lambda x, y, t: constructor.apply(params, x, y, t)))
    nx, ny, na = 65, 65, 64
    x, y = np.linspace(-1.0, 1.0, nx), np.linspace(-1.0, 1.0, ny)
    theta = np.linspace(0.0, 2.0 * np.pi, na, endpoint=False)
    numerical = np.empty((nx, ny, na))
    started = perf_counter()
    for i in range(nx):
        yy, tt = np.meshgrid(y, theta, indexing="ij")
        xx = np.full_like(yy, x[i])
        numerical[i] = np.asarray(approx(jnp.asarray(xx.reshape(-1, 1)), jnp.asarray(yy.reshape(-1, 1)), jnp.asarray(tt.reshape(-1, 1)))).reshape(ny, na)
    xx, yy, tt = np.meshgrid(x, y, theta, indexing="ij")
    if problem == "p3":
        reference = np.exp(-xx - yy)
        mask = np.ones((nx, ny), dtype=bool)
    elif problem == "p4":
        reference_rho = 1.0 / (1.0 + xx[:, :, 0] ** 2 + yy[:, :, 0] ** 2)
        reference = np.broadcast_to(reference_rho[:, :, None], numerical.shape)
        mask = xx[:, :, 0] ** 2 + yy[:, :, 0] ** 2 >= 0.25
    else:
        reference = np.asarray(vmap(model.exact_solution)(jnp.asarray(xx.reshape(-1, 1)), jnp.asarray(yy.reshape(-1, 1)), jnp.asarray(tt.reshape(-1, 1)))).reshape(nx, ny, na)
        mask = np.ones((nx, ny), dtype=bool)
    rho, reference_rho = numerical.mean(axis=2), reference.mean(axis=2)
    phase_mask = np.broadcast_to(mask[:, :, None], numerical.shape)
    error_f = float(np.linalg.norm((numerical-reference)[phase_mask]) / np.linalg.norm(reference[phase_mask]))
    error_rho = float(np.linalg.norm((rho-reference_rho)[mask]) / np.linalg.norm(reference_rho[mask]))
    evaluation_seconds = perf_counter() - started
    record = {"problem": problem, "method": "mm_aprfm", "epsilon": epsilon, "seed": seed, "relative_l2_f": error_f, "relative_l2_rho": error_rho, "condition_number": condition_number, "rank": int(rank), "num_rows": int(matrix.shape[0]), "num_columns": int(unknowns), "oversampling_ratio": float(matrix.shape[0] / unknowns), "feature_matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(), "partitions": list(partitions), "features_per_field_patch": features, "total_features": unknowns, "rcond": rcond, "collocation": list(collocation), "feature_seconds": feature_seconds, "assembly_seconds": assembly_seconds, "solve_seconds": solve_seconds, "evaluation_seconds": evaluation_seconds, "total_seconds": feature_seconds + assembly_seconds + solve_seconds + evaluation_seconds}
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{problem}_mm_aprfm_eps_{epsilon:.0e}_seed_{seed}"
    (output_dir / f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
    np.savez_compressed(output_dir / f"{stem}.npz", x=x, y=y, theta=theta, f=numerical, rho=rho, reference_f=reference, reference_rho=reference_rho, error_f=np.abs(numerical-reference), error_rho=np.abs(rho-reference_rho), mask=mask)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=("p3", "p4", "p5"), required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--partitions", type=int, nargs=3, default=(1, 1, 1))
    parser.add_argument("--features", type=int, default=256)
    parser.add_argument("--rcond", type=float, default=1e-12)
    parser.add_argument("--collocation", type=int, nargs=3, default=(16, 16, 16))
    parser.add_argument("--output-dir", type=Path, default=Path("results/baselines/mm_aprfm"))
    args = parser.parse_args()
    print(json.dumps(run(args.problem, args.epsilon, args.seed, args.output_dir, partitions=tuple(args.partitions), features=args.features, rcond=args.rcond, collocation=tuple(args.collocation)), indent=2))
