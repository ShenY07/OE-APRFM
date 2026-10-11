"""Shared, frozen protocol for the five JSC numerical experiments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax.numpy as jnp
import ml_collections

SEEDS = (11, 23, 37)
EPSILON_SWEEPS = {
    "p1": (1.0, 1.0e-3, 1.0e-6),
    "p2": (1.0, 1.0e-3, 1.0e-6),
    "p3": (1.0, 1.0e-3, 1.0e-6),
    "p4": (1.0, 1.0e-3),
    "p5": (1.0, 1.0e-3),
    "p6": (1.0, 1.0e-3, 1.0e-6),
}


def build_config(
    *,
    problem: str,
    dimension: int,
    domain: dict[str, Any],
    partitions: tuple[int, ...],
    knudsen_number: float,
    scattering: Callable[..., Any],
    absorption: Callable[..., Any],
    source: Callable[..., Any],
    boundary: dict[str, Callable[..., Any]] | None = None,
    exact_solution: Callable[..., Any] | None = None,
    reference: str = "exact",
    outputs: Sequence[str] = ("f", "rho"),
) -> ml_collections.ConfigDict:
    """Build one configuration with the reporting metadata kept beside it."""
    if problem not in EPSILON_SWEEPS:
        raise ValueError(f"Unknown core problem: {problem}")
    if dimension == 1:
        axes = ("x", "v")
        feature_count = 64
        collocation = {"interior": (32, 32), "boundary": 64}
        evaluation_grid = (257, 128)
        angular_rule = "gauss_legendre"
    elif dimension == 2:
        axes = ("x", "y", "theta")
        feature_count = 128
        collocation = {
            "interior": (32, 32, 32),
            "boundary": (64, 64, 64),
        }
        evaluation_grid = (129, 129, 128)
        angular_rule = "periodic_trapezoidal"
    else:
        raise ValueError(f"Unsupported spatial dimension: {dimension}")

    strides = {
        axis: (domain[axis][-1] - domain[axis][0]) / count
        for axis, count in zip(axes, partitions)
    }
    patch_count = int(jnp.prod(jnp.asarray(partitions)))
    config = ml_collections.ConfigDict()
    config.problem = problem
    config.mesh = {"domain": domain, "strides": strides}
    config.model = {
        "eqn_type": f"rte_{dimension}d",
        "knudsen_number": knudsen_number,
        "coeff": {"scattering": scattering, "absorption": absorption},
        "Mp": {"j": patch_count, "r": patch_count, "f": patch_count},
        "Jn": {"j": feature_count, "r": feature_count, "f": feature_count},
        "scale": 1.0,
        "activation": jnp.tanh,
        "random_seed": SEEDS[0],
        "num_quads": 8 if dimension == 1 else 32,
        "sample_mode": "uniform",
        "collocation_sizes": collocation,
        "regularizer": 1.0,
        "grid_sizes": evaluation_grid,
        "source": source,
    }
    if boundary is not None:
        config.model.bdy_cond = boundary
    if exact_solution is not None:
        config.model.exact_solution = exact_solution

    config.protocol = {
        "epsilon_values": EPSILON_SWEEPS[problem],
        "seeds": SEEDS,
        "precision": "float64",
        "feature_distribution": "uniform",
        "feature_scale": 1.0,
        "least_squares_solver": "svd",
        "row_scaling": "unit_l2",
        "rcond": 1.0e-12,
        "angular_rule": angular_rule,
        "evaluation_grid": evaluation_grid,
        "evaluation_is_independent": True,
        "reference": reference,
        "reference_refinement_ratio": 2,
        "reference_error_fraction": 0.1,
        "reported_statistics": ("median", "minimum", "maximum"),
        "outputs": tuple(outputs),
    }
    model = config.model
    model.num_unknowns = {
        "j/r": model.Jn.j * model.Mp.j + model.Jn.r * model.Mp.r,
        "f": model.Jn.f * model.Mp.f,
    }
    return config
