"""Provide rte 2d settings functionality for the configuration layer.

Distinction: This file is the canonical implementation in this module family; numerical logic is preserved.
"""

import jax.numpy as jnp
import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.mesh = dict(
        domain=dict(
            x=(-1.0, 1.0),
            y=(-1.0, 1.0),
            theta=(0.0, 2.0 * jnp.pi),
        ),
        strides=dict(x=2.0 / 128.0, y=2.0 / 128.0, theta=2.0 * jnp.pi / 64.0),
    )

    domain = config.mesh.domain
    strides = config.mesh.strides
    num_x = int(round((domain["x"][-1] - domain["x"][0]) / strides["x"]))
    num_y = int(round((domain["y"][-1] - domain["y"][0]) / strides["y"]))
    num_theta = int(
        round((domain["theta"][-1] - domain["theta"][0]) / strides["theta"])
    )

    config.model = dict(
        eqn_type="rte_2d",
        knudsen_number=1.0e-3,
        coeff={
            "scattering": lambda x, y: 1.0,
            "absorption": lambda x, y: 0.2,
        },
        Mp=dict(
            rho=num_x * num_y,
            g=num_x * num_y * num_theta,
            f=num_x * num_y * num_theta,
        ),
        Jn=dict(rho=64, g=128, f=128),
        scale=1.0,
        activation=lambda x: jnp.tanh(x),
        random_seed=42,
        num_quads=16,
        sample_mode="uniform",
        collocation_sizes={"interior": (32, 32, 32), "boundary": (64, 64, 64)},
        bdy_cond={
            "f_l": lambda y: 0.1 * jnp.ones_like(y),
            "f_r": lambda y: 0.1 * jnp.ones_like(y),
            "f_b": lambda x: 0.1 * jnp.ones_like(x),
            "f_t": lambda x: 0.1 * jnp.ones_like(x),
        },
        regularizer=1.0,
        grid_sizes=(num_x, num_y, num_theta),
        experiment=dict(
            exact_solution=None,
            description=(
                "Example 9: constant scattering and positive absorption, "
                "compatible nonzero inflow, and a smooth low-frequency spatial "
                "source; accuracy is measured against a numerical reference."
            ),
        ),
    )

    config.model.num_unknowns = {
        "rho/g": config.model.Jn["rho"] * config.model.Mp["rho"]
        + config.model.Jn["g"] * config.model.Mp["g"],
        "f": config.model.Jn["f"] * config.model.Mp["f"],
    }

    config.model.source = lambda x, y, theta: (
        0.5
        + 0.1 * jnp.cos(0.5 * jnp.pi * x) * jnp.cos(0.5 * jnp.pi * y)
        + 0.0 * theta
    )
    return config
