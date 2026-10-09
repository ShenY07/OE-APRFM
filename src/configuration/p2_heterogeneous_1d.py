"""P2: heterogeneous pure scattering with an SI+DSA reference."""

import jax.numpy as jnp

from configuration.experiment_defaults import build_config


def get_config(epsilon: float = 1.0):
    return build_config(
        problem="p2",
        dimension=1,
        domain={"x": (0.0, 1.0), "v": (0.0, 1.0)},
        partitions=(2, 2),
        knudsen_number=epsilon,
        scattering=lambda x: (
            0.01 + (jnp.tanh(6.5 - 11.0 * x) + jnp.tanh(11.0 * x - 4.5)) / 2.0
        ),
        absorption=lambda x: 0.0,
        source=lambda x, v: jnp.zeros_like(v),
        boundary={"f_l": lambda v: 0.5, "f_r": lambda v: 0.0},
        reference="two_level_si_dsa",
        outputs=("relative_l2_f", "relative_l2_rho", "time", "iterations"),
    )
