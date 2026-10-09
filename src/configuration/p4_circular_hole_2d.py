"""P4: corrected manufactured problem on a square with a circular hole."""

import jax.numpy as jnp

from configuration.experiment_defaults import build_config


def get_config(epsilon: float = 1.0):
    def exact(x, y, theta):
        return 1.0 / (1.0 + x**2 + y**2) + 0.0 * theta

    def trace(x, y):
        return 1.0 / (1.0 + x**2 + y**2)

    return build_config(
        problem="p4",
        dimension=2,
        domain={
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
            "theta": (0.0, 0.5 * jnp.pi),
            "hole": {"center": (0.0, 0.0), "radius": 0.5},
        },
        partitions=(1, 1, 2),
        knudsen_number=epsilon,
        scattering=lambda x, y: 1.0,
        absorption=lambda x, y: 0.0,
        source=lambda x, y, theta: (
            -2.0
            * (x * jnp.cos(theta) + y * jnp.sin(theta))
            / (epsilon * (1.0 + x**2 + y**2) ** 2)
        ),
        boundary={
            "trace": trace,
            "f_l": lambda y: trace(-1.0, y),
            "f_r": lambda y: trace(1.0, y),
            "f_b": lambda x: trace(x, -1.0),
            "f_t": lambda x: trace(x, 1.0),
            "f_circle": trace,
        },
        exact_solution=exact,
        outputs=(
            "relative_l2_rho",
            "relative_l2_f",
            "radial_cut",
            "time",
            "memory",
        ),
    )
