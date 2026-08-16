"""P3: smooth square-domain manufactured solution."""

import jax.numpy as jnp

from configuration.experiment_defaults import build_config


def get_config(epsilon: float = 1.0):
    def exact(x, y, theta):
        return jnp.exp(-x - y) + 0.0 * theta

    return build_config(
        problem="p3",
        dimension=2,
        domain={"x": (-1.0, 1.0), "y": (-1.0, 1.0), "theta": (0.0, 0.5 * jnp.pi)},
        partitions=(1, 1, 1),
        knudsen_number=epsilon,
        scattering=lambda x, y: 1.0,
        absorption=lambda x, y: 0.0,
        source=lambda x, y, theta: (
            -(jnp.cos(theta) + jnp.sin(theta)) * jnp.exp(-x - y) / epsilon
        ),
        boundary={
            "f_l": lambda y: jnp.exp(1.0 - y),
            "f_r": lambda y: jnp.exp(-1.0 - y),
            "f_b": lambda x: jnp.exp(1.0 - x),
            "f_t": lambda x: jnp.exp(-1.0 - x),
        },
        exact_solution=exact,
        outputs=("relative_l2_f", "relative_l2_rho", "density_error_map", "time"),
    )
