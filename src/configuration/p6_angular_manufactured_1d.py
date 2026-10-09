"""P6: manufactured solution with nontrivial even and scaled-odd parts."""

import jax.numpy as jnp

from configuration.experiment_defaults import build_config


def get_config(epsilon: float = 1.0):
    rho = lambda x: 1.0 + 0.25 * jnp.sin(2.0 * jnp.pi * x)
    g = lambda x: 0.25 * jnp.cos(jnp.pi * x)
    rho_prime = lambda x: 0.5 * jnp.pi * jnp.cos(2.0 * jnp.pi * x)
    g_prime = lambda x: -0.25 * jnp.pi * jnp.sin(jnp.pi * x)
    exact = lambda x, v: rho(x) + epsilon * v * g(x)
    source = lambda x, v: v * (
        rho_prime(x) + g(x)
    ) / epsilon + v**2 * g_prime(x)
    return build_config(
        problem="p6",
        dimension=1,
        domain={"x": (0.0, 1.0), "v": (0.0, 1.0)},
        partitions=(1, 1),
        knudsen_number=epsilon,
        scattering=lambda x: 1.0,
        absorption=lambda x: 0.0,
        source=source,
        boundary={
            "f_l": lambda v: exact(0.0, v),
            "f_r": lambda v: exact(1.0, -v),
        },
        exact_solution=exact,
        outputs=(
            "relative_l2_f",
            "relative_l2_rho",
            "relative_l2_r",
            "relative_l2_j",
        ),
    )
