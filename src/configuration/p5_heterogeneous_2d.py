"""P5: heterogeneous 2D benchmark with a smooth manufactured solution."""

import jax.numpy as jnp

from configuration.experiment_defaults import build_config


SOURCE_CENTER = (-0.65, 0.55)
SOURCE_WIDTH = 0.12
SOURCE_AMPLITUDE = 1.0
INTERFACE_WIDTH = 0.02


def scattering(x, y):
    """Fixed tanh regularization; the channel blend has precedence."""
    radius = jnp.sqrt(x**2 + y**2 + 1.0e-30)
    disk_weight = 0.5 * (1.0 - jnp.tanh((radius - 0.35) / INTERFACE_WIDTH))
    disk_medium = 1.0 + 9.0 * disk_weight
    channel_distance = jnp.abs(y - 0.35 * x)
    channel_weight = 0.5 * (
        1.0 - jnp.tanh((channel_distance - 0.10) / INTERFACE_WIDTH)
    )
    return (1.0 - channel_weight) * disk_medium + 0.05 * channel_weight


def absorption(x, y):
    """Use the manuscript's optional absorption 0.1 inside the disk."""
    radius = jnp.sqrt(x**2 + y**2 + 1.0e-30)
    disk_weight = 0.5 * (1.0 - jnp.tanh((radius - 0.35) / INTERFACE_WIDTH))
    return 0.01 + 0.09 * disk_weight


def exact_solution(x, y, theta):
    """Positive, genuinely two-dimensional, angle-independent exact solution."""
    return jnp.exp(-0.35 * x - 0.2 * y) * (
        1.0 + 0.15 * jnp.cos(jnp.pi * x) * jnp.cos(jnp.pi * y)
    ) + 0.0 * theta


def get_config(epsilon: float = 1.0):
    def source(x, y, theta):
        base = jnp.exp(-0.35 * x - 0.2 * y)
        cx, cy = jnp.cos(jnp.pi * x), jnp.cos(jnp.pi * y)
        sx, sy = jnp.sin(jnp.pi * x), jnp.sin(jnp.pi * y)
        dx = base * (-0.35 * (1.0 + 0.15 * cx * cy) - 0.15 * jnp.pi * sx * cy)
        dy = base * (-0.2 * (1.0 + 0.15 * cx * cy) - 0.15 * jnp.pi * cx * sy)
        return (jnp.cos(theta) * dx + jnp.sin(theta) * dy) / epsilon + absorption(x, y) * exact_solution(x, y, theta)

    config = build_config(
        problem="p5",
        dimension=2,
        domain={"x": (-1.0, 1.0), "y": (-1.0, 1.0), "theta": (0.0, 0.5 * jnp.pi)},
        partitions=(1, 1, 1),
        knudsen_number=epsilon,
        scattering=scattering,
        absorption=absorption,
        source=source,
        boundary={
            "f_l": lambda y: exact_solution(-1.0, y, 0.0),
            "f_r": lambda y: exact_solution(1.0, y, 0.0),
            "f_b": lambda x: exact_solution(x, -1.0, 0.0),
            "f_t": lambda x: exact_solution(x, 1.0, 0.0),
        },
        exact_solution=exact_solution,
        outputs=(
            "relative_l2_rho",
            "relative_l2_f",
            "scaled_condition_number",
            "cuts",
            "time",
            "memory",
            "iterations",
        ),
    )
    config.problem_data = {
        "source_center": SOURCE_CENTER,
        "source_width": SOURCE_WIDTH,
        "source_amplitude": SOURCE_AMPLITUDE,
        "coefficient_regularization": "tanh",
        "interface_width": INTERFACE_WIDTH,
        "absorption_background": 0.01,
        "absorption_disk": 0.1,
        "channel_precedence": True,
        "description": (
            "Smooth heterogeneous manufactured benchmark with an analytic "
            "reference at epsilon 1 and 1e-3."
        ),
    }
    return config
