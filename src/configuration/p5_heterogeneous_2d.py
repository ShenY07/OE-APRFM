"""P5: smooth, non-manufactured heterogeneous transport in two dimensions."""

import jax.numpy as jnp

from configuration.experiment_defaults import build_config


def scattering(x, y):
    """Smoothly stratified scattering coefficient, 1 <= sigma_s <= 2."""
    return 1.0 + y + 0.0 * x


def absorption(x, y):
    """Constant absorption in the participating medium."""
    return 0.1 + 0.0 * x + 0.0 * y


def source(x, y, theta):
    """The replacement P5 is driven entirely by boundary irradiation."""
    return 0.0 * x + 0.0 * y + 0.0 * theta


def get_config(epsilon: float = 1.0, *, constant_inflow: bool = False):
    left_right = (
        (lambda y: 1.0 + 0.0 * y)
        if constant_inflow
        else (lambda y: 1.0 + 0.2 * y)
    )
    top_value = 1.0 if constant_inflow else 1.2
    config = build_config(
        problem="p5",
        dimension=2,
        domain={
            "x": (0.0, 1.0),
            "y": (0.0, 1.0),
            "theta": (0.0, 0.5 * jnp.pi),
        },
        partitions=(2, 2, 1),
        knudsen_number=epsilon,
        scattering=scattering,
        absorption=absorption,
        source=source,
        boundary={
            "f_l": left_right,
            "f_r": left_right,
            "f_b": lambda x: 1.0 + 0.0 * x,
            "f_t": lambda x: top_value + 0.0 * x,
        },
        reference="oe_sn_refined",
        outputs=(
            "relative_l2_rho",
            "relative_l2_f",
            "normalized_residual",
            "correlation_rho",
            "minimum_f",
            "minimum_rho",
            "time",
        ),
    )

    # One frozen approximation space for both Knudsen numbers.  The v2 2-D
    # odd-even representation has four components (j1, r1, j2, r2).
    config.model.Jn = {"j": 64, "r": 64, "f": 64}
    config.model.Mp = {"j": 4, "r": 4, "f": 4}
    config.model.num_unknowns = {"j/r": 1024, "f": 0}
    config.model.num_quads = 16
    config.model.collocation_sizes = {
        "interior": (32, 32, 16),
        "boundary": (32, 32, 16),
    }
    config.problem_data = {
        "physical_interpretation": (
            "Radiative transport in a smoothly stratified participating medium "
            "subject to diffuse external irradiation."
        ),
        "scattering": "1 + x2",
        "scattering_range": (1.0, 2.0),
        "absorption": 0.1,
        "source": "0",
        "source_range": (0.0, 0.0),
        "source_epsilon_independent": True,
        "boundary_condition": (
            "constant diffuse inflow 1"
            if constant_inflow
            else "diffuse inflow 1 + 0.2 x2"
        ),
        "angular_components": ("j1", "r1", "j2", "r2"),
        "adaptive_sampling": False,
        "description": (
            "Non-manufactured smooth heterogeneous 2-D generalization test "
            "with a deterministic OE-S_N reference."
        ),
    }
    return config
