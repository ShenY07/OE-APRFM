"""P1: source-driven manufactured problem for the full epsilon sweep."""

from configuration.experiment_defaults import build_config


def get_config(epsilon: float = 1.0):
    return build_config(
        problem="p1",
        dimension=1,
        domain={"x": (0.0, 1.0), "v": (0.0, 1.0)},
        partitions=(1, 1),
        knudsen_number=epsilon,
        scattering=lambda x: 1.0,
        absorption=lambda x: 0.0,
        source=lambda x, v: -v / epsilon,
        boundary={"f_l": lambda v: 1.0, "f_r": lambda v: 0.0},
        exact_solution=lambda x, v: 1.0 - x + 0.0 * v,
        outputs=("relative_l2_f", "scaled_condition_number", "time"),
    )
