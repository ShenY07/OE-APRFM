import ml_collections
import jax.numpy as jnp


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.mesh = dict(
        domain=dict(
            x=(-2, 2),
            y=(-3, 3),
            theta=(0, jnp.pi / 2),
            hole=dict(x=(-1, 2), y=(-2, 2)),
        ),
        strides=dict(x=4, y=6, theta=jnp.pi / 4),
    )
    Domain = config.mesh.domain
    Strides = config.mesh.strides
    num_x = int((Domain["x"][-1] - Domain["x"][0]) / Strides["x"])
    num_y = int((Domain["y"][-1] - Domain["y"][0]) / Strides["y"])
    num_v = int((Domain["theta"][-1] - Domain["theta"][0]) / Strides["theta"])
    config.model = dict(
        eqn_type="rte_2d",
        knudsen_number=1e0,
        # knudsen_number=1e-3,
        coeff={
            "scattering": lambda x, y: 1.0,
            "absorption": lambda x, y: 0.0,
        },
        Mp=dict(
            j=num_x * num_y * num_v,
            r=num_x * num_y * num_v,
            f=num_x * num_y * num_v,
        ),
        Jn=dict(j=128, r=128, f=64),
        scale=1.0,
        # activation=lambda x: jnp.sin(jnp.pi * x),
        activation=lambda x: jnp.tanh(x),
        random_seed=42,
        num_quads=16,
        sample_mode="uniform",
        # sample_mode={"uniform": "uniform", "random": "random", "lhs": "lhs"},
        collocation_sizes={"interior": (16, 16, 32), "boundary": (32, 32, 32)},
        regularizer=1.0,
    )
    kn = config.model.knudsen_number
    Jn = config.model.Jn
    Mp = config.model.Mp
    config.model.num_unknowns = {
        "j/r": Jn["j"] * Mp["j"] + Jn["r"] * Mp["r"],
        "f": Jn["f"] * Mp["f"],
    }
    config.model.source = lambda x, y, theta: (
        -(jnp.cos(theta) + jnp.sin(theta)) / kn * jnp.exp(-x - y)
    )
    config.model.grid_sizes = (64, 64, 64)
    return config
