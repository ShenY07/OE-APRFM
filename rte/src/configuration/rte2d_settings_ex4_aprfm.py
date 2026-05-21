import ml_collections
import jax.numpy as jnp


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.mesh = dict(
        domain=dict(x=(-1, 1), y=(-1, 1), theta=(0.0, 2.0 * jnp.pi)),
        strides=dict(x=2, y=2, theta=jnp.pi / 2),
    )
    Domain = config.mesh.domain
    Strides = config.mesh.strides
    num_x = int((Domain["x"][-1] - Domain["x"][0]) / Strides["x"])
    num_y = int((Domain["y"][-1] - Domain["y"][0]) / Strides["y"])
    num_v = int((Domain["theta"][-1] - Domain["theta"][0]) / Strides["theta"])
    config.model = dict(
        eqn_type="rte_2d",
        # knudsen_number=1e0,
        knudsen_number=1e-3,
        coeff={
            "scattering": lambda x, y: 1.0,
            "absorption": lambda x, y: 0.0,
        },
        Mp=dict(
            rho=num_x * num_y, g=num_x * num_y * num_v, f=num_x * num_y * num_v
        ),
        Jn=dict(rho=64, g=128, f=128),
        scale=1.0,
        # activation=lambda x: jnp.sin(jnp.pi * x),
        activation=lambda x: jnp.tanh(x),
        random_seed=42,
        num_quads=16,
        sample_mode="uniform",
        # sample_mode={"uniform": "uniform", "random": "random", "lhs": "lhs"},
        collocation_sizes={"interior": (32, 32, 32), "boundary": (64, 64, 64)},
        bdy_cond={
            "f_l": lambda y: jnp.exp(1 - y),
            "f_r": lambda y: jnp.exp(-1 - y),
            "f_b": lambda x: jnp.exp(1 - x),
            "f_t": lambda x: jnp.exp(-1 - x),
        },
        regularizer=1.0,
    )
    kn = config.model.knudsen_number
    Jn = config.model.Jn
    Mp = config.model.Mp
    config.model.num_unknowns = {
        "rho/g": Jn["rho"] * Mp["rho"] + Jn["g"] * Mp["g"],
        "f": Jn["f"] * Mp["f"],
    }
    config.model.source = lambda x, y, theta: (
        -(jnp.cos(theta) + jnp.sin(theta)) / kn * jnp.exp(-x - y)
    )
    config.model.grid_sizes = (32, 32, 32)
    return config
