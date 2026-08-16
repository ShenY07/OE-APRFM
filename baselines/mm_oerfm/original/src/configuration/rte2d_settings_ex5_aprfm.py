import ml_collections
import jax.numpy as jnp


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.mesh = dict(
        domain=dict(x=(-1, 1), y=(-1, 1), theta=(0.0, 2.0 * jnp.pi)),
        strides=dict(x=2, y=2, theta=0.5 * jnp.pi),
    )
    Domain = config.mesh.domain
    Strides = config.mesh.strides
    num_x = int((Domain["x"][-1] - Domain["x"][0]) / Strides["x"])
    num_y = int((Domain["y"][-1] - Domain["y"][0]) / Strides["y"])
    num_v = int((Domain["theta"][-1] - Domain["theta"][0]) / Strides["theta"])
    config.model = dict(
        eqn_type="rte_2d",
        # knudsen_number=1e0,
        knudsen_number=1e-1,
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
        collocation_sizes={
            "interior": (32, 32, 32),
            "boundary": (32, 32, 32),
        },
        bdy_cond={
            "f_l": lambda y: 0.0 * y,
            "f_r": lambda y: 0.0 * y,
            "f_b": lambda x: 0.0 * x,
            "f_t": lambda x: 0.0 * x,
        },
        regularizer=1.0,
    )
    Jn = config.model.Jn
    Mp = config.model.Mp
    config.model.num_unknowns = {
        "rho/g": Jn["rho"] * Mp["rho"] + Jn["g"] * Mp["g"],
        "f": Jn["f"] * Mp["f"],
    }
    config.model.source = lambda x, y, theta: 0.5
    config.model.grid_sizes = (64, 64, 32)
    return config
