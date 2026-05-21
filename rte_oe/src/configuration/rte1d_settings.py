import ml_collections
import jax.numpy as jnp


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.mesh = dict(
        domain=dict(x=(0, 1), v=(-1, 1)),
        strides=dict(x=1.0, v=1.0),
    )
    Domain = config.mesh.domain
    Strides = config.mesh.strides
    num_x = int((Domain["x"][-1] - Domain["x"][0]) / Strides["x"])
    num_v = int((Domain["v"][-1] - Domain["v"][0]) / Strides["v"])

    config.model = dict(
        eqn_type="rte_1d",
        knudsen_number=1e0,
        coeff={
            "scattering": lambda x: 1.0,
            # "scattering": lambda x: 1e-2 + 0.5 * (jnp.tanh(6.5 - 11 * x) + jnp.tanh(11 * x - 4.5)),
            "absorption": lambda x: 0.0,
        },
        Mp=dict(rho=num_x, g=num_x * num_v, f=num_x * num_v),
        Jn=dict(rho=64, g=64, f=64),
        scale=1.0,
        activation=lambda x: jnp.tanh(x),
        random_seed=42,
        num_quads=16,
        sample_mode="uniform",
        # {"uniform": "uniform", "random": "random", "lhs": "lhs"},
        collocation_sizes={"interior": (32, 32), "boundary": 64},
        bdy_cond={
            "f_l": lambda v: 1.0,  # 0.5,
            "f_r": lambda v: 0.0,
        },
        regularizer=1.0,
    )
    Jn = config.model.Jn
    Mp = config.model.Mp
    config.model.num_unknowns = {
        "rho/g": Jn["rho"] * Mp["rho"] + Jn["g"] * Mp["g"],
        "f": Jn["f"] * Mp["f"],
    }
    # for rfm
    # config.model.source = lambda x, v: -v / config.model.knudsen_number
    # for aprfm
    config.model.source = lambda x, v: -v / config.model.knudsen_number
    config.model.grid_sizes = (1024, 2048)
    return config
