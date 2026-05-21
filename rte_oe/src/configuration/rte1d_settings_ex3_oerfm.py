import ml_collections
import jax.numpy as jnp
import jax.nn as jnn


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.mesh = dict(
        domain=dict(x=(0, 1), v=(0, 1)),
        strides=dict(x=0.5, v=0.5),
    )
    Domain = config.mesh.domain
    Strides = config.mesh.strides
    num_x = int((Domain["x"][-1] - Domain["x"][0]) / Strides["x"])
    num_v = int((Domain["v"][-1] - Domain["v"][0]) / Strides["v"])

    config.model = dict(
        eqn_type="rte_1d",
        knudsen_number=5e-1,
        coeff={
            "scattering": lambda x: (
                1e-2 + 0.5 * (jnp.tanh(6.5 - 11 * x) + jnp.tanh(11 * x - 4.5))
            ),
            "absorption": lambda x: 0.0,
        },
        Mp=dict(j=num_x * num_v, r=num_x * num_v, f=num_x * num_v),
        Jn=dict(j=64, r=64, f=128),
        scale=1.0,
        activation=lambda x: jnn.tanh(x),
        random_seed=42,
        num_quads=16,
        sample_mode="uniform",
        # {"uniform": "uniform", "random": "random", "lhs": "lhs"},
        collocation_sizes={"interior": (64, 64), "boundary": 128},
        bdy_cond={
            "f_l": lambda v: 0.5,
            "f_r": lambda v: 0.0,
        },
        regularizer=1.0,
    )
    Jn = config.model.Jn
    Mp = config.model.Mp
    config.model.num_unknowns = {
        "j/r": Jn["j"] * Mp["j"] + Jn["r"] * Mp["r"],
        "f": Jn["f"] * Mp["f"],
    }
    # for aprfm
    config.model.source = lambda x, v: jnp.zeros_like(v)
    config.model.grid_sizes = (64, 64)
    return config
