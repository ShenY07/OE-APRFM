import jax.numpy as jnp
# from jax import grad


def psi(x: jnp.ndarray) -> jnp.ndarray:
    funcs = [
        lambda x: jnp.ones_like(x),
        lambda x: 0.5 * (1.0 - jnp.sin(2.0 * jnp.pi * jnp.abs(x))),
        lambda x: jnp.zeros_like(x),
    ]
    value = jnp.where(
        jnp.abs(x) < 3 / 4,
        funcs[0](x),
        jnp.where(
            jnp.abs(x) > 5 / 4,
            funcs[-1](x),
            funcs[1](x),
        ),
    )
    return value
    # return jnp.exp(-(x**2))


def dpsi(x: jnp.ndarray) -> jnp.ndarray:
    funcs = [
        lambda x: jnp.zeros_like(x),
        lambda x: -jnp.pi * jnp.cos(2.0 * jnp.pi * x) * jnp.sign(x),
        lambda x: jnp.zeros_like(x),
    ]
    value = jnp.where(
        jnp.abs(x) < 3 / 4,
        funcs[0](x),
        jnp.where(
            jnp.abs(x) > 5 / 4,
            funcs[-1](x),
            funcs[1](x),
        ),
    )
    return value
    # return -2.0 * x * psi(x)
