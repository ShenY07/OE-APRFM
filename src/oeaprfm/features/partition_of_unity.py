"""Evaluate partition weights and their first derivatives."""

import jax.numpy as jnp


def psi_a(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(jnp.abs(x) <= 1.0, 1.0, 0.0)


def dpsi_a(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros_like(x)


def psi_b(x: jnp.ndarray) -> jnp.ndarray:
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


def dpsi_b(x: jnp.ndarray) -> jnp.ndarray:
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
