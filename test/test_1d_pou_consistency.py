"""The assembled continuous operator must act on the evaluated trial space."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from modules.function_space import RandomFeatureSpaceXV
from constraints.continuous_1d_odd_even import (
    OddEvenDecompositionPointwiseInteriorConstraint1D,
)
from utils.quadrature import leggauss


def test_multipatch_operator_matches_evaluated_field():
    domain = {"x": (0.0, 1.0), "v": (0.0, 1.0)}
    strides = {"x": 0.5, "v": 0.5}
    key = jax.random.key(11)
    field = RandomFeatureSpaceXV(
        domain=domain, strides=strides, Jn=3, scale=1.0, activation=jnp.tanh
    )
    x = jnp.array([0.48])
    v = jnp.array([0.48])
    params = field.init(key, x, v)

    def even(x, v):
        return (
            0.5 * (field.apply(params, x, v) + field.apply(params, x, -v))
        ).ravel()

    def odd(x, v):
        return (
            0.5 * (field.apply(params, x, v) - field.apply(params, x, -v))
        ).ravel()

    op = OddEvenDecompositionPointwiseInteriorConstraint1D(
        domain=domain,
        strides=strides,
        Jn={"j": 3, "r": 3},
        scale=1.0,
        init_rng=key,
        kn=0.1,
        activation=jnp.tanh,
        num_quads=4,
        coeff_fns={"scattering": lambda x: 1.0, "absorption": lambda x: 0.0},
    )
    state = op.init(key, x, v)
    for xx, vv in ((0.48, 0.48), (0.52, -0.52)):
        x = jnp.array([xx])
        v = jnp.array([vv])
        mat = np.asarray(op.apply(state, x, v))
        expected = np.concatenate(
            (
                np.asarray(odd(x, v)),
                vv * np.asarray(jax.jacrev(even, 0)(x, v)).ravel(),
            )
        )
        np.testing.assert_allclose(mat[2], expected, atol=1e-10, rtol=1e-9)
        nodes, weights = leggauss(4, interval=(0.0, 1.0))
        avg_r = sum(
            w * even(x, jnp.atleast_1d(q)) for q, w in zip(nodes, weights)
        )
        avg_vjx = sum(
            w * q * jax.jacrev(odd, 0)(x, jnp.atleast_1d(q)).ravel()
            for q, w in zip(nodes, weights)
        )
        np.testing.assert_allclose(
            mat[0],
            np.concatenate((avg_vjx, np.zeros(12))),
            atol=1e-10,
            rtol=1e-9,
        )
        expected_micro = 0.01 * np.concatenate(
            (vv * jax.jacrev(odd, 0)(x, v).ravel() - avg_vjx, np.zeros(12))
        )
        expected_micro[12:] = np.asarray(even(x, v) - avg_r)
        np.testing.assert_allclose(
            mat[1], expected_micro, atol=1e-10, rtol=1e-9
        )


def test_single_patch_matches_legacy_normalized_space():
    field = RandomFeatureSpaceXV(
        domain={"x": (0.0, 1.0), "v": (0.0, 1.0)},
        strides={"x": 1.0, "v": 1.0},
        Jn=3,
        scale=1.0,
        activation=jnp.tanh,
    )
    x = jnp.array([0.37])
    v = jnp.array([0.2])
    key = jax.random.key(11)
    params = field.init(key, x, v)
    for vv in (0.2, -0.2, 1.0, -1.0):
        v = jnp.array([vv])
        evaluated, raw = field.apply(params, x, v, has_aux=True)
        # For one patch normalization cancels either psi_a or psi_b exactly.
        np.testing.assert_allclose(
            evaluated, raw[0](x, v), rtol=1e-12, atol=1e-12
        )
