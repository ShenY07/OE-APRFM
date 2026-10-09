"""Check acceleration preserves OE equations, symmetry, and real batch rows."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import constraints.continuous_2d_odd_even_v2 as oe
from modules.oe_assembly import evaluate_padded_batch, make_source_evaluator

jax.config.update("jax_enable_x64", True)


def test_padding_reuses_shape_and_discards_fictitious_rows():
    traced_shapes = []

    @jax.jit
    def evaluate(x, y, theta):
        traced_shapes.append(x.shape)
        return jnp.stack((x + y, theta**2), axis=1)

    full = tuple(jnp.arange(4.0).reshape(-1, 1) + i for i in range(3))
    tail = tuple(point[:1] for point in full)
    a = evaluate_padded_batch(evaluate, full, 4)
    b = evaluate_padded_batch(evaluate, tail, 4)
    np.testing.assert_array_equal(b, a[:1])
    assert b.shape[0] == 1
    assert traced_shapes == [(4, 1)]


@pytest.mark.parametrize("use_v2", [False, True])
@pytest.mark.parametrize("epsilon", [1.0, 1e-3])
def test_compiled_rhs_preserves_even_odd_and_nonzero_angular_mean(
    use_v2, epsilon
):
    def source(x, y, theta):
        return 1 + x * y + jnp.cos(theta) * x + jnp.cos(2 * theta) * y

    x = jnp.array([[-0.7], [0.1], [0.8]])
    y = jnp.array([[0.2], [-0.4], [0.6]])
    theta = jnp.array([[0.03], [0.7], [1.5]])
    value = jax.vmap(source)
    qp = np.asarray(value(x, y, theta)).ravel()
    qm = np.asarray(value(x, y, (theta + jnp.pi) % (2 * jnp.pi))).ravel()
    even, odd = 0.5 * (qp + qm), 0.5 * (qp - qm)
    average = np.zeros_like(even)
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        average += (
            np.asarray(value(x, y, jnp.full_like(theta, angle))).ravel() / 8
        )
    if use_v2:
        q2p = np.asarray(value(x, y, -theta)).ravel()
        q2m = np.asarray(value(x, y, jnp.pi - theta)).ravel()
        even2, odd2 = 0.5 * (q2p + q2m), 0.5 * (q2p - q2m)
        expected = np.column_stack(
            (
                2 * average,
                epsilon**2 * (even + even2 - 2 * average),
                epsilon * (odd + odd2),
                epsilon**2 * (even - even2),
                epsilon * (odd - odd2),
            )
        )
    else:
        expected = np.column_stack(
            (average, epsilon**2 * (even - average), epsilon * odd)
        )
    evaluator = make_source_evaluator(source, epsilon, 8, use_v2)
    actual = evaluate_padded_batch(evaluator, (x, y, theta), 4)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def make_constraints(partitions, epsilon):
    domain = {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "theta": (0.0, np.pi / 2)}
    strides = {
        axis: (domain[axis][1] - domain[axis][0]) / count
        for axis, count in zip(("x", "y", "theta"), partitions)
    }
    common = dict(
        domain=domain,
        strides=strides,
        Jn={"j": 4, "r": 4},
        scale=0.7,
        init_rng=jax.random.key(11),
        kn=epsilon,
        activation=jnp.tanh,
    )
    interior = oe.OddEvenDecompositionPointwiseInteriorConstraint2D(
        **common,
        num_quads=4,
        coeff_fns={
            "scattering": lambda x, y: 1 + 0.1 * y,
            "absorption": lambda x, y: 0.1,
        },
    )
    boundary = oe.OddEvenDecompositionPointwiseBoundaryConstraint2D(**common)
    return interior, boundary


@pytest.mark.parametrize("partitions", [(1, 1, 1), (2, 1, 2)])
@pytest.mark.parametrize("epsilon", [1.0, 1e-3])
def test_forward_derivatives_match_reverse_quadrant_operator(
    partitions, epsilon, monkeypatch
):
    interior, _ = make_constraints(partitions, epsilon)
    points = (
        jnp.array([[-0.6], [0.2]]),
        jnp.array([[0.3], [-0.4]]),
        jnp.array([[0.2], [1.3]]),
    )
    forward = jax.jit(jax.vmap(lambda x, y, t: interior.apply({}, x, y, t)))
    actual = np.asarray(forward(*points))
    # Independent AD direction checks all five residuals and every feature.
    with monkeypatch.context() as context:
        context.setattr(oe, "jacfwd", jax.jacrev)
        reverse = jax.jit(
            jax.vmap(lambda x, y, t: interior.apply({}, x, y, t))
        )
        expected = np.asarray(reverse(*points))
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)


def test_empty_external_variables_match_initialization_and_all_boundary_quadrants():
    interior, boundary = make_constraints((1, 1, 1), 1e-3)
    point = (jnp.array([0.2]), jnp.array([-0.4]), jnp.array([0.3]))
    for module, method in ((interior, None), (boundary, boundary.trace)):
        variables = module.init(jax.random.key(11), *point, method=method)
        assert not variables
        for angle in (0.3, 1.9, 3.5, 5.1) if method else (0.3,):
            args = (point[0], point[1], jnp.array([angle]))
            np.testing.assert_array_equal(
                module.apply({}, *args, method=method),
                module.apply(variables, *args, method=method),
            )
