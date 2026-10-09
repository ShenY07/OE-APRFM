"""Batch evaluation helpers for the odd-even transport runners."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np


def evaluate_padded_batch(
    evaluator: Callable, points: tuple, batch_size: int
) -> np.ndarray:
    """Pad with a valid point and discard padding before assembling any rows.

    The evaluator is pointwise over the leading axis. Fixed input shapes avoid
    compiling a second operator for the final, incomplete batch.
    """
    count = points[0].shape[0]
    if not 0 < count <= batch_size:
        raise ValueError(
            "batch must contain between one and batch_size points"
        )
    if any(point.shape[0] != count for point in points):
        raise ValueError("all point arrays must have the same batch length")
    if count < batch_size:
        points = tuple(
            jnp.concatenate(
                (point, jnp.repeat(point[-1:], batch_size - count, axis=0)),
                axis=0,
            )
            for point in points
        )
    return np.asarray(evaluator(*points))[:count]


def make_source_evaluator(source, epsilon, num_quads, use_v2):
    """Compile the existing OE source equations and angular average together."""
    source_batch = jax.vmap(source)
    angles = jnp.asarray(
        np.linspace(0.0, 2.0 * np.pi, num_quads, endpoint=False)
    )

    def evaluate(x, y, theta):
        qp = source_batch(x, y, theta).reshape(-1)
        qm = source_batch(x, y, (theta + jnp.pi) % (2 * jnp.pi)).reshape(-1)
        even1, odd1 = 0.5 * (qp + qm), 0.5 * (qp - qm)

        def add_angle(average, angle):
            value = source_batch(x, y, jnp.full_like(theta, angle)).reshape(-1)
            return average + value / num_quads, None

        average, _ = jax.lax.scan(add_angle, jnp.zeros_like(even1), angles)
        if use_v2:
            q2p = source_batch(x, y, -theta).reshape(-1)
            q2m = source_batch(x, y, jnp.pi - theta).reshape(-1)
            even2, odd2 = 0.5 * (q2p + q2m), 0.5 * (q2p - q2m)
            return jnp.column_stack(
                (
                    2 * average,
                    epsilon**2 * ((even1 - average) + (even2 - average)),
                    epsilon * (odd1 + odd2),
                    epsilon**2 * (even1 - even2),
                    epsilon * (odd1 - odd2),
                )
            )
        return jnp.column_stack(
            (average, epsilon**2 * (even1 - average), epsilon * odd1)
        )

    return jax.jit(evaluate)
