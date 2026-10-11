"""Integration operator."""

from collections.abc import Callable
from typing import Tuple

import jax.numpy as jnp
from jax import vmap


def quad(
    fun: Callable,
    quadratures: Tuple[jnp.ndarray, jnp.ndarray],
    argnum: int = 0,
) -> Callable:
    """Compute the integral operator for a scalar function using
    quadratures.
    """
    points, weights = quadratures

    def integral_fn(*args):
        args = list(args)
        in_axes_ = [None] * len(args)
        args.insert(argnum, points)
        in_axes_.insert(argnum, int(0))
        out = vmap(fun, in_axes=in_axes_, out_axes=-1)(*args)
        return jnp.dot(out, weights)

    return integral_fn


def leggauss(num: int, interval: Tuple[float] = (-1.0, 1.0)):
    """Compute the Gauss-Legendre quadrature points
    and weights for the given interval."""
    from scipy.special import roots_legendre

    points, weights = roots_legendre(num)
    # import numpy as np
    # points, weights = np.polynomial.legendre.leggauss(num)
    points = 0.5 * (1.0 + points) * (interval[1] - interval[0]) + interval[0]
    weights = 0.5 * weights * (interval[1] - interval[0])
    return jnp.array(points)[:, None], jnp.array(weights)[:, None]
