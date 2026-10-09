import jax.numpy as jnp
from jax.tree_util import tree_map
from typing import Tuple
from collections.abc import Callable


def tree_map_funcs(
    funcs: Callable,
    invars: jnp.ndarray
    | Tuple[jnp.ndarray, jnp.ndarray]
    | Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
):
    results = tree_map(lambda fn: fn(*invars), funcs)
    # print(results)
    values_list = (
        list(results.values()) if isinstance(results, dict) else results
    )
    # print(values_list)
    return values_list
