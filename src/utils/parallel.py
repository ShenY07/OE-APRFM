"""Provide parallel functionality for the utils layer.

Distinction: This file is the canonical implementation in this module family; numerical logic is preserved.
"""

from collections.abc import Callable
from typing import Any, Sequence

from jax.tree_util import tree_map


def tree_map_funcs(funcs: Callable, invars: Sequence[Any] | Any):
    """Apply each function in `funcs` to the provided invars.

    - `funcs` can be a dict or any pytree of callables; if dict, the return
      value preserves the ordering of values.
    - `invars` may be a single value or a sequence/tuple of values; these
      will be expanded when calling each function (fn(*invars)).

    Returns a Python list of results (or the original results if not a dict).
    """
    # Ensure invars is a tuple/list so we can splat it into the call
    if not isinstance(invars, (tuple, list)):
        invars = (invars,)

    results = tree_map(lambda fn: fn(*invars), funcs)
    values_list = (
        list(results.values()) if isinstance(results, dict) else results
    )
    return values_list
