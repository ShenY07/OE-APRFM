"""Provide dnn functionality for the modules layer.

Distinction: This file is the canonical implementation in this module family; numerical logic is preserved.
"""

import jax.numpy as jnp
from jax import random
from jax import vmap
from collections.abc import Callable
from typing import Dict, Tuple
from flax import linen as nn
from jax.nn.initializers import uniform

from functools import partial


class MLP(nn.Module):
    din: int
    dout: int
    hiddens: Tuple[int]
    activation: Callable

    def setup(self):
        super().setup()
        self._layer_in = nn.Dense(
            features=self.hiddens[0], kernel_init=uniform()
        )
        self._layers = [
            nn.Dense(features=h, kernel_init=uniform())
            for h in self.hiddens[1:]
        ]
        self._layer_out = nn.Dense(features=self.dout, kernel_init=uniform())
        self._acti_fn = self.activation

    @nn.compact
    def __call__(self, x, v):
        if len(x) == 1:
            xv = jnp.concatenate([x, v], axis=-1)
        if len(x) == 2:
            xi, eta = jnp.cos(v), jnp.sin(v)
            xv = jnp.concatenate([x, xi, eta], axis=-1)
        h = self._layer_in(xv)
        h = self._acti_fn(h)
        for layer in self._layers:
            h = layer(h)
            h = self._acti_fn(h)
        return self._layer_out(h)
