"""Provide function space variant 02 functionality for the modules layer.

Distinction: This file is variant 02, kept separate for controlled comparison; numerical logic is preserved.
"""

import jax.numpy as jnp
from jax import random
from collections.abc import Callable
from typing import Dict, Tuple
from flax import linen as nn
from jax.nn.initializers import uniform
from geometry.uniform_mesh import (
    UniformMeshXYV,
)
# V2 is the continuous overlapping-patch space.  Its differential operators
# use psi_b/dpsi_b, so evaluation must use the same partition of unity.
from modules.partition_of_unity import psi_b as psi
from functools import partial
import jax

seedX, seedXV = 1, 42

# input:(x,y,cos(theta),sin(theta))


class RandomFeatureSpaceXYV(nn.Module):
    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: int
    scale: float
    activation: Callable

    def setup(self):
        super().setup()
        self._dim = 3
        self._psi = psi
        self._mesh = UniformMeshXYV(domain=self.domain, strides=self.strides)
        self._center = lambda i: self._mesh.center_of_cell(i)[:2]  # (x_c, y_c)
        self._radius = self._mesh.radius_of_cell[:2]
        self._Mp = self._mesh.number_of_cells

        feats_list, params_list = [], []
        key = random.key(seedX)
        rng_keys = random.split(key, self._Mp)
        for i in range(self._Mp):
            feats = RandomFeatureFunctionsXYV(
                domain=self.domain,
                strides=self.strides,
                query_index=i,
                Jn=self.Jn,
                scale=self.scale,
                activation=self.activation,
            )
            dummy_x, dummy_y, dummy_v = jnp.split(
                jnp.empty((self._dim,)), self._dim, axis=0
            )
            # rng_key = random.key(i + self._Mp * (self._dim - 1))
            params = feats.init(rng_keys[i], dummy_x, dummy_y, dummy_v)
            feats_list.append(feats)
            params_list.append(params)

        self._feats2d_list = feats_list
        self._params_list = params_list

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_y: jnp.ndarray,
        invar_theta: jnp.ndarray,
        has_aux=False,
    ) -> jnp.ndarray:
        results = []
        psi_norm = 0.0
        for i, param, model in zip(
            range(self._Mp), self._params_list, self._feats2d_list
        ):
            center_x, center_y = self._center(i)
            r_x, r_y = self._radius

            psi = self._psi((invar_x - center_x) / r_x) * self._psi(
                (invar_y - center_y) / r_y
            )

            psi_norm += psi  # summation over psi_i
            outvar = model.apply(
                param,
                invar_x,
                invar_y,
                invar_theta,
            )

            outvar = psi * outvar
            results.append(outvar)

        outvar = jnp.array(results).squeeze()
        outvar = jnp.where(psi_norm > 0, outvar / psi_norm, 0.0)

        if has_aux:
            return outvar, self._get_operators()
        else:
            return outvar

    def _get_operators(self):
        feat_funcs = [
            partial(self._feats2d_list[i].apply, self._params_list[i])
            for i in range(self._Mp)
        ]
        return feat_funcs


class RandomFeatureFunctionsXYV(nn.Module):
    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Tuple[jnp.ndarray, jnp.ndarray]
    query_index: int
    Jn: int
    scale: float
    activation: Callable

    def setup(self):
        self._mesh = UniformMeshXYV(domain=self.domain, strides=self.strides)
        center = self._mesh.center_of_cell(self.query_index)
        center_xy = center[:2]
        radius = self._mesh.radius_of_cell
        radius_xy = radius[:2]
        self._theta_center = center[2]
        self._theta_radius = radius[2]
        theta_min, theta_max = self.domain["theta"]
        theta_stride = self.strides["theta"]
        num_theta_cells = int(round(float((theta_max - theta_min) / theta_stride)))
        self._theta_centers = theta_min + (
            jnp.arange(num_theta_cells) + 0.5
        ) * theta_stride

        self._center = jnp.array([center_xy[0], center_xy[1], 0.0, 0.0])
        self._radius = jnp.array([radius_xy[0], radius_xy[1], 1.0, 1.0])

        self._transform_fn = partial(
            lambda invar, center, radius: (invar - center) / radius,
            center=self._center,
            radius=self._radius,
        )

        self._feat_layer = nn.Dense(
            features=self.Jn,
            kernel_init=uniform(self.scale),
            bias_init=uniform(self.scale),
        )
        self._acti_fn = self.activation

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_y: jnp.ndarray,
        invar_theta: jnp.ndarray,
    ):
        cos_theta = jnp.cos(invar_theta)
        sin_theta = jnp.sin(invar_theta)

        invar_original = jnp.concatenate(
            [invar_x, invar_y, cos_theta, sin_theta], axis=-1
        )
        invar_normalized = self._transform_fn(
            invar_original, center=self._center, radius=self._radius
        )
        x_norm = invar_normalized[..., 0:1]
        y_norm = invar_normalized[..., 1:2]
        cos_theta = invar_normalized[..., 2:3]
        sin_theta = invar_normalized[..., 3:4]

        invar_new = jnp.concatenate(
            [x_norm, y_norm, cos_theta, sin_theta], axis=-1
        )
        out = self._feat_layer(invar_new)
        out = self._acti_fn(out)
        # Localize every V2 feature on the independent folded angular domain.
        # Antipodal and reflected directions share the same folded coordinate,
        # so this preserves the j/r parity construction while making the theta
        # entry of the mesh a genuine overlapping angular decomposition.
        folded_theta = jnp.arctan2(
            jnp.abs(jnp.sin(invar_theta)), jnp.abs(jnp.cos(invar_theta))
        )
        angular_window = psi(
            (folded_theta - self._theta_center) / self._theta_radius
        )
        angular_normalizer = jnp.sum(
            psi(
                (folded_theta - self._theta_centers)
                / self._theta_radius
            )
        )
        normalized_window = jnp.where(
            angular_normalizer > 0.0,
            angular_window / angular_normalizer,
            0.0,
        )
        return normalized_window * out
