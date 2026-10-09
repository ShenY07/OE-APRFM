"""Provide solution construction variant 02 alternative functionality for the modules layer.

Distinction: This file is an explicitly retained alternative to the numbered variant; numerical logic is preserved.
"""

import jax.numpy as jnp
from jax import random
from collections.abc import Callable
from typing import Dict, Tuple
from modules.function_space_v2 import (
    RandomFeatureSpaceXYV,
)
from functools import partial


class OddEvenDecompositionConstructor2D(RandomFeatureSpaceXYV):
    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: int
    scale: float
    init_rng: random.PRNGKey
    kn: float
    activation: Callable
    coefficients: jnp.ndarray

    def setup(self):
        self._model_j1 = RandomFeatureSpaceXYV(
            self.domain,
            self.strides,
            self.Jn["j"],
            self.scale,
            self.activation,
        )
        self._model_r1 = RandomFeatureSpaceXYV(
            self.domain,
            self.strides,
            self.Jn["r"],
            self.scale,
            self.activation,
        )
        self._model_j2 = RandomFeatureSpaceXYV(
            self.domain,
            self.strides,
            self.Jn["j"],
            self.scale,
            self.activation,
        )
        self._model_r2 = RandomFeatureSpaceXYV(
            self.domain,
            self.strides,
            self.Jn["r"],
            self.scale,
            self.activation,
        )
        self._dim = 3
        self._dummy_x, self._dummy_y, self._dummy_theta = jnp.split(
            jnp.empty((self._dim,)), self._dim, axis=0
        )
        self._params_j1 = self._model_j1.init(
            self.init_rng, self._dummy_x, self._dummy_y, self._dummy_theta
        )
        self._params_r1 = self._model_r1.init(
            self.init_rng, self._dummy_x, self._dummy_y, self._dummy_theta
        )
        self._params_j2 = self._model_j2.init(
            self.init_rng, self._dummy_x, self._dummy_y, self._dummy_theta
        )
        self._params_r2 = self._model_r2.init(
            self.init_rng, self._dummy_x, self._dummy_y, self._dummy_theta
        )
        self._feats_fn_j1 = partial(self._model_j1.apply, self._params_j1)
        self._feats_fn_r1 = partial(self._model_r1.apply, self._params_r1)
        self._feats_fn_j2 = partial(self._model_j2.apply, self._params_j2)
        self._feats_fn_r2 = partial(self._model_r2.apply, self._params_r2)
        self._mesh_j1 = self._model_j1._mesh
        self._mesh_r1 = self._model_r1._mesh
        self._Mp_j1 = self._mesh_j1.number_of_cells
        self._Mp_r1 = self._mesh_r1.number_of_cells
        self._mesh_j2 = self._model_j2._mesh
        self._mesh_r2 = self._model_r2._mesh
        self._Mp_j2 = self._mesh_j2.number_of_cells
        self._Mp_r2 = self._mesh_r2.number_of_cells
        self._Jn_j1 = self.Jn["j"]
        self._Jn_r1 = self.Jn["r"]
        self._Jn_j2 = self.Jn["j"]
        self._Jn_r2 = self.Jn["r"]

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_y: jnp.ndarray,
        invar_theta: jnp.ndarray,
    ) -> jnp.ndarray:
        # invar_theta_2 = -invar_theta
        invar_theta_3 = invar_theta + jnp.pi
        # invar_theta_4 = -invar_theta + jnp.pi

        feat_j1_v = self._feats_fn_j1(
            invar_x, invar_y, invar_theta
        )  # (Mp_j, Jn_j)
        feat_r1_v = self._feats_fn_r1(
            invar_x, invar_y, invar_theta
        )  # (Mp_r, Jn_r)
        feat_j1_negv = self._feats_fn_j1(
            invar_x, invar_y, invar_theta_3
        )  # (Mp_j, Jn_j)
        feat_r1_negv = self._feats_fn_r1(
            invar_x, invar_y, invar_theta_3
        )  # (Mp_r, Jn_r)

        # feat_j2_v = self._feats_fn_j2(
        #     invar_x, invar_y, invar_theta_2)  # (Mp_j, Jn_j)
        # feat_r2_v = self._feats_fn_r2(
        #     invar_x, invar_y, invar_theta_2)  # (Mp_r, Jn_r)
        # feat_j2_negv = self._feats_fn_j2(
        #     invar_x, invar_y, invar_theta_4
        # )  # (Mp_j, Jn_j)
        # feat_r2_negv = self._feats_fn_r2(
        #     invar_x, invar_y, invar_theta_4
        # )  # (Mp_r, Jn_r)

        feat_j1_outvar = 0.5 * (feat_j1_v - feat_j1_negv)  # j_new
        feat_r1_outvar = 0.5 * (feat_r1_v + feat_r1_negv)  # r_new
        # feat_j2_outvar = 0.5 * (feat_j2_v - feat_j2_negv)  # j_new
        # feat_r2_outvar = 0.5 * (feat_r2_v + feat_r2_negv)  # r_new

        coeff_j1 = self.coefficients[: self._Mp_j1 * self._Jn_j1].reshape(
            self._Mp_j1, self._Jn_j1
        )
        coeff_r1 = self.coefficients[
            self._Mp_j1 * self._Jn_j1 : self._Mp_j1 * self._Jn_j1
            + self._Mp_r1 * self._Jn_r1
        ].reshape(self._Mp_r1, self._Jn_r1)
        # coeff_j2 = self.coefficients[
        #     self._Mp_j1 * self._Jn_j1 + self._Mp_r1 * self._Jn_r1: self._Mp_j1
        #     * self._Jn_j1
        #     + self._Mp_r1 * self._Jn_r1
        #     + self._Mp_j2 * self._Jn_j2
        # ].reshape(self._Mp_j2, self._Jn_j2)
        # coeff_r2 = self.coefficients[
        #     self._Mp_j1 * self._Jn_j1
        #     + self._Mp_r1 * self._Jn_r1
        #     + self._Mp_j2 * self._Jn_j2:
        # ].reshape(self._Mp_r2, self._Jn_r2)

        approx_j1 = jnp.einsum(
            "ij,ij->",
            (
                feat_j1_outvar
                if self._Mp_j1 > 1
                else feat_j1_outvar.reshape(1, self._Jn_j1)
            ),
            coeff_j1,
        )
        approx_r1 = jnp.einsum(
            "ij,ij->",
            (
                feat_r1_outvar
                if self._Mp_r1 > 1
                else feat_r1_outvar.reshape(1, self._Jn_r1)
            ),
            coeff_r1,
        )
        # approx_j2 = jnp.einsum(
        #     "ij,ij->",
        #     feat_j2_outvar
        #     if self._Mp_j2 > 1
        #     else feat_j2_outvar.reshape(1, self._Jn_j2),
        #     coeff_j2,
        # )
        # approx_r2 = jnp.einsum(
        #     "ij,ij->",
        #     feat_r2_outvar
        #     if self._Mp_r2 > 1
        #     else feat_r2_outvar.reshape(1, self._Jn_r2),
        #     coeff_r2,
        # )

        approx_solution = self.kn * approx_j1 + approx_r1
        return approx_solution


class OddEvenDecompositionAverage2D(RandomFeatureSpaceXYV):
    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: int
    scale: float
    init_rng: random.PRNGKey
    kn: float
    activation: Callable
    coefficients: jnp.ndarray

    def setup(self):
        self._model_j1 = RandomFeatureSpaceXYV(
            self.domain,
            self.strides,
            self.Jn["j"],
            self.scale,
            self.activation,
        )
        self._model_r1 = RandomFeatureSpaceXYV(
            self.domain,
            self.strides,
            self.Jn["r"],
            self.scale,
            self.activation,
        )
        self._model_j2 = RandomFeatureSpaceXYV(
            self.domain,
            self.strides,
            self.Jn["j"],
            self.scale,
            self.activation,
        )
        self._model_r2 = RandomFeatureSpaceXYV(
            self.domain,
            self.strides,
            self.Jn["r"],
            self.scale,
            self.activation,
        )
        self._dim = 3
        self._dummy_x, self._dummy_y, self._dummy_theta = jnp.split(
            jnp.empty((self._dim,)), self._dim, axis=0
        )
        self._params_j1 = self._model_j1.init(
            self.init_rng, self._dummy_x, self._dummy_y, self._dummy_theta
        )
        self._params_r1 = self._model_r1.init(
            self.init_rng, self._dummy_x, self._dummy_y, self._dummy_theta
        )
        self._params_j2 = self._model_j2.init(
            self.init_rng, self._dummy_x, self._dummy_y, self._dummy_theta
        )
        self._params_r2 = self._model_r2.init(
            self.init_rng, self._dummy_x, self._dummy_y, self._dummy_theta
        )
        self._feats_fn_j1 = partial(self._model_j1.apply, self._params_j1)
        self._feats_fn_r1 = partial(self._model_r1.apply, self._params_r1)
        self._feats_fn_j2 = partial(self._model_j2.apply, self._params_j2)
        self._feats_fn_r2 = partial(self._model_r2.apply, self._params_r2)
        self._mesh_j1 = self._model_j1._mesh
        self._mesh_r1 = self._model_r1._mesh
        self._Mp_j1 = self._mesh_j1.number_of_cells
        self._Mp_r1 = self._mesh_r1.number_of_cells
        self._mesh_j2 = self._model_j2._mesh
        self._mesh_r2 = self._model_r2._mesh
        self._Mp_j2 = self._mesh_j2.number_of_cells
        self._Mp_r2 = self._mesh_r2.number_of_cells
        self._Jn_j1 = self.Jn["j"]
        self._Jn_r1 = self.Jn["r"]
        self._Jn_j2 = self.Jn["j"]
        self._Jn_r2 = self.Jn["r"]

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_y: jnp.ndarray,
        invar_theta: jnp.ndarray,
    ) -> jnp.ndarray:
        invar_theta_2 = -invar_theta
        invar_theta_3 = invar_theta + jnp.pi
        invar_theta_4 = -invar_theta + jnp.pi

        feat_j1_v = self._feats_fn_j1(
            invar_x, invar_y, invar_theta
        )  # (Mp_j, Jn_j)
        feat_r1_v = self._feats_fn_r1(
            invar_x, invar_y, invar_theta
        )  # (Mp_r, Jn_r)
        feat_j1_negv = self._feats_fn_j1(
            invar_x, invar_y, invar_theta_3
        )  # (Mp_j, Jn_j)
        feat_r1_negv = self._feats_fn_r1(
            invar_x, invar_y, invar_theta_3
        )  # (Mp_r, Jn_r)

        feat_j2_v = self._feats_fn_j2(
            invar_x, invar_y, invar_theta_2
        )  # (Mp_j, Jn_j)
        feat_r2_v = self._feats_fn_r2(
            invar_x, invar_y, invar_theta_2
        )  # (Mp_r, Jn_r)
        feat_j2_negv = self._feats_fn_j2(
            invar_x, invar_y, invar_theta_4
        )  # (Mp_j, Jn_j)
        feat_r2_negv = self._feats_fn_r2(
            invar_x, invar_y, invar_theta_4
        )  # (Mp_r, Jn_r)

        feat_j1_outvar = 0.5 * (feat_j1_v - feat_j1_negv)  # j_new
        feat_r1_outvar = 0.5 * (feat_r1_v + feat_r1_negv)  # r_new
        feat_j2_outvar = 0.5 * (feat_j2_v - feat_j2_negv)  # j_new
        feat_r2_outvar = 0.5 * (feat_r2_v + feat_r2_negv)  # r_new

        coeff_j1 = self.coefficients[: self._Mp_j1 * self._Jn_j1].reshape(
            self._Mp_j1, self._Jn_j1
        )
        coeff_r1 = self.coefficients[
            self._Mp_j1 * self._Jn_j1 : self._Mp_j1 * self._Jn_j1
            + self._Mp_r1 * self._Jn_r1
        ].reshape(self._Mp_r1, self._Jn_r1)
        coeff_j2 = self.coefficients[
            self._Mp_j1 * self._Jn_j1
            + self._Mp_r1 * self._Jn_r1 : self._Mp_j1 * self._Jn_j1
            + self._Mp_r1 * self._Jn_r1
            + self._Mp_j2 * self._Jn_j2
        ].reshape(self._Mp_j2, self._Jn_j2)
        coeff_r2 = self.coefficients[
            self._Mp_j1 * self._Jn_j1
            + self._Mp_r1 * self._Jn_r1
            + self._Mp_j2 * self._Jn_j2 :
        ].reshape(self._Mp_r2, self._Jn_r2)

        approx_j1 = jnp.einsum(
            "ij,ij->",
            (
                feat_j1_outvar
                if self._Mp_j1 > 1
                else feat_j1_outvar.reshape(1, self._Jn_j1)
            ),
            coeff_j1,
        )
        approx_r1 = jnp.einsum(
            "ij,ij->",
            (
                feat_r1_outvar
                if self._Mp_r1 > 1
                else feat_r1_outvar.reshape(1, self._Jn_r1)
            ),
            coeff_r1,
        )
        approx_j2 = jnp.einsum(
            "ij,ij->",
            (
                feat_j2_outvar
                if self._Mp_j2 > 1
                else feat_j2_outvar.reshape(1, self._Jn_j2)
            ),
            coeff_j2,
        )
        approx_r2 = jnp.einsum(
            "ij,ij->",
            (
                feat_r2_outvar
                if self._Mp_r2 > 1
                else feat_r2_outvar.reshape(1, self._Jn_r2)
            ),
            coeff_r2,
        )

        approx_solution = (approx_r1 + approx_r2) / 2.0
        return approx_solution
