"""Reconstruct 1D and 2D transport fields from fitted coefficients."""

from collections.abc import Callable
from functools import partial
from typing import Dict, Tuple

import jax.numpy as jnp
from jax import random

from oeaprfm.features.function_space import (
    FeatureSpaceXV,
    FeatureSpaceXYV,
)


class OEField1D(FeatureSpaceXV):
    """Odd/even decomposition for 1D RTE solution."""

    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: Dict[str, int]
    scale: float
    init_rng: random.PRNGKey
    kn: float
    activation: Callable
    coefficients: jnp.ndarray

    def setup(self):
        self._model_j = FeatureSpaceXV(
            self.domain,
            self.strides,
            self.Jn["j"],
            self.scale,
            self.activation,
        )
        self._model_r = FeatureSpaceXV(
            self.domain,
            self.strides,
            self.Jn["r"],
            self.scale,
            self.activation,
        )
        self._dummy_x = jnp.zeros((1,))
        self._dummy_v = jnp.zeros((1,))
        self._params_j = self._model_j.init(
            self.init_rng, self._dummy_x, self._dummy_v
        )
        self._params_r = self._model_r.init(
            self.init_rng, self._dummy_x, self._dummy_v
        )
        self._feats_fn_j = partial(self._model_j.apply, self._params_j)
        self._feats_fn_r = partial(self._model_r.apply, self._params_r)
        self._mesh_j = self._model_j._mesh
        self._mesh_r = self._model_r._mesh
        self._Mp_j = self._mesh_j.number_of_cells
        self._Mp_r = self._mesh_r.number_of_cells
        self._Jn_j = self.Jn["j"]
        self._Jn_r = self.Jn["r"]

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_v: jnp.ndarray,
    ) -> jnp.ndarray:
        feat_j_v = self._feats_fn_j(invar_x, invar_v)  # (Mp_j, Jn_j)
        feat_r_v = self._feats_fn_r(invar_x, invar_v)  # (Mp_r, Jn_r)

        feat_j_negv = self._feats_fn_j(invar_x, -invar_v)  # (Mp_j, Jn_j)
        feat_r_negv = self._feats_fn_r(invar_x, -invar_v)  # (Mp_r, Jn_r)

        feat_j_outvar = 0.5 * (feat_j_v - feat_j_negv)
        feat_r_outvar = 0.5 * (feat_r_v + feat_r_negv)

        coeff_j = self.coefficients[: self._Mp_j * self._Jn_j].reshape(
            self._Mp_j, self._Jn_j
        )
        coeff_r = self.coefficients[self._Mp_j * self._Jn_j :].reshape(
            self._Mp_r, self._Jn_r
        )

        approx_j = jnp.sum(feat_j_outvar * coeff_j)
        approx_r = jnp.sum(feat_r_outvar * coeff_r)
        approx_solution = approx_r + self.kn * approx_j
        return approx_solution


class HalfOEField1D(FeatureSpaceXV):
    """Unprojected positive-half-range features with even/odd reconstruction."""

    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: Dict[str, int]
    scale: float
    init_rng: random.PRNGKey
    kn: float
    activation: Callable
    coefficients: jnp.ndarray

    def setup(self):
        self._model_j = FeatureSpaceXV(
            self.domain,
            self.strides,
            self.Jn["j"],
            self.scale,
            self.activation,
        )
        self._model_r = FeatureSpaceXV(
            self.domain,
            self.strides,
            self.Jn["r"],
            self.scale,
            self.activation,
        )
        self._dummy_x = jnp.zeros((1,))
        self._dummy_v = jnp.zeros((1,))
        self._params_j = self._model_j.init(
            self.init_rng, self._dummy_x, self._dummy_v
        )
        self._params_r = self._model_r.init(
            self.init_rng, self._dummy_x, self._dummy_v
        )
        self._feats_fn_j = partial(self._model_j.apply, self._params_j)
        self._feats_fn_r = partial(self._model_r.apply, self._params_r)
        self._mesh_j = self._model_j._mesh
        self._mesh_r = self._model_r._mesh
        self._Mp_j = self._mesh_j.number_of_cells
        self._Mp_r = self._mesh_r.number_of_cells
        self._Jn_j = self.Jn["j"]
        self._Jn_r = self.Jn["r"]

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_v: jnp.ndarray,
    ) -> jnp.ndarray:
        feature_v = jnp.abs(invar_v)
        feat_j_outvar = self._feats_fn_j(invar_x, feature_v)
        feat_r_outvar = self._feats_fn_r(invar_x, feature_v)

        coeff_j = self.coefficients[: self._Mp_j * self._Jn_j].reshape(
            self._Mp_j, self._Jn_j
        )
        coeff_r = self.coefficients[self._Mp_j * self._Jn_j :].reshape(
            self._Mp_r, self._Jn_r
        )

        approx_j = jnp.sum(feat_j_outvar * coeff_j)
        approx_r = jnp.sum(feat_r_outvar * coeff_r)
        sign = jnp.sign(invar_v)
        approx_solution = approx_r + self.kn * approx_j * sign
        return approx_solution


class OEField2D(FeatureSpaceXYV):
    """Odd/even decomposition for 2D RTE solution."""

    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: Dict[str, int]
    scale: float
    init_rng: random.PRNGKey
    kn: float
    activation: Callable
    coefficients: jnp.ndarray

    def setup(self):
        self._model_j = FeatureSpaceXYV(
            self.domain,
            self.strides,
            self.Jn["j"],
            self.scale,
            self.activation,
        )
        self._model_r = FeatureSpaceXYV(
            self.domain,
            self.strides,
            self.Jn["r"],
            self.scale,
            self.activation,
        )
        self._dummy_x = jnp.zeros((1,))
        self._dummy_y = jnp.zeros((1,))
        self._dummy_theta = jnp.zeros((1,))
        self._params_j = self._model_j.init(
            self.init_rng, self._dummy_x, self._dummy_y, self._dummy_theta
        )
        self._params_r = self._model_r.init(
            self.init_rng, self._dummy_x, self._dummy_y, self._dummy_theta
        )
        self._feats_fn_j = partial(self._model_j.apply, self._params_j)
        self._feats_fn_r = partial(self._model_r.apply, self._params_r)
        self._mesh_j = self._model_j._mesh
        self._mesh_r = self._model_r._mesh
        self._Mp_j = self._mesh_j.number_of_cells
        self._Mp_r = self._mesh_r.number_of_cells
        self._Jn_j = self.Jn["j"]
        self._Jn_r = self.Jn["r"]

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_y: jnp.ndarray,
        invar_theta: jnp.ndarray,
    ) -> jnp.ndarray:
        feat_j_v = self._feats_fn_j(
            invar_x, invar_y, invar_theta
        )  # (Mp_j, Jn_j)
        feat_r_v = self._feats_fn_r(
            invar_x, invar_y, invar_theta
        )  # (Mp_r, Jn_r)

        paired_theta = (invar_theta + jnp.pi) % (2 * jnp.pi)
        feat_j_negv = self._feats_fn_j(
            invar_x, invar_y, paired_theta
        )  # (Mp_j, Jn_j)
        feat_r_negv = self._feats_fn_r(
            invar_x, invar_y, paired_theta
        )  # (Mp_r, Jn_r)

        feat_j_outvar = 0.5 * (feat_j_v - feat_j_negv)
        feat_r_outvar = 0.5 * (feat_r_v + feat_r_negv)

        coeff_j = self.coefficients[: self._Mp_j * self._Jn_j].reshape(
            self._Mp_j, self._Jn_j
        )
        coeff_r = self.coefficients[self._Mp_j * self._Jn_j :].reshape(
            self._Mp_r, self._Jn_r
        )

        approx_j = jnp.sum(feat_j_outvar * coeff_j)
        approx_r = jnp.sum(feat_r_outvar * coeff_r)
        approx_solution = self.kn * approx_j + approx_r
        return approx_solution
