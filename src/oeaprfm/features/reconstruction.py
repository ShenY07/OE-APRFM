"""Reconstruct 1D and 2D transport fields from fitted coefficients."""

from collections.abc import Callable
from functools import partial
from typing import Dict, Tuple

import jax.numpy as jnp
from jax import random

from oeaprfm.features.function_space import (
    FeatureSpaceX,
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


class RTEField1D(FeatureSpaceXV):
    """Direct reconstruction without odd/even split."""

    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: Dict[str, int]
    scale: float
    init_rng: random.PRNGKey
    kn: float
    activation: Callable
    coefficients: jnp.ndarray

    def setup(self):
        self._model_f = FeatureSpaceXV(
            self.domain,
            self.strides,
            self.Jn["f"],
            self.scale,
            self.activation,
        )
        self._dummy_x = jnp.zeros((1,))
        self._dummy_v = jnp.zeros((1,))
        # 与 OEField1D 保持一致：init 需要 (x, v) 两个变量
        self._params_f = self._model_f.init(
            self.init_rng, self._dummy_x, self._dummy_v
        )
        self._feats_fn_f = partial(self._model_f.apply, self._params_f)
        self._mesh_f = self._model_f._mesh
        self._Mp_f = self._mesh_f.number_of_cells
        self._Jn_f = self.Jn["f"]

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_v: jnp.ndarray,
    ) -> jnp.ndarray:
        feat_f_outvar = self._feats_fn_f(invar_x, invar_v)
        coeff_f = self.coefficients[: self._Mp_f * self._Jn_f].reshape(
            self._Mp_f, self._Jn_f
        )
        approx_f = jnp.sum(feat_f_outvar * coeff_f)
        approx_solution = approx_f
        return approx_solution


class MaterialField1D(FeatureSpaceX):
    """Construct sigma_a(x) and sigma_s(x) via random features."""

    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: Dict[str, int]
    scale: float
    init_rng: random.PRNGKey
    activation: Callable
    coefficients: jnp.ndarray

    def setup(self):
        key1, key2 = random.split(self.init_rng)
        self._model_a = FeatureSpaceX(
            self.domain,
            self.strides,
            self.Jn["sigma_a"],
            self.scale,
            self.activation,
        )
        self._model_s = FeatureSpaceX(
            self.domain,
            self.strides,
            self.Jn["sigma_s"],
            self.scale,
            self.activation,
        )

        dummy_x = jnp.zeros((1,))
        self._params_a = self._model_a.init(key1, dummy_x)
        self._params_s = self._model_s.init(key2, dummy_x)

        self._feats_fn_a = partial(
            self._model_a.apply, self._params_a, has_aux=True
        )
        self._feats_fn_s = partial(
            self._model_s.apply, self._params_s, has_aux=True
        )

        self._mesh_a = self._model_a._mesh
        self._mesh_s = self._model_s._mesh
        self._Mp_a = self._mesh_a.number_of_cells
        self._Mp_s = self._mesh_s.number_of_cells
        self._Jn_a = self.Jn["sigma_a"]
        self._Jn_s = self.Jn["sigma_s"]

    def __call__(
        self, invar_x: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Evaluate sigma_a(x) and sigma_s(x) at given points invar_x.

        Args:
            invar_x: array of shape (batch,) or scalar

        Returns:
            Tuple of (sigma_a, sigma_s)
        """
        feat_a, _ = self._feats_fn_a(invar_x)
        feat_s, _ = self._feats_fn_s(invar_x)

        feat_a = feat_a.reshape(self._Mp_a, self._Jn_a)
        feat_s = feat_s.reshape(self._Mp_s, self._Jn_s)

        # Scattering coefficients precede absorption coefficients.
        n_dof_a = self._Mp_a * self._Jn_a
        n_dof_s = self._Mp_s * self._Jn_s

        coeff_s = self.coefficients[:n_dof_s].reshape(self._Mp_s, self._Jn_s)
        coeff_a = self.coefficients[n_dof_s:].reshape(self._Mp_a, self._Jn_a)

        sigma_a = jnp.sum(feat_a * coeff_a)
        sigma_s = jnp.sum(feat_s * coeff_s)

        return sigma_a, sigma_s


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


def inflow_value(
    x: jnp.ndarray, v: jnp.ndarray, bdy_value: Tuple[jnp.ndarray, jnp.ndarray]
) -> jnp.ndarray:
    """Inflow boundary condition for x = xl (v>0) or x = xr (v<0)."""

    inflow_value = jnp.where(
        v > 0, bdy_value[0], jnp.where(v < 0, bdy_value[-1], 0.0)
    )
    return inflow_value
