# -*- coding: utf-8 -*-
import jax.numpy as jnp
from jax import random, vmap, jacrev
from utils.integrate import leggauss
from utils.parallel import tree_map_funcs
from collections.abc import Callable
from typing import Dict, Tuple
from modules.pou_func import psi_b as psi, dpsi_b as dpsi
from modules.func_space import RandomFeatureSpaceXV
from functools import partial


class OddEvenDecompositionPointwiseInteriorConstraint1D(RandomFeatureSpaceXV):
    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: int
    scale: float
    init_rng: random.PRNGKey
    num_quads: int
    kn: float
    activation: Callable
    coeff_fns: Dict[str, Callable]

    def setup(self):
        # Quadrature points and weights
        self._pts, self._ws = leggauss(self.num_quads, interval=[0, 1])

        # Initialize models
        dummy_x, dummy_v = jnp.split(jnp.empty((2,)), 2, axis=0)

        self._model_j = RandomFeatureSpaceXV(
            self.domain, self.strides, self.Jn["j"], self.scale, self.activation
        )
        self._model_r = RandomFeatureSpaceXV(
            self.domain, self.strides, self.Jn["r"], self.scale, self.activation
        )

        params_j = self._model_j.init(self.init_rng, dummy_x, dummy_v)
        params_r = self._model_r.init(self.init_rng, dummy_x, dummy_v)

        self._feats_fn_j = partial(self._model_j.apply, params_j, has_aux=True)
        self._feats_fn_r = partial(self._model_r.apply, params_r, has_aux=True)

        # Cache mesh properties
        mesh_j = self._model_j._mesh
        mesh_r = self._model_r._mesh

        self._center_j = mesh_j.center_of_cell
        self._center_r = mesh_r.center_of_cell
        self._radius_j = mesh_j.radius_of_cell
        self._radius_r = mesh_r.radius_of_cell
        self._Mp_j = mesh_j.number_of_cells
        self._Mp_r = mesh_r.number_of_cells
        self._Jn_j = self.Jn["j"]
        self._Jn_r = self.Jn["r"]

        # Coefficient functions
        self._sigma_s = self.coeff_fns["scattering"]
        self._sigma_a = self.coeff_fns["absorption"]

        # Partition of unity functions
        self._psi_j, self._dpsi_j = self.normalize_j_pou_fn()
        self._psi_r, self._dpsi_r = self.normalize_r_pou_fn()

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_v: jnp.ndarray,
    ) -> jnp.ndarray:
        x, v = invar_x, invar_v
        _, feats_j = self._feats_fn_j(x, v)
        _, feats_r = self._feats_fn_r(x, v)

        funcs_j = {
            f"j_{i}": lambda x, v, i=i: (
                0.5
                * (
                    self._psi_j[i](x, v) * feats_j[i](x, v)
                    - self._psi_j[i](x, -v) * feats_j[i](x, -v)
                )
            )
            for i in range(self._Mp_j)
        }

        funcs_r = {
            f"r_{i}": lambda x, v, i=i: (
                0.5
                * (
                    self._psi_r[i](x, v) * feats_r[i](x, v)
                    + self._psi_r[i](x, -v) * feats_r[i](x, -v)
                )
            )
            for i in range(self._Mp_r)
        }

        funcs_dj_dx = {
            f"dj_dx_{i}": lambda x, v, i=i: (
                0.5
                * (
                    (
                        self._dpsi_j[i](x, v) * feats_j[i](x, v)
                        + self._psi_j[i](x, v) * jacrev(feats_j[i], 0)(x, v).squeeze()
                    )
                    - (
                        self._dpsi_j[i](x, -v) * feats_j[i](x, -v)
                        + self._psi_j[i](x, -v) * jacrev(feats_j[i], 0)(x, -v).squeeze()
                    )
                )
            )
            for i in range(self._Mp_j)
        }

        funcs_vdr_dx = {
            f"dr_dx_{i}": lambda x, v, i=i: (
                0.5
                * v
                * (
                    (
                        self._dpsi_r[i](x, v) * feats_r[i](x, v)
                        + self._psi_r[i](x, v) * jacrev(feats_r[i], 0)(x, v).squeeze()
                    )
                    + (
                        self._dpsi_r[i](x, -v) * feats_r[i](x, -v)
                        + self._psi_r[i](x, -v) * jacrev(feats_r[i], 0)(x, -v).squeeze()
                    )
                )
            )
            for i in range(self._Mp_r)
        }

        funcs_v_dj = {
            f"v_dj_{i}": lambda x, v, i=i: (
                0.5
                * v
                * (
                    (
                        self._dpsi_j[i](x, v) * feats_j[i](x, v)
                        + self._psi_j[i](x, v) * jacrev(feats_j[i], 0)(x, v).squeeze()
                    )
                    - (
                        self._dpsi_j[i](x, -v) * feats_j[i](x, -v)
                        + self._psi_j[i](x, -v) * jacrev(feats_j[i], 0)(x, -v).squeeze()
                    )
                )
            )
            for i in range(self._Mp_j)
        }

        # Compute vectors
        vec_j = jnp.array(tree_map_funcs(funcs_j, (x, v))).flatten()
        vec_r = jnp.array(tree_map_funcs(funcs_r, (x, v))).flatten()
        vec_dj_dx = jnp.array(tree_map_funcs(funcs_dj_dx, (x, v))).flatten()
        vec_vdr_dx = jnp.array(tree_map_funcs(funcs_vdr_dx, (x, v))).flatten()

        # Integrate with respect to v
        def integrate_v(fn_dict):
            return (
                jnp
                .array([
                    self._integrator(fn, (self._pts, self._ws), 1)(x)
                    for fn in fn_dict.values()
                ])
                .squeeze()
                .flatten()
            )

        vec_avg_vdj_dx = integrate_v(funcs_v_dj)
        vec_avg_r = integrate_v(funcs_r)

        # Assemble residual matrix
        n_dof = self._Mp_j * self._Jn_j + self._Mp_r * self._Jn_r
        n_j = self._Mp_j * self._Jn_j

        residual = jnp.zeros((3, n_dof))
        sigma_s_val = self._sigma_s(x)
        sigma_a_val = self._sigma_a(x)
        kn2 = self.kn**2
        coeff = sigma_s_val + kn2 * sigma_a_val

        residual = residual.at[0, :n_j].set(vec_avg_vdj_dx)
        residual = residual.at[0, n_j:].set(sigma_a_val * vec_avg_r)
        residual = residual.at[1, :n_j].set(kn2 * (v * vec_dj_dx - vec_avg_vdj_dx))
        residual = residual.at[1, n_j:].set(coeff * (vec_r - vec_avg_r))
        residual = residual.at[2, :n_j].set(coeff * vec_j)
        residual = residual.at[2, n_j:].set(vec_vdr_dx)

        return residual

    def _integrator(
        self,
        fn: Callable,
        quadratures: Tuple[jnp.ndarray, jnp.ndarray],
        argnum: int,
    ) -> Callable:
        """Integrate function with respect to specified argument using Gauss quadrature."""
        pts, ws = quadratures

        def integral_fn(*args):
            args_list = list(args)
            in_axes = [None] * len(args_list)
            args_list.insert(argnum, pts)
            in_axes.insert(argnum, 0)

            vmap_result = vmap(fn, in_axes=in_axes, out_axes=-1)(*args_list)
            return jnp.array([jnp.dot(vmap_result[i], ws) for i in range(self._Jn_r)])

        return integral_fn

    def normalize_j_pou_fn(self):
        """Construct normalized partition of unity functions for j model."""

        def psi_sym(x, v, i):
            # center_v = self._center_j(i)[1]
            center_v = jnp.where(v >= 0, self._center_j(i)[-1], -self._center_j(i)[-1])
            v_rel = v - center_v
            x_norm = (x - self._center_j(i)[0]) / self._radius_j[0]
            v_norm = v_rel / self._radius_j[-1]
            return psi(x_norm) * psi(v_norm)

        def dpsi_sym(x, v, i):

            center_v = jnp.where(v >= 0, self._center_j(i)[-1], -self._center_j(i)[-1])
            v_rel = v - center_v
            x_norm = (x - self._center_j(i)[0]) / self._radius_j[0]
            v_norm = v_rel / self._radius_j[-1]
            return dpsi(x_norm) / self._radius_j[0] * psi(v_norm)

        def psi_sum(x, v):
            return jnp.sum(jnp.array([psi_sym(x, v, i) for i in range(self._Mp_j)]))

        def dpsi_sum(x, v):
            return jnp.sum(jnp.array([dpsi_sym(x, v, i) for i in range(self._Mp_j)]))

        psi_list = [
            lambda x, v, i=i: psi_sym(x, v, i) / psi_sum(x, v)
            for i in range(self._Mp_j)
        ]
        dpsi_list = [
            lambda x, v, i=i: (
                (dpsi_sym(x, v, i) * psi_sum(x, v) - dpsi_sum(x, v) * psi_sym(x, v, i))
                / psi_sum(x, v) ** 2
            )
            for i in range(self._Mp_j)
        ]

        return psi_list, dpsi_list

    def normalize_r_pou_fn(self):
        """Construct normalized partition of unity functions for r model."""

        def psi_sym(x, v, i):

            center_v = jnp.where(v >= 0, self._center_r(i)[-1], -self._center_r(i)[-1])
            v_rel = v - center_v
            x_norm = (x - self._center_r(i)[0]) / self._radius_r[0]
            v_norm = v_rel / self._radius_r[-1]
            return psi(x_norm) * psi(v_norm)

        def dpsi_sym(x, v, i):
            center_v = jnp.where(v >= 0, self._center_r(i)[-1], -self._center_r(i)[-1])
            v_rel = v - center_v
            x_norm = (x - self._center_r(i)[0]) / self._radius_r[0]
            v_norm = v_rel / self._radius_r[-1]
            return dpsi(x_norm) / self._radius_r[0] * psi(v_norm)

        def psi_sum(x, v):
            return jnp.sum(jnp.array([psi_sym(x, v, i) for i in range(self._Mp_r)]))

        def dpsi_sum(x, v):
            return jnp.sum(jnp.array([dpsi_sym(x, v, i) for i in range(self._Mp_r)]))

        psi_list = [
            lambda x, v, i=i: psi_sym(x, v, i) / psi_sum(x, v)
            for i in range(self._Mp_r)
        ]
        dpsi_list = [
            lambda x, v, i=i: (
                (dpsi_sym(x, v, i) * psi_sum(x, v) - dpsi_sum(x, v) * psi_sym(x, v, i))
                / psi_sum(x, v) ** 2
            )
            for i in range(self._Mp_r)
        ]

        return psi_list, dpsi_list


class OddEvenPointwiseInteriorConstraint1D(RandomFeatureSpaceXV):
    """
    Without the embedding of even and odd functions into the model architecture, this constraint directly computes the odd-even decomposition of the features and their derivatives, and integrates with respect to v using quadrature. The residual is then assembled based on the odd-even decomposition and the integrated values.
    """

    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: int
    scale: float
    init_rng: random.PRNGKey
    num_quads: int
    kn: float
    activation: Callable
    coeff_fns: Dict[str, Callable]

    def setup(self):
        # Quadrature points and weights
        self._pts, self._ws = leggauss(self.num_quads, interval=[0, 1])

        # Initialize models
        dummy_x, dummy_v = jnp.split(jnp.empty((2,)), 2, axis=0)

        self._model_j = RandomFeatureSpaceXV(
            self.domain, self.strides, self.Jn["j"], self.scale, self.activation
        )
        self._model_r = RandomFeatureSpaceXV(
            self.domain, self.strides, self.Jn["r"], self.scale, self.activation
        )

        params_j = self._model_j.init(self.init_rng, dummy_x, dummy_v)
        params_r = self._model_r.init(self.init_rng, dummy_x, dummy_v)

        self._feats_fn_j = partial(self._model_j.apply, params_j, has_aux=True)
        self._feats_fn_r = partial(self._model_r.apply, params_r, has_aux=True)

        # Cache mesh properties
        mesh_j = self._model_j._mesh
        mesh_r = self._model_r._mesh

        self._center_j = mesh_j.center_of_cell
        self._center_r = mesh_r.center_of_cell
        self._radius_j = mesh_j.radius_of_cell
        self._radius_r = mesh_r.radius_of_cell
        self._Mp_j = mesh_j.number_of_cells
        self._Mp_r = mesh_r.number_of_cells
        self._Jn_j = self.Jn["j"]
        self._Jn_r = self.Jn["r"]

        # Coefficient functions
        self._sigma_s = self.coeff_fns["scattering"]
        self._sigma_a = self.coeff_fns["absorption"]

        # Partition of unity functions
        self._psi_j, self._dpsi_j = self.normalize_j_pou_fn()
        self._psi_r, self._dpsi_r = self.normalize_r_pou_fn()

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_v: jnp.ndarray,
    ) -> jnp.ndarray:
        x, v = invar_x, invar_v
        _, feats_j = self._feats_fn_j(x, v)
        _, feats_r = self._feats_fn_r(x, v)

        funcs_j = {
            f"j_{i}": lambda x, v, i=i: self._psi_j[i](x, v) * feats_j[i](x, v)
            for i in range(self._Mp_j)
        }

        funcs_r = {
            f"r_{i}": lambda x, v, i=i: self._psi_r[i](x, v) * feats_r[i](x, v)
            for i in range(self._Mp_r)
        }

        funcs_dj_dx = {
            f"dj_dx_{i}": lambda x, v, i=i: (
                self._dpsi_j[i](x, v) * feats_j[i](x, v)
                + self._psi_j[i](x, v) * jacrev(feats_j[i], 0)(x, v).squeeze()
            )
            for i in range(self._Mp_j)
        }

        funcs_vdr_dx = {
            f"dr_dx_{i}": lambda x, v, i=i: (
                v
                * (
                    self._dpsi_r[i](x, v) * feats_r[i](x, v)
                    + self._psi_r[i](x, v) * jacrev(feats_r[i], 0)(x, v).squeeze()
                )
            )
            for i in range(self._Mp_r)
        }

        funcs_v_dj = {
            f"v_dj_{i}": lambda x, v, i=i: (
                v
                * (
                    self._dpsi_j[i](x, v) * feats_j[i](x, v)
                    + self._psi_j[i](x, v) * jacrev(feats_j[i], 0)(x, v).squeeze()
                )
            )
            for i in range(self._Mp_j)
        }

        # Compute vectors
        vec_j = jnp.array(tree_map_funcs(funcs_j, (x, v))).flatten()
        vec_r = jnp.array(tree_map_funcs(funcs_r, (x, v))).flatten()
        vec_dj_dx = jnp.array(tree_map_funcs(funcs_dj_dx, (x, v))).flatten()
        vec_vdr_dx = jnp.array(tree_map_funcs(funcs_vdr_dx, (x, v))).flatten()

        # Integrate with respect to v
        def integrate_v(fn_dict):
            return (
                jnp
                .array([
                    self._integrator(fn, (self._pts, self._ws), 1)(x)
                    for fn in fn_dict.values()
                ])
                .squeeze()
                .flatten()
            )

        vec_avg_vdj_dx = integrate_v(funcs_v_dj)
        vec_avg_r = integrate_v(funcs_r)

        # Assemble residual matrix
        n_dof = self._Mp_j * self._Jn_j + self._Mp_r * self._Jn_r
        n_j = self._Mp_j * self._Jn_j

        residual = jnp.zeros((3, n_dof))
        sigma_s_val = self._sigma_s(x)
        sigma_a_val = self._sigma_a(x)
        kn2 = self.kn**2
        coeff = sigma_s_val + kn2 * sigma_a_val

        residual = residual.at[0, :n_j].set(vec_avg_vdj_dx)
        residual = residual.at[0, n_j:].set(sigma_a_val * vec_avg_r)
        residual = residual.at[1, :n_j].set(kn2 * (v * vec_dj_dx - vec_avg_vdj_dx))
        residual = residual.at[1, n_j:].set(coeff * (vec_r - vec_avg_r))
        residual = residual.at[2, :n_j].set(coeff * vec_j)
        residual = residual.at[2, n_j:].set(vec_vdr_dx)

        return residual

    def _integrator(
        self,
        fn: Callable,
        quadratures: Tuple[jnp.ndarray, jnp.ndarray],
        argnum: int,
    ) -> Callable:
        """Integrate function with respect to specified argument using Gauss quadrature."""
        pts, ws = quadratures

        def integral_fn(*args):
            args_list = list(args)
            in_axes = [None] * len(args_list)
            args_list.insert(argnum, pts)
            in_axes.insert(argnum, 0)

            vmap_result = vmap(fn, in_axes=in_axes, out_axes=-1)(*args_list)
            return jnp.array([jnp.dot(vmap_result[i], ws) for i in range(self._Jn_r)])

        return integral_fn

    def normalize_j_pou_fn(self):
        """Construct normalized partition of unity functions for j model."""

        def psi_sym(x, v, i):
            # center_v = self._center_j(i)[1]
            center_v = jnp.where(v >= 0, self._center_j(i)[-1], -self._center_j(i)[-1])
            v_rel = v - center_v
            x_norm = (x - self._center_j(i)[0]) / self._radius_j[0]
            v_norm = v_rel / self._radius_j[-1]
            return psi(x_norm) * psi(v_norm)

        def dpsi_sym(x, v, i):

            center_v = jnp.where(v >= 0, self._center_j(i)[-1], -self._center_j(i)[-1])
            v_rel = v - center_v
            x_norm = (x - self._center_j(i)[0]) / self._radius_j[0]
            v_norm = v_rel / self._radius_j[-1]
            return dpsi(x_norm) / self._radius_j[0] * psi(v_norm)

        def psi_sum(x, v):
            return jnp.sum(jnp.array([psi_sym(x, v, i) for i in range(self._Mp_j)]))

        def dpsi_sum(x, v):
            return jnp.sum(jnp.array([dpsi_sym(x, v, i) for i in range(self._Mp_j)]))

        psi_list = [
            lambda x, v, i=i: psi_sym(x, v, i) / psi_sum(x, v)
            for i in range(self._Mp_j)
        ]
        dpsi_list = [
            lambda x, v, i=i: (
                (dpsi_sym(x, v, i) * psi_sum(x, v) - dpsi_sum(x, v) * psi_sym(x, v, i))
                / psi_sum(x, v) ** 2
            )
            for i in range(self._Mp_j)
        ]

        return psi_list, dpsi_list

    def normalize_r_pou_fn(self):
        """Construct normalized partition of unity functions for r model."""

        def psi_sym(x, v, i):

            center_v = jnp.where(v >= 0, self._center_r(i)[-1], -self._center_r(i)[-1])
            v_rel = v - center_v
            x_norm = (x - self._center_r(i)[0]) / self._radius_r[0]
            v_norm = v_rel / self._radius_r[-1]
            return psi(x_norm) * psi(v_norm)

        def dpsi_sym(x, v, i):
            center_v = jnp.where(v >= 0, self._center_r(i)[-1], -self._center_r(i)[-1])
            v_rel = v - center_v
            x_norm = (x - self._center_r(i)[0]) / self._radius_r[0]
            v_norm = v_rel / self._radius_r[-1]
            return dpsi(x_norm) / self._radius_r[0] * psi(v_norm)

        def psi_sum(x, v):
            return jnp.sum(jnp.array([psi_sym(x, v, i) for i in range(self._Mp_r)]))

        def dpsi_sum(x, v):
            return jnp.sum(jnp.array([dpsi_sym(x, v, i) for i in range(self._Mp_r)]))

        psi_list = [
            lambda x, v, i=i: psi_sym(x, v, i) / psi_sum(x, v)
            for i in range(self._Mp_r)
        ]
        dpsi_list = [
            lambda x, v, i=i: (
                (dpsi_sym(x, v, i) * psi_sum(x, v) - dpsi_sum(x, v) * psi_sym(x, v, i))
                / psi_sum(x, v) ** 2
            )
            for i in range(self._Mp_r)
        ]

        return psi_list, dpsi_list


class OddEvenOriginalDecompositionPointwiseInteriorConstraint1D(RandomFeatureSpaceXV):
    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: int
    scale: float
    init_rng: random.PRNGKey
    num_quads: int
    kn: float
    activation: Callable
    coeff_fns: Dict[str, Callable]

    def setup(self):
        # Quadrature points and weights
        self._pts, self._ws = leggauss(self.num_quads, interval=[0, 1])

        # Initialize models
        dummy_x, dummy_v = jnp.split(jnp.empty((2,)), 2, axis=0)

        self._model_j = RandomFeatureSpaceXV(
            self.domain, self.strides, self.Jn["j"], self.scale, self.activation
        )
        self._model_r = RandomFeatureSpaceXV(
            self.domain, self.strides, self.Jn["r"], self.scale, self.activation
        )

        params_j = self._model_j.init(self.init_rng, dummy_x, dummy_v)
        params_r = self._model_r.init(self.init_rng, dummy_x, dummy_v)

        self._feats_fn_j = partial(self._model_j.apply, params_j, has_aux=True)
        self._feats_fn_r = partial(self._model_r.apply, params_r, has_aux=True)

        # Cache mesh properties
        mesh_j = self._model_j._mesh
        mesh_r = self._model_r._mesh

        self._center_j = mesh_j.center_of_cell
        self._center_r = mesh_r.center_of_cell
        self._radius_j = mesh_j.radius_of_cell
        self._radius_r = mesh_r.radius_of_cell
        self._Mp_j = mesh_j.number_of_cells
        self._Mp_r = mesh_r.number_of_cells
        self._Jn_j = self.Jn["j"]
        self._Jn_r = self.Jn["r"]

        # Coefficient functions
        self._sigma_s = self.coeff_fns["scattering"]
        self._sigma_a = self.coeff_fns["absorption"]

        # Partition of unity functions
        self._psi_j, self._dpsi_j = self.normalize_j_pou_fn()
        self._psi_r, self._dpsi_r = self.normalize_r_pou_fn()

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_v: jnp.ndarray,
    ) -> jnp.ndarray:
        x, v = invar_x, invar_v
        _, feats_j = self._feats_fn_j(x, v)
        _, feats_r = self._feats_fn_r(x, v)

        funcs_j = {
            f"j_{i}": lambda x, v, i=i: (
                0.5
                * (
                    self._psi_j[i](x, v) * feats_j[i](x, v)
                    - self._psi_j[i](x, -v) * feats_j[i](x, -v)
                )
            )
            for i in range(self._Mp_j)
        }

        funcs_r = {
            f"r_{i}": lambda x, v, i=i: (
                0.5
                * (
                    self._psi_r[i](x, v) * feats_r[i](x, v)
                    + self._psi_r[i](x, -v) * feats_r[i](x, -v)
                )
            )
            for i in range(self._Mp_r)
        }

        funcs_vdr_dx = {
            f"dr_dx_{i}": lambda x, v, i=i: (
                0.5
                * v
                * (
                    (
                        self._dpsi_r[i](x, v) * feats_r[i](x, v)
                        + self._psi_r[i](x, v) * jacrev(feats_r[i], 0)(x, v).squeeze()
                    )
                    + (
                        self._dpsi_r[i](x, -v) * feats_r[i](x, -v)
                        + self._psi_r[i](x, -v) * jacrev(feats_r[i], 0)(x, -v).squeeze()
                    )
                )
            )
            for i in range(self._Mp_r)
        }

        funcs_vdj_dx = {
            f"v_dj_{i}": lambda x, v, i=i: (
                0.5
                * v
                * (
                    (
                        self._dpsi_j[i](x, v) * feats_j[i](x, v)
                        + self._psi_j[i](x, v) * jacrev(feats_j[i], 0)(x, v).squeeze()
                    )
                    - (
                        self._dpsi_j[i](x, -v) * feats_j[i](x, -v)
                        + self._psi_j[i](x, -v) * jacrev(feats_j[i], 0)(x, -v).squeeze()
                    )
                )
            )
            for i in range(self._Mp_j)
        }

        # Compute vectors
        vec_j = jnp.array(tree_map_funcs(funcs_j, (x, v))).flatten()
        vec_r = jnp.array(tree_map_funcs(funcs_r, (x, v))).flatten()
        vec_vdr_dx = jnp.array(tree_map_funcs(funcs_vdr_dx, (x, v))).flatten()
        vec_vdj_dx = jnp.array(tree_map_funcs(funcs_vdj_dx, (x, v))).flatten()

        # Integrate with respect to v
        def integrate_v(fn_dict):
            return (
                jnp
                .array([
                    self._integrator(fn, (self._pts, self._ws), 1)(x)
                    for fn in fn_dict.values()
                ])
                .squeeze()
                .flatten()
            )

        vec_avg_r = integrate_v(funcs_r)

        # Assemble residual matrix
        n_dof = self._Mp_j * self._Jn_j + self._Mp_r * self._Jn_r
        n_j = self._Mp_j * self._Jn_j

        residual = jnp.zeros((2, n_dof))
        sigma_s_val = self._sigma_s(x)
        sigma_a_val = self._sigma_a(x)
        kn2 = self.kn**2

        residual = residual.at[0, :n_j].set(kn2 * vec_vdj_dx)
        residual = residual.at[0, n_j:].set(
            (sigma_s_val + kn2 * sigma_a_val) * vec_r - sigma_s_val * vec_avg_r
        )
        residual = residual.at[1, :n_j].set((sigma_s_val + kn2 * sigma_a_val) * vec_j)
        residual = residual.at[1, n_j:].set(vec_vdr_dx)

        return residual

    def _integrator(
        self,
        fn: Callable,
        quadratures: Tuple[jnp.ndarray, jnp.ndarray],
        argnum: int,
    ) -> Callable:
        """Integrate function with respect to specified argument using Gauss quadrature."""
        pts, ws = quadratures

        def integral_fn(*args):
            args_list = list(args)
            in_axes = [None] * len(args_list)
            args_list.insert(argnum, pts)
            in_axes.insert(argnum, 0)

            vmap_result = vmap(fn, in_axes=in_axes, out_axes=-1)(*args_list)
            return jnp.array([jnp.dot(vmap_result[i], ws) for i in range(self._Jn_r)])

        return integral_fn

    def normalize_j_pou_fn(self):
        """Construct normalized partition of unity functions for j model."""

        def psi_sym(x, v, i):
            # center_v = self._center_j(i)[1]
            center_v = jnp.where(v >= 0, self._center_j(i)[-1], -self._center_j(i)[-1])
            v_rel = v - center_v
            x_norm = (x - self._center_j(i)[0]) / self._radius_j[0]
            v_norm = v_rel / self._radius_j[-1]
            return psi(x_norm) * psi(v_norm)

        def dpsi_sym(x, v, i):

            center_v = jnp.where(v >= 0, self._center_j(i)[-1], -self._center_j(i)[-1])
            v_rel = v - center_v
            x_norm = (x - self._center_j(i)[0]) / self._radius_j[0]
            v_norm = v_rel / self._radius_j[-1]
            return dpsi(x_norm) / self._radius_j[0] * psi(v_norm)

        def psi_sum(x, v):
            return jnp.sum(jnp.array([psi_sym(x, v, i) for i in range(self._Mp_j)]))

        def dpsi_sum(x, v):
            return jnp.sum(jnp.array([dpsi_sym(x, v, i) for i in range(self._Mp_j)]))

        psi_list = [
            lambda x, v, i=i: psi_sym(x, v, i) / psi_sum(x, v)
            for i in range(self._Mp_j)
        ]
        dpsi_list = [
            lambda x, v, i=i: (
                (dpsi_sym(x, v, i) * psi_sum(x, v) - dpsi_sum(x, v) * psi_sym(x, v, i))
                / psi_sum(x, v) ** 2
            )
            for i in range(self._Mp_j)
        ]

        return psi_list, dpsi_list

    def normalize_r_pou_fn(self):
        """Construct normalized partition of unity functions for r model."""

        def psi_sym(x, v, i):

            center_v = jnp.where(v >= 0, self._center_r(i)[-1], -self._center_r(i)[-1])
            v_rel = v - center_v
            x_norm = (x - self._center_r(i)[0]) / self._radius_r[0]
            v_norm = v_rel / self._radius_r[-1]
            return psi(x_norm) * psi(v_norm)

        def dpsi_sym(x, v, i):
            center_v = jnp.where(v >= 0, self._center_r(i)[-1], -self._center_r(i)[-1])
            v_rel = v - center_v
            x_norm = (x - self._center_r(i)[0]) / self._radius_r[0]
            v_norm = v_rel / self._radius_r[-1]
            return dpsi(x_norm) / self._radius_r[0] * psi(v_norm)

        def psi_sum(x, v):
            return jnp.sum(jnp.array([psi_sym(x, v, i) for i in range(self._Mp_r)]))

        def dpsi_sum(x, v):
            return jnp.sum(jnp.array([dpsi_sym(x, v, i) for i in range(self._Mp_r)]))

        psi_list = [
            lambda x, v, i=i: psi_sym(x, v, i) / psi_sum(x, v)
            for i in range(self._Mp_r)
        ]
        dpsi_list = [
            lambda x, v, i=i: (
                (dpsi_sym(x, v, i) * psi_sum(x, v) - dpsi_sum(x, v) * psi_sym(x, v, i))
                / psi_sum(x, v) ** 2
            )
            for i in range(self._Mp_r)
        ]

        return psi_list, dpsi_list


class PointwiseInteriorConstraint1D(RandomFeatureSpaceXV):
    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: int
    scale: float
    init_rng: random.PRNGKey
    num_quads: int
    kn: float
    activation: Callable
    coeff_fns: Dict[str, Callable]

    def setup(self):
        # Quadrature points and weights
        self._pts, self._ws = leggauss(self.num_quads)

        # Initialize model
        dummy_x, dummy_v = jnp.split(jnp.empty((2,)), 2, axis=0)

        self._model_f = RandomFeatureSpaceXV(
            self.domain, self.strides, self.Jn["f"], self.scale, self.activation
        )

        params_f = self._model_f.init(self.init_rng, dummy_x, dummy_v)
        self._feats_fn_f = partial(self._model_f.apply, params_f, has_aux=True)

        # Cache mesh properties
        mesh_f = self._model_f._mesh
        self._center_f = mesh_f.center_of_cell
        self._radius_f = mesh_f.radius_of_cell
        self._Mp_f = mesh_f.number_of_cells
        self._Jn_f = self.Jn["f"]

        # Coefficient functions
        self._sigma_s = self.coeff_fns["scattering"]
        self._sigma_a = self.coeff_fns["absorption"]

        # Partition of unity functions
        self._psiXV, self._dpsiXV = self.normalize_f_pou_fn()

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_v: jnp.ndarray,
    ) -> jnp.ndarray:
        x, v = invar_x, invar_v
        _, feats_f = self._feats_fn_f(x, v)

        funcs_f = {
            f"f_{i}": lambda x, v, i=i: self._psiXV[i](x, v) * feats_f[i](x, v)
            for i in range(self._Mp_f)
        }
        funcs_df_dx = {
            f"df_dx_{i}": lambda x, v, i=i: (
                self._dpsiXV[i](x, v) * feats_f[i](x, v)
                + self._psiXV[i](x, v) * jacrev(feats_f[i], 0)(x, v).squeeze()
            )
            for i in range(self._Mp_f)
        }

        vec_f = jnp.array(tree_map_funcs(funcs_f, (x, v))).flatten()
        vec_df_dx = jnp.array(tree_map_funcs(funcs_df_dx, (x, v))).flatten()

        # Integrate with respect to v
        vec_avg_f = (
            jnp
            .array([
                self._integrator(fn, (self._pts, self._ws), 1)(x)
                for fn in funcs_f.values()
            ])
            .squeeze()
            .flatten()
            / 2.0
        )

        # Compute residual
        sigma_s_val = self._sigma_s(x)
        sigma_a_val = self._sigma_a(x)

        residual = self.kn * v * vec_df_dx - (
            sigma_s_val * (vec_avg_f - vec_f) - self.kn**2 * sigma_a_val * vec_f
        )

        return residual.reshape(1, -1)

    def _integrator(
        self,
        fn: Callable,
        quadratures: Tuple[jnp.ndarray, jnp.ndarray],
        argnum: int,
    ) -> Callable:
        """Integrate function with respect to specified argument using Gauss quadrature."""
        pts, ws = quadratures

        def integral_fn(*args):
            args_list = list(args)
            in_axes = [None] * len(args_list)
            args_list.insert(argnum, pts)
            in_axes.insert(argnum, 0)

            vmap_result = vmap(fn, in_axes=in_axes, out_axes=-1)(*args_list)
            return jnp.array([jnp.dot(vmap_result[i], ws) for i in range(self._Jn_f)])

        return integral_fn

    def normalize_f_pou_fn(self):
        """Construct normalized partition of unity functions for f model."""

        def psi_sym(x, v, i):
            center_v = jnp.where(v >= 0, self._center_f(i)[-1], -self._center_f(i)[-1])
            v_rel = v - center_v
            x_norm = (x - self._center_f(i)[0]) / self._radius_f[0]
            v_norm = v_rel / self._radius_f[-1]
            return psi(x_norm) * psi(v_norm)

        def dpsi_sym(x, v, i):
            center_v = jnp.where(v >= 0, self._center_f(i)[-1], -self._center_f(i)[-1])
            v_rel = v - center_v
            x_norm = (x - self._center_f(i)[0]) / self._radius_f[0]
            v_norm = v_rel / self._radius_f[-1]
            return dpsi(x_norm) / self._radius_f[0] * psi(v_norm)

        def psi_sum(x, v):
            return jnp.sum(jnp.array([psi_sym(x, v, i) for i in range(self._Mp_f)]))

        def dpsi_sum(x, v):
            return jnp.sum(jnp.array([dpsi_sym(x, v, i) for i in range(self._Mp_f)]))

        psi_list = [
            lambda x, v, i=i: psi_sym(x, v, i) / psi_sum(x, v)
            for i in range(self._Mp_f)
        ]
        dpsi_list = [
            lambda x, v, i=i: (
                (dpsi_sym(x, v, i) * psi_sum(x, v) - dpsi_sum(x, v) * psi_sym(x, v, i))
                / psi_sum(x, v) ** 2
            )
            for i in range(self._Mp_f)
        ]

        return psi_list, dpsi_list


class OddEvenDecompositionPointwiseBoundaryConstraint1D(RandomFeatureSpaceXV):
    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: int
    scale: float
    init_rng: random.PRNGKey
    kn: float
    activation: Callable

    def setup(self):
        # Initialize models
        dummy_x, dummy_v = jnp.split(jnp.empty((2,)), 2, axis=0)

        self._model_j = RandomFeatureSpaceXV(
            self.domain, self.strides, self.Jn["j"], self.scale, self.activation
        )
        self._model_r = RandomFeatureSpaceXV(
            self.domain, self.strides, self.Jn["r"], self.scale, self.activation
        )

        params_j = self._model_j.init(self.init_rng, dummy_x, dummy_v)
        params_r = self._model_r.init(self.init_rng, dummy_x, dummy_v)

        self._feats_fn_j = partial(self._model_j.apply, params_j)
        self._feats_fn_r = partial(self._model_r.apply, params_r)

        # Cache mesh properties
        mesh_j = self._model_j._mesh
        mesh_r = self._model_r._mesh

        self._center_j = mesh_j.center_of_cell
        self._center_r = mesh_r.center_of_cell
        self._radius_j = mesh_j.radius_of_cell
        self._radius_r = mesh_r.radius_of_cell
        self._Mp_j = mesh_j.number_of_cells
        self._Mp_r = mesh_r.number_of_cells

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_v: jnp.ndarray,
    ) -> jnp.ndarray:
        x, v = invar_x, invar_v

        # Compute odd-even decomposition
        feats_j = 0.5 * (self._feats_fn_j(x, v) - self._feats_fn_j(x, -v))
        feats_r = 0.5 * (self._feats_fn_r(x, v) + self._feats_fn_r(x, -v))

        # Flatten features
        vec_j = feats_j.flatten() if self._Mp_j > 1 else feats_j.reshape(-1)
        vec_r = feats_r.flatten() if self._Mp_r > 1 else feats_r.reshape(-1)

        sign = jnp.where(x > 0, -1, 1)

        return jnp.concatenate([sign * self.kn * vec_j, vec_r])


class OddEvenPointwiseBoundaryConstraint1D(RandomFeatureSpaceXV):
    """
    Without the embedding of even and odd functions into the model architecture, this constraint directly computes the odd-even decomposition of the features, and applies a sign change to the odd part based on the sign of x. The resulting vector concatenates the signed odd features and the even features.
    """

    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: int
    scale: float
    init_rng: random.PRNGKey
    kn: float
    activation: Callable

    def setup(self):
        # Initialize models
        dummy_x, dummy_v = jnp.split(jnp.empty((2,)), 2, axis=0)

        self._model_j = RandomFeatureSpaceXV(
            self.domain, self.strides, self.Jn["j"], self.scale, self.activation
        )
        self._model_r = RandomFeatureSpaceXV(
            self.domain, self.strides, self.Jn["r"], self.scale, self.activation
        )

        params_j = self._model_j.init(self.init_rng, dummy_x, dummy_v)
        params_r = self._model_r.init(self.init_rng, dummy_x, dummy_v)

        self._feats_fn_j = partial(self._model_j.apply, params_j)
        self._feats_fn_r = partial(self._model_r.apply, params_r)

        # Cache mesh properties
        mesh_j = self._model_j._mesh
        mesh_r = self._model_r._mesh

        self._center_j = mesh_j.center_of_cell
        self._center_r = mesh_r.center_of_cell
        self._radius_j = mesh_j.radius_of_cell
        self._radius_r = mesh_r.radius_of_cell
        self._Mp_j = mesh_j.number_of_cells
        self._Mp_r = mesh_r.number_of_cells

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_v: jnp.ndarray,
    ) -> jnp.ndarray:
        x, v = invar_x, invar_v

        # Compute odd-even decomposition
        feats_j = self._feats_fn_j(x, v)
        feats_r = self._feats_fn_r(x, v)

        # Flatten features
        vec_j = feats_j.flatten() if self._Mp_j > 1 else feats_j.reshape(-1)
        vec_r = feats_r.flatten() if self._Mp_r > 1 else feats_r.reshape(-1)

        sign = jnp.where(x > 0, -1, 1)

        return jnp.concatenate([sign * self.kn * vec_j, vec_r])


class PointwiseBoundaryConstraint1D(RandomFeatureSpaceXV):
    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: int
    scale: float
    init_rng: random.PRNGKey
    kn: float
    activation: Callable

    def setup(self):
        # Initialize model
        dummy_x, dummy_v = jnp.split(jnp.empty((2,)), 2, axis=0)

        self._model_f = RandomFeatureSpaceXV(
            self.domain, self.strides, self.Jn["f"], self.scale, self.activation
        )

        params_f = self._model_f.init(self.init_rng, dummy_x, dummy_v)
        self._feats_fn_f = partial(self._model_f.apply, params_f)

        # Cache mesh properties
        mesh_f = self._model_f._mesh
        self._center_f = mesh_f.center_of_cell
        self._radius_f = mesh_f.radius_of_cell
        self._Mp_f = mesh_f.number_of_cells

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_v: jnp.ndarray,
    ) -> jnp.ndarray:
        feats = self._feats_fn_f(invar_x, invar_v)
        return feats.flatten() if self._Mp_f > 1 else feats.reshape(-1)
