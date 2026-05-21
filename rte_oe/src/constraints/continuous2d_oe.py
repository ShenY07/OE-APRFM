import jax.numpy as jnp
from jax import random, vmap, jacrev
from utils.integrate import leggauss
from utils.parallel import tree_map_funcs
from collections.abc import Callable
from typing import Dict, Tuple
from modules.pou_func import (psi_a as psi, dpsi_a as dpsi)
from modules.func_space import RandomFeatureSpaceXYV
from functools import partial
from jax.tree_util import tree_map
import jax


class OddEvenDecompositionPointwiseInteriorConstraint2D(RandomFeatureSpaceXYV):
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
        self._quadratures = leggauss(self.num_quads, interval=(0, jnp.pi))
        self._pts, self._ws = self._quadratures
        self._model_j = RandomFeatureSpaceXYV(
            self.domain,
            self.strides,
            self.Jn["j"],
            self.scale,
            self.activation,
        )
        self._model_r = RandomFeatureSpaceXYV(
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
        self._params_j = self._model_j.init(
            self.init_rng, self._dummy_x, self._dummy_y, self._dummy_theta
        )
        self._params_r = self._model_r.init(
            self.init_rng, self._dummy_x, self._dummy_y, self._dummy_theta
        )
        self._feats_fn_j = partial(
            self._model_j.apply, self._params_j, has_aux=True)
        self._feats_fn_r = partial(
            self._model_r.apply, self._params_r, has_aux=True)
        self._mesh_j = self._model_j._mesh
        self._center_j = self._mesh_j.center_of_cell
        self._radius_j = self._mesh_j.radius_of_cell
        self._Mp_j = self._mesh_j.number_of_cells
        self._mesh_r = self._model_r._mesh
        self._center_r = self._mesh_r.center_of_cell
        self._radius_r = self._mesh_r.radius_of_cell
        self._Mp_r = self._mesh_r.number_of_cells
        self._Jn_j = self.Jn["j"]
        self._Jn_r = self.Jn["r"]
        self._sigma_s = self.coeff_fns["scattering"]
        self._sigma_a = self.coeff_fns["absorption"]
        self._psi = psi
        self._dpsi = dpsi
        self._psi_j_XYV, self._dpsi_j_dX, self._dpsi_j_dY = self.normalize_pou_fn(
            kind="j"
        )
        self._psi_r_XYV, self._dpsi_r_dX, self._dpsi_r_dY = self.normalize_pou_fn(
            kind="r"
        )

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_y: jnp.ndarray,
        invar_theta: jnp.ndarray,
    ) -> jnp.ndarray:
        _, feat_operators_j = self._feats_fn_j(invar_x, invar_y, invar_theta)
        _, feat_operators_r = self._feats_fn_r(invar_x, invar_y, invar_theta)
        x, y, theta = invar_x, invar_y, invar_theta
        feats_j = feat_operators_j
        feats_r = feat_operators_r
        funcs_j = {
            i: (
                lambda x, y, theta, i=i: 0.5
                * (
                    self._psi_j_XYV[i](x, y, theta) * feats_j[i](x, y, theta)
                    - self._psi_j_XYV[i](x, y, (jnp.pi + theta) % (2 * jnp.pi))
                    * feats_j[i](x, y, (jnp.pi + theta) % (2 * jnp.pi))
                )
            )
            for i in range(self._Mp_j)
        }
        funcs_r = {
            i: (
                lambda x, y, theta, i=i: 0.5
                * (
                    self._psi_r_XYV[i](x, y, theta) * feats_r[i](x, y, theta)
                    + self._psi_r_XYV[i](x, y, (jnp.pi + theta) % (2 * jnp.pi))
                    * feats_r[i](x, y, (jnp.pi + theta) % (2 * jnp.pi))
                )
            )
            for i in range(self._Mp_r)
        }
        funcs_dj_dx = {
            i: (
                lambda x, y, theta, i=i: 0.5
                * (
                    (
                        self._dpsi_j_dX[i](x, y, theta) *
                        feats_j[i](x, y, theta)
                        + self._psi_j_XYV[i](x, y, theta)
                        * jacrev(feats_j[i], 0)(x, y, theta).squeeze()
                    )
                    - (
                        self._dpsi_j_dX[i](
                            x, y, (jnp.pi + theta) % (2 * jnp.pi))
                        * feats_j[i](x, y, (jnp.pi + theta) % (2 * jnp.pi))
                        + self._psi_j_XYV[i](x, y,
                                             (jnp.pi + theta) % (2 * jnp.pi))
                        * jacrev(feats_j[i], 0)(
                            x, y, (jnp.pi + theta) % (2 * jnp.pi)
                        ).squeeze()
                    )
                )
            )
            for i in range(self._Mp_j)
        }
        funcs_dj_dy = {
            i: (
                lambda x, y, theta, i=i: 0.5
                * (
                    (
                        self._dpsi_j_dY[i](x, y, theta) *
                        feats_j[i](x, y, theta)
                        + self._psi_j_XYV[i](x, y, theta)
                        * jacrev(feats_j[i], 1)(x, y, theta).squeeze()
                    )
                    - (
                        self._dpsi_j_dY[i](
                            x, y, (jnp.pi + theta) % (2 * jnp.pi))
                        * feats_j[i](x, y, (jnp.pi + theta) % (2 * jnp.pi))
                        + self._psi_j_XYV[i](x, y,
                                             (jnp.pi + theta) % (2 * jnp.pi))
                        * jacrev(feats_j[i], 1)(
                            x, y, (jnp.pi + theta) % (2 * jnp.pi)
                        ).squeeze()
                    )
                )
            )
            for i in range(self._Mp_j)
        }
        funcs_vdj = {
            i: (
                lambda x, y, theta, i=i: 0.5
                * (
                    (
                        jnp.cos(theta) * funcs_dj_dx[i](x, y, theta)
                        + jnp.sin(theta) * funcs_dj_dy[i](x, y, theta)
                    )
                    - (
                        jnp.cos(theta)
                        * funcs_dj_dx[i](x, y, (jnp.pi + theta) % (2 * jnp.pi))
                        + jnp.sin(theta)
                        * funcs_dj_dy[i](x, y, (jnp.pi + theta) % (2 * jnp.pi))
                    )
                )
            )
            for i in range(self._Mp_j)
        }
        funcs_dr_dx = {
            i: (
                lambda x, y, theta, i=i: 0.5
                * (
                    (
                        self._dpsi_r_dX[i](x, y, theta) *
                        feats_r[i](x, y, theta)
                        + self._psi_r_XYV[i](x, y, theta)
                        * jacrev(feats_r[i], 0)(x, y, theta).squeeze()
                    )
                    + (
                        self._dpsi_r_dX[i](
                            x, y, (jnp.pi + theta) % (2 * jnp.pi))
                        * feats_r[i](x, y, (jnp.pi + theta) % (2 * jnp.pi))
                        + self._psi_r_XYV[i](x, y,
                                             (jnp.pi + theta) % (2 * jnp.pi))
                        * jacrev(feats_r[i], 0)(
                            x, y, (jnp.pi + theta) % (2 * jnp.pi)
                        ).squeeze()
                    )
                )
            )
            for i in range(self._Mp_r)
        }
        funcs_dr_dy = {
            i: (
                lambda x, y, theta, i=i: 0.5
                * (
                    (
                        self._dpsi_r_dY[i](x, y, theta) *
                        feats_r[i](x, y, theta)
                        + self._psi_r_XYV[i](x, y, theta)
                        * jacrev(feats_r[i], 1)(x, y, theta).squeeze()
                    )
                    + (
                        self._dpsi_r_dY[i](
                            x, y, (jnp.pi + theta) % (2 * jnp.pi))
                        * feats_r[i](x, y, (jnp.pi + theta) % (2 * jnp.pi))
                        + self._psi_r_XYV[i](x, y,
                                             (jnp.pi + theta) % (2 * jnp.pi))
                        * jacrev(feats_r[i], 1)(
                            x, y, (jnp.pi + theta) % (2 * jnp.pi)
                        ).squeeze()
                    )
                )
            )
            for i in range(self._Mp_r)
        }
        funcs_vdr = {
            i: (
                lambda x, y, theta, i=i: 0.5
                * (
                    (
                        jnp.cos(theta) * funcs_dr_dx[i](x, y, theta)
                        + jnp.sin(theta) * funcs_dr_dy[i](x, y, theta)
                    )
                    + (
                        jnp.cos(theta)
                        * funcs_dr_dx[i](x, y, (jnp.pi + theta) % (2 * jnp.pi))
                        + jnp.sin(theta)
                        * funcs_dr_dy[i](x, y, (jnp.pi + theta) % (2 * jnp.pi))
                    )
                )
            )
            for i in range(self._Mp_r)
        }

        vector_j = jnp.array(tree_map_funcs(funcs_j, (x, y, theta))).flatten()
        vector_r = jnp.array(tree_map_funcs(funcs_r, (x, y, theta))).flatten()
        vector_vdj = jnp.array(tree_map_funcs(
            funcs_vdj, (x, y, theta))).flatten()

        vector_vdr = jnp.array(tree_map_funcs(
            funcs_vdr, (x, y, theta))).flatten()

        vector_aver_vdj = (
            jnp.array(
                list(
                    tree_map(
                        lambda fn: self._integrator(
                            fn=fn,
                            quadratures=(self._pts, self._ws),
                            argnum=2,
                        )(x, y),
                        funcs_vdj,
                    ).values()
                )
            )
            .squeeze()
            .flatten()
            / jnp.pi
        )
        vector_aver_r = (
            jnp.array(
                list(
                    tree_map(
                        lambda fn: self._integrator(
                            fn=fn,
                            quadratures=(self._pts, self._ws),
                            argnum=2,
                        )(x, y),
                        funcs_r,
                    ).values()
                )
            )
            .squeeze()
            .flatten()
            / jnp.pi
        )

        eqn_residual = jnp.zeros((
            3,
            self._Mp_j * self._Jn_j + self._Mp_r * self._Jn_r,
        ))
        eqn_residual = eqn_residual.at[0, : self._Mp_j * self._Jn_j].set(
            vector_aver_vdj
        )
        eqn_residual = eqn_residual.at[0, self._Mp_j * self._Jn_j:].set(
            self._sigma_a(x, y) * vector_aver_r
        )
        eqn_residual = eqn_residual.at[1, : self._Mp_j * self._Jn_j].set(
            self.kn**2 * (vector_vdj - vector_aver_vdj)
        )
        eqn_residual = eqn_residual.at[1, self._Mp_j * self._Jn_j:].set(
            (self._sigma_s(x, y) + self.kn**2 * self._sigma_a(x, y))
            * (vector_r - vector_aver_r)
        )
        eqn_residual = eqn_residual.at[2, : self._Mp_j * self._Jn_j].set(
            (self._sigma_s(x, y) + self.kn**2 * self._sigma_a(x, y)) * vector_j
        )
        eqn_residual = eqn_residual.at[2,
                                       self._Mp_j * self._Jn_j:].set(vector_vdr)
        return eqn_residual

    def _integrator(
        self,
        fn: Callable,
        quadratures: Tuple[jnp.ndarray, jnp.ndarray],
        argnum: int,
    ) -> Callable:
        """Integrate with respect to theta.

        Supports scalar or vector-valued integrands. If the integrand returns
        a vector of length K (e.g., per-basis component), the integral is
        computed component-wise to return shape (K,). If the integrand returns
        a scalar, the integral returns a scalar.
        """
        pts, ws = quadratures

        def integral_fn(*args):
            args = list(args)
            in_axes_ = [None] * len(args)
            args.insert(argnum, pts)
            in_axes_.insert(argnum, int(0))
            vmap_fns = vmap(fn, in_axes=in_axes_, out_axes=-1)(*args)

            # Integrate over the last axis (theta quadrature) for each component
            # If vmap_fns has shape (Q,), returns scalar; if (K, Q), returns (K,)
            out = jnp.tensordot(vmap_fns, ws, axes=(-1, 0))
            return out

        return integral_fn

    def normalize_pou_fn(self, kind="j"):
        # 根据 kind 选择 mesh
        if kind == "j":
            center = self._center_j
            radius = self._radius_j
            Mp = self._Mp_j
        else:
            center = self._center_r
            radius = self._radius_r
            Mp = self._Mp_r

        def psi_sym(x, y, theta, i):
            center_theta = center(i)[2] + jnp.pi * jnp.floor(theta / jnp.pi)
            theta_rel = theta - center_theta
            return (
                self._psi((x - center(i)[0]) / radius[0])
                * self._psi((y - center(i)[1]) / radius[1])
                * self._psi(theta_rel / radius[2])
            )

        def dpsi_dx_sym(x, y, theta, i):
            center_theta = center(i)[2] + jnp.pi * jnp.floor(theta / jnp.pi)
            theta_rel = theta - center_theta
            return (
                self._dpsi((x - center(i)[0]) / radius[0])
                / radius[0]
                * self._psi((y - center(i)[1]) / radius[1])
                * self._psi(theta_rel / radius[2])
            )

        def dpsi_dy_sym(x, y, theta, i):
            center_theta = center(i)[2] + jnp.pi * jnp.floor(theta / jnp.pi)
            theta_rel = theta - center_theta
            return (
                self._dpsi((y - center(i)[1]) / radius[1])
                / radius[1]
                * self._psi((x - center(i)[0]) / radius[0])
                * self._psi(theta_rel / radius[2])
            )

        def psi_sum(x, y, theta):
            return jnp.sum(vmap(lambda i: psi_sym(x, y, theta, i))(jnp.arange(Mp)))

        def dpsi_dx_sum(x, y, theta):
            return jnp.sum(vmap(lambda i: dpsi_dx_sym(x, y, theta, i))(jnp.arange(Mp)))

        def dpsi_dy_sum(x, y, theta):
            return jnp.sum(vmap(lambda i: dpsi_dy_sym(x, y, theta, i))(jnp.arange(Mp)))

        psi_list = [
            lambda x, y, theta, i=i: psi_sym(
                x, y, theta, i) / psi_sum(x, y, theta)
            for i in range(Mp)
        ]
        dpsi_dx_list = [
            lambda x, y, theta, i=i: (
                dpsi_dx_sym(x, y, theta, i) * psi_sum(x, y, theta)
                - dpsi_dx_sum(x, y, theta) * psi_sym(x, y, theta, i)
            )
            / psi_sum(x, y, theta) ** 2
            for i in range(Mp)
        ]
        dpsi_dy_list = [
            lambda x, y, theta, i=i: (
                dpsi_dy_sym(x, y, theta, i) * psi_sum(x, y, theta)
                - dpsi_dy_sum(x, y, theta) * psi_sym(x, y, theta, i)
            )
            / psi_sum(x, y, theta) ** 2
            for i in range(Mp)
        ]

        return psi_list, dpsi_dx_list, dpsi_dy_list


class OddEvenDecompositionPointwiseBoundaryConstraint2D(RandomFeatureSpaceXYV):
    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: int
    scale: float
    init_rng: random.PRNGKey
    kn: float
    activation: Callable

    def setup(self):
        self._model_j = RandomFeatureSpaceXYV(
            self.domain,
            self.strides,
            self.Jn["j"],
            self.scale,
            self.activation,
        )
        self._model_r = RandomFeatureSpaceXYV(
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
        self._params_j = self._model_j.init(
            self.init_rng, self._dummy_x, self._dummy_y, self._dummy_theta
        )
        self._params_r = self._model_r.init(
            self.init_rng, self._dummy_x, self._dummy_y, self._dummy_theta
        )
        self._feats_fn_j = partial(self._model_j.apply, self._params_j)
        self._feats_fn_r = partial(self._model_r.apply, self._params_r)
        self._mesh_j = self._model_j._mesh
        self._center_j = self._mesh_j.center_of_cell
        self._radius_j = self._mesh_j.radius_of_cell
        self._Mp_j = self._mesh_j.number_of_cells
        self._mesh_r = self._model_r._mesh
        self._center_r = self._mesh_r.center_of_cell
        self._radius_r = self._mesh_r.radius_of_cell
        self._Mp_r = self._mesh_r.number_of_cells

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_y: jnp.ndarray,
        invar_theta: jnp.ndarray,
    ) -> jnp.ndarray:
        x, y, theta = invar_x, invar_y, invar_theta
        feats_j_pos = self._feats_fn_j(x, y, theta)
        feats_j_neg = self._feats_fn_j(x, y, (jnp.pi + theta) % (2 * jnp.pi))
        feats_r_pos = self._feats_fn_r(x, y, theta)
        feats_r_neg = self._feats_fn_r(x, y, (jnp.pi + theta) % (2 * jnp.pi))
        feats_j = 0.5 * (feats_j_pos - feats_j_neg)
        feats_r = 0.5 * (feats_r_pos + feats_r_neg)
        vector_j = (
            jnp.array([feats_j[i] for i in range(self._Mp_j)]).flatten()
            if self._Mp_j > 1
            else feats_j.reshape(-1)
        )
        vector_r = (
            jnp.array([feats_r[i] for i in range(self._Mp_r)]).flatten()
            if self._Mp_r > 1
            else feats_r.reshape(-1)
        )
        vector_f = jnp.concatenate([
            self.kn * vector_j,
            vector_r,
        ])  # (Mp_j*Jn_j + Mp_r*Jn_r, )
        return vector_f
