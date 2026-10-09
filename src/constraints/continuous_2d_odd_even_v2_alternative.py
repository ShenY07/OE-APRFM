"""Provide continuous 2d odd even variant 02 alternative functionality for the constraints layer.

Distinction: This file is an explicitly retained alternative to the numbered variant; numerical logic is preserved.
"""

import jax.numpy as jnp
from jax import random, vmap, jacrev
from utils.quadrature import leggauss
from utils.parallel import tree_map_funcs
from collections.abc import Callable
from typing import Dict, Tuple
from modules.partition_of_unity import psi_b as psi, dpsi_b as dpsi
from modules.function_space_v2 import RandomFeatureSpaceXYV
from functools import partial
from jax.tree_util import tree_map

"""
j1(x, y, theta) = -j1(x, y, theta + pi)
r1(x, y, theta) = r1(x, y, theta + pi)
j2(x, y, theta + pi/2) = -j2(x, y, theta + 3pi/2)
r2(x, y, theta + pi/2) = r2(x, y, theta + 3pi/2)
f(x, y, theta) = kn * j1(x, y, theta) + r1(x, y, theta) + kn * j2(x, y, theta) + r2(x, y, theta)
theta \in (0, pi/2)
"""


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
        self._feats_fn_j1 = partial(
            self._model_j1.apply, self._params_j1, has_aux=True
        )
        self._feats_fn_r1 = partial(
            self._model_r1.apply, self._params_r1, has_aux=True
        )
        self._feats_fn_j2 = partial(
            self._model_j2.apply, self._params_j2, has_aux=True
        )
        self._feats_fn_r2 = partial(
            self._model_r2.apply, self._params_r2, has_aux=True
        )
        self._mesh_j1 = self._model_j1._mesh
        self._center_j1 = self._mesh_j1.center_of_cell
        self._radius_j1 = self._mesh_j1.radius_of_cell
        self._Mp_j1 = self._mesh_j1.number_of_cells
        self._mesh_r1 = self._model_r1._mesh
        self._center_r1 = self._mesh_r1.center_of_cell
        self._radius_r1 = self._mesh_r1.radius_of_cell
        self._Mp_r1 = self._mesh_r1.number_of_cells
        self._mesh_j2 = self._model_j2._mesh
        self._center_j2 = self._mesh_j2.center_of_cell
        self._radius_j2 = self._mesh_j2.radius_of_cell
        self._Mp_j2 = self._mesh_j2.number_of_cells
        self._mesh_r2 = self._model_r2._mesh
        self._center_r2 = self._mesh_r2.center_of_cell
        self._radius_r2 = self._mesh_r2.radius_of_cell
        self._Mp_r2 = self._mesh_r2.number_of_cells
        self._jn_j1 = self.Jn["j"]
        self._jn_r1 = self.Jn["r"]
        self._jn_j2 = self.Jn["j"]
        self._jn_r2 = self.Jn["r"]
        self._sigma_s = self.coeff_fns["scattering"]
        self._sigma_a = self.coeff_fns["absorption"]
        self._psi = psi
        self._dpsi = dpsi
        self._psi_j1_XYV, self._dpsi_j1_XV, self._dpsi_j1_YV = (
            self.normalize_pou_fn(kind="j")
        )
        self._psi_r1_XYV, self._dpsi_r1_XV, self._dpsi_r1_YV = (
            self.normalize_pou_fn(kind="r")
        )
        self._psi_j2_XYV, self._dpsi_j2_XV, self._dpsi_j2_YV = (
            self.normalize_pou_fn(kind="j")
        )
        self._psi_r2_XYV, self._dpsi_r2_XV, self._dpsi_r2_YV = (
            self.normalize_pou_fn(kind="r")
        )

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_y: jnp.ndarray,
        invar_theta: jnp.ndarray,
    ) -> jnp.ndarray:
        _, feat_operators_j1 = self._feats_fn_j1(invar_x, invar_y, invar_theta)
        _, feat_operators_r1 = self._feats_fn_r1(invar_x, invar_y, invar_theta)
        _, feat_operators_j2 = self._feats_fn_j2(invar_x, invar_y, invar_theta)
        _, feat_operators_r2 = self._feats_fn_r2(invar_x, invar_y, invar_theta)
        x, y, theta_1 = invar_x, invar_y, invar_theta
        theta_2 = invar_theta + jnp.pi / 2
        feats_j1 = feat_operators_j1
        feats_r1 = feat_operators_r1
        feats_j2 = feat_operators_j2
        feats_r2 = feat_operators_r2

        funcs_j1 = {
            i: (
                lambda x, y, theta, i=i: (
                    0.5
                    * self._psi_j1_XYV[i](x, y)
                    * (
                        feats_j1[i](x, y, theta)
                        - feats_j1[i](x, y, jnp.pi + theta)
                    )
                )
            )
            for i in range(self._Mp_j1)
        }
        funcs_r1 = {
            i: (
                lambda x, y, theta, i=i: (
                    0.5
                    * self._psi_r1_XYV[i](x, y)
                    * (
                        feats_r1[i](x, y, theta)
                        + feats_r1[i](x, y, jnp.pi + theta)
                    )
                )
            )
            for i in range(self._Mp_r1)
        }
        funcs_dj1_dx = {
            i: (
                lambda x, y, theta, i=i: (
                    0.5
                    * (
                        (
                            self._dpsi_j1_XV[i](x, y)
                            * feats_j1[i](x, y, theta)
                            + self._psi_j1_XYV[i](x, y)
                            * jacrev(feats_j1[i], 0)(x, y, theta).squeeze()
                        )
                        - (
                            self._dpsi_j1_XV[i](x, y)
                            * feats_j1[i](x, y, jnp.pi + theta)
                            + self._psi_j1_XYV[i](x, y)
                            * jacrev(feats_j1[i], 0)(
                                x, y, jnp.pi + theta
                            ).squeeze()
                        )
                    )
                )
            )
            for i in range(self._Mp_j1)
        }
        funcs_dj1_dy = {
            i: (
                lambda x, y, theta, i=i: (
                    0.5
                    * (
                        (
                            self._dpsi_j1_YV[i](x, y)
                            * feats_j1[i](x, y, theta)
                            + self._psi_j1_XYV[i](x, y)
                            * jacrev(feats_j1[i], 1)(x, y, theta).squeeze()
                        )
                        - (
                            self._dpsi_j1_YV[i](x, y)
                            * feats_j1[i](x, y, jnp.pi + theta)
                            + self._psi_j1_XYV[i](x, y)
                            * jacrev(feats_j1[i], 1)(
                                x, y, jnp.pi + theta
                            ).squeeze()
                        )
                    )
                )
            )
            for i in range(self._Mp_j1)
        }
        funcs_vdj1 = {
            i: (
                lambda x, y, theta, i=i: (
                    jnp.cos(theta) * funcs_dj1_dx[i](x, y, theta)
                    + jnp.sin(theta) * funcs_dj1_dy[i](x, y, theta)
                )
            )
            for i in range(self._Mp_j1)
        }
        funcs_dr1_dx = {
            i: (
                lambda x, y, theta, i=i: (
                    0.5
                    * (
                        (
                            self._dpsi_r1_XV[i](x, y)
                            * feats_r1[i](x, y, theta)
                            + self._psi_r1_XYV[i](x, y)
                            * jacrev(feats_r1[i], 0)(x, y, theta).squeeze()
                        )
                        + (
                            self._dpsi_r1_XV[i](x, y)
                            * feats_r1[i](x, y, jnp.pi + theta)
                            + self._psi_r1_XYV[i](x, y)
                            * jacrev(feats_r1[i], 0)(
                                x, y, jnp.pi + theta
                            ).squeeze()
                        )
                    )
                )
            )
            for i in range(self._Mp_r1)
        }
        funcs_dr1_dy = {
            i: (
                lambda x, y, theta, i=i: (
                    0.5
                    * (
                        (
                            self._dpsi_r1_YV[i](x, y)
                            * feats_r1[i](x, y, theta)
                            + self._psi_r1_XYV[i](x, y)
                            * jacrev(feats_r1[i], 1)(x, y, theta).squeeze()
                        )
                        + (
                            self._dpsi_r1_YV[i](x, y)
                            * feats_r1[i](x, y, jnp.pi + theta)
                            + self._psi_r1_XYV[i](x, y)
                            * jacrev(feats_r1[i], 1)(
                                x, y, jnp.pi + theta
                            ).squeeze()
                        ).squeeze()
                    )
                )
            )
            for i in range(self._Mp_r1)
        }
        funcs_vdr1 = {
            i: (
                lambda x, y, theta, i=i: (
                    jnp.cos(theta) * funcs_dr1_dx[i](x, y, theta)
                    + jnp.sin(theta) * funcs_dr1_dy[i](x, y, theta)
                )
            )
            for i in range(self._Mp_r1)
        }

        vector_j1 = jnp.array(
            tree_map_funcs(funcs_j1, (x, y, theta_1))
        ).flatten()
        vector_r1 = jnp.array(
            tree_map_funcs(funcs_r1, (x, y, theta_1))
        ).flatten()
        vector_vdj1 = jnp.array(
            tree_map_funcs(funcs_vdj1, (x, y, theta_1))
        ).flatten()

        vector_vdr1 = jnp.array(
            tree_map_funcs(funcs_vdr1, (x, y, theta_1))
        ).flatten()

        vector_aver_vdj1 = (
            jnp.array(
                list(
                    tree_map(
                        lambda fn: self._integrator(
                            fn=fn,
                            quadratures=(self._pts, self._ws),
                            argnum=2,
                        )(x, y),
                        funcs_vdj1,
                    ).values()
                )
            )
            .squeeze()
            .flatten()
            / jnp.pi
        )
        vector_aver_r1 = (
            jnp.array(
                list(
                    tree_map(
                        lambda fn: self._integrator(
                            fn=fn,
                            quadratures=(self._pts, self._ws),
                            argnum=2,
                        )(x, y),
                        funcs_r1,
                    ).values()
                )
            )
            .squeeze()
            .flatten()
            / jnp.pi
        )

        funcs_j2 = {
            i: (
                lambda x, y, theta, i=i: (
                    0.5
                    * self._psi_j2_XYV[i](x, y)
                    * (
                        feats_j2[i](x, y, theta)
                        - feats_j2[i](x, y, jnp.pi + theta)
                    )
                )
            )
            for i in range(self._Mp_j2)
        }
        funcs_r2 = {
            i: (
                lambda x, y, theta, i=i: (
                    0.5
                    * self._psi_r2_XYV[i](x, y)
                    * (
                        feats_r2[i](x, y, theta)
                        + feats_r2[i](x, y, jnp.pi + theta)
                    )
                )
            )
            for i in range(self._Mp_r2)
        }
        funcs_dj2_dx = {
            i: (
                lambda x, y, theta, i=i: (
                    0.5
                    * (
                        (
                            self._dpsi_j2_XV[i](x, y)
                            * feats_j2[i](x, y, theta)
                            + self._psi_j2_XYV[i](x, y)
                            * jacrev(feats_j2[i], 0)(x, y, theta).squeeze()
                        )
                        - (
                            self._dpsi_j2_XV[i](x, y)
                            * feats_j2[i](x, y, jnp.pi + theta)
                            + self._psi_j2_XYV[i](x, y)
                            * jacrev(feats_j2[i], 0)(
                                x, y, jnp.pi + theta
                            ).squeeze()
                        )
                    )
                )
            )
            for i in range(self._Mp_j2)
        }
        funcs_dj2_dy = {
            i: (
                lambda x, y, theta, i=i: (
                    0.5
                    * (
                        (
                            self._dpsi_j2_YV[i](x, y)
                            * feats_j2[i](x, y, theta)
                            + self._psi_j2_XYV[i](x, y)
                            * jacrev(feats_j2[i], 1)(x, y, theta).squeeze()
                        )
                        - (
                            self._dpsi_j2_YV[i](x, y)
                            * feats_j2[i](x, y, jnp.pi + theta)
                            + self._psi_j2_XYV[i](x, y)
                            * jacrev(feats_j2[i], 1)(
                                x, y, jnp.pi + theta
                            ).squeeze()
                        )
                    )
                )
            )
            for i in range(self._Mp_j2)
        }
        funcs_vdj2 = {
            i: (
                lambda x, y, theta, i=i: (
                    jnp.cos(theta - jnp.pi / 2) * funcs_dj2_dx[i](x, y, theta)
                    + jnp.sin(theta - jnp.pi / 2)
                    * funcs_dj2_dy[i](x, y, theta)
                )
            )
            for i in range(self._Mp_j2)
        }
        funcs_dr2_dx = {
            i: (
                lambda x, y, theta, i=i: (
                    0.5
                    * (
                        (
                            self._dpsi_r2_XV[i](x, y)
                            * feats_r2[i](x, y, theta)
                            + self._psi_r2_XYV[i](x, y)
                            * jacrev(feats_r2[i], 0)(x, y, theta).squeeze()
                        )
                        + (
                            self._dpsi_r2_XV[i](x, y)
                            * feats_r2[i](x, y, jnp.pi + theta)
                            + self._psi_r2_XYV[i](x, y)
                            * jacrev(feats_r2[i], 0)(
                                x, y, jnp.pi + theta
                            ).squeeze()
                        )
                    )
                )
            )
            for i in range(self._Mp_r2)
        }
        funcs_dr2_dy = {
            i: (
                lambda x, y, theta, i=i: (
                    0.5
                    * (
                        (
                            self._dpsi_r2_YV[i](x, y)
                            * feats_r2[i](x, y, theta)
                            + self._psi_r2_XYV[i](x, y)
                            * jacrev(feats_r2[i], 1)(x, y, theta).squeeze()
                        )
                        + (
                            self._dpsi_r2_YV[i](x, y)
                            * feats_r2[i](x, y, jnp.pi + theta)
                            + self._psi_r2_XYV[i](x, y)
                            * jacrev(feats_r2[i], 1)(
                                x, y, jnp.pi + theta
                            ).squeeze()
                        )
                    )
                )
            )
            for i in range(self._Mp_r2)
        }
        funcs_vdr2 = {
            i: (
                lambda x, y, theta, i=i: (
                    jnp.cos(theta - jnp.pi / 2) * funcs_dr2_dx[i](x, y, theta)
                    + jnp.sin(theta - jnp.pi / 2)
                    * funcs_dr2_dy[i](x, y, theta)
                )
            )
            for i in range(self._Mp_r2)
        }

        vector_j2 = jnp.array(
            tree_map_funcs(funcs_j2, (x, y, theta_2))
        ).flatten()
        vector_r2 = jnp.array(
            tree_map_funcs(funcs_r2, (x, y, theta_2))
        ).flatten()
        vector_vdj2 = jnp.array(
            tree_map_funcs(funcs_vdj2, (x, y, theta_2))
        ).flatten()

        vector_vdr2 = jnp.array(
            tree_map_funcs(funcs_vdr2, (x, y, theta_2))
        ).flatten()

        vector_aver_vdj2 = (
            jnp.array(
                list(
                    tree_map(
                        lambda fn: self._integrator(
                            fn=fn,
                            quadratures=(self._pts, self._ws),
                            argnum=2,
                        )(x, y),
                        funcs_vdj2,
                    ).values()
                )
            )
            .squeeze()
            .flatten()
            / jnp.pi
        )
        vector_aver_r2 = (
            jnp.array(
                list(
                    tree_map(
                        lambda fn: self._integrator(
                            fn=fn,
                            quadratures=(self._pts, self._ws),
                            argnum=2,
                        )(x, y),
                        funcs_r2,
                    ).values()
                )
            )
            .squeeze()
            .flatten()
            / jnp.pi
        )

        eqn_residual = jnp.zeros(
            (
                3,
                self._Mp_j1 * self._jn_j1
                + self._Mp_r1 * self._jn_r1
                + self._Mp_j2 * self._jn_j2
                + self._Mp_r2 * self._jn_r2,
            )
        )
        eqn_residual = eqn_residual.at[0, : self._Mp_j1 * self._jn_j1].set(
            vector_aver_vdj1
        )
        eqn_residual = eqn_residual.at[
            0,
            self._Mp_j1 * self._jn_j1 : self._Mp_j1 * self._jn_j1
            + self._Mp_r1 * self._jn_r1,
        ].set(self._sigma_a(x, y) * vector_aver_r1)
        eqn_residual = eqn_residual.at[
            0,
            self._Mp_j1 * self._jn_j1
            + self._Mp_r1 * self._jn_r1 : self._Mp_j1 * self._jn_j1
            + self._Mp_r1 * self._jn_r1
            + self._Mp_j2 * self._jn_j2,
        ].set(vector_aver_vdj2)
        eqn_residual = eqn_residual.at[
            0,
            self._Mp_j1 * self._jn_j1
            + self._Mp_r1 * self._jn_r1
            + self._Mp_j2 * self._jn_j2 :,
        ].set(self._sigma_a(x, y) * vector_aver_r2)

        # j1, r1部分
        eqn_residual_1 = jnp.zeros(
            (
                3,
                self._Mp_j1 * self._jn_j1 + self._Mp_r1 * self._jn_r1,
            )
        )

        # 第一行
        eqn_residual_1 = eqn_residual_1.at[0, : self._Mp_j1 * self._jn_j1].set(
            vector_aver_vdj1
        )
        eqn_residual_1 = eqn_residual_1.at[0, self._Mp_j1 * self._jn_j1 :].set(
            self._sigma_a(x, y) * vector_aver_r1
        )

        # 第二行
        eqn_residual_1 = eqn_residual_1.at[1, : self._Mp_j1 * self._jn_j1].set(
            self.kn**2 * (vector_vdj1 - vector_aver_vdj1)
        )
        eqn_residual_1 = eqn_residual_1.at[1, self._Mp_j1 * self._jn_j1 :].set(
            (self._sigma_s(x, y) + self.kn**2 * self._sigma_a(x, y))
            * (vector_r1 - vector_aver_r1)
        )

        # 第三行
        eqn_residual_1 = eqn_residual_1.at[2, : self._Mp_j1 * self._jn_j1].set(
            (self._sigma_s(x, y) + self.kn**2 * self._sigma_a(x, y))
            * vector_j1
        )
        eqn_residual_1 = eqn_residual_1.at[2, self._Mp_j1 * self._jn_j1 :].set(
            vector_vdr1
        )

        # j2, r2部分
        eqn_residual_2 = jnp.zeros(
            (
                3,
                self._Mp_j2 * self._jn_j2 + self._Mp_r2 * self._jn_r2,
            )
        )

        # 第一行
        eqn_residual_2 = eqn_residual_2.at[0, : self._Mp_j2 * self._jn_j2].set(
            vector_aver_vdj2
        )
        eqn_residual_2 = eqn_residual_2.at[0, self._Mp_j2 * self._jn_j2 :].set(
            self._sigma_a(x, y) * vector_aver_r2
        )

        # 第二行
        eqn_residual_2 = eqn_residual_2.at[1, : self._Mp_j2 * self._jn_j2].set(
            self.kn**2 * (vector_vdj2 - vector_aver_vdj2)
        )
        eqn_residual_2 = eqn_residual_2.at[1, self._Mp_j2 * self._jn_j2 :].set(
            (self._sigma_s(x, y) + self.kn**2 * self._sigma_a(x, y))
            * (vector_r2 - vector_aver_r2)
        )

        # 第三行
        eqn_residual_2 = eqn_residual_2.at[2, : self._Mp_j2 * self._jn_j2].set(
            (self._sigma_s(x, y) + self.kn**2 * self._sigma_a(x, y))
            * vector_j2
        )
        eqn_residual_2 = eqn_residual_2.at[2, self._Mp_j2 * self._jn_j2 :].set(
            vector_vdr2
        )

        # 拼接两个部分
        eqn_residual = jnp.concatenate(
            [eqn_residual_1, eqn_residual_2], axis=1
        )
        return eqn_residual

    def _integrator(
        self,
        fn: Callable,
        quadratures: Tuple[jnp.ndarray, jnp.ndarray],
        argnum: int,
    ) -> Callable:
        """Integrate with respect to theta_1."""
        pts, ws = quadratures

        def integral_fn(*args):
            args = list(args)
            in_axes_ = [None] * len(args)
            args.insert(argnum, pts)
            in_axes_.insert(argnum, int(0))
            vmap_fns = vmap(fn, in_axes=in_axes_, out_axes=-1)(*args)

            out = [
                jnp.dot(vmap_fns[i], ws).squeeze() for i in range(self._jn_r1)
            ]
            return jnp.array(out)

        return integral_fn

    def normalize_pou_fn(self, kind="j"):
        # 根据 kind 选择 mesh
        if kind == "j":
            center = self._center_j1
            radius = self._radius_j1
            Mp = self._Mp_j1
        else:
            center = self._center_r1
            radius = self._radius_r1
            Mp = self._Mp_r1

        def psi_sym(x, y, i):

            return self._psi((x - center(i)[0]) / radius[0]) * self._psi(
                (y - center(i)[1]) / radius[1]
            )

        def dpsi_dx_sym(x, y, i):
            return (
                self._dpsi((x - center(i)[0]) / radius[0])
                / radius[0]
                * self._psi((y - center(i)[1]) / radius[1])
            )

        def dpsi_dy_sym(x, y, i):
            return (
                self._dpsi((y - center(i)[1]) / radius[1])
                / radius[1]
                * self._psi((x - center(i)[0]) / radius[0])
            )

        def psi_sum(x, y):
            return jnp.sum(vmap(lambda i: psi_sym(x, y, i))(jnp.arange(Mp)))

        def dpsi_dx_sum(x, y):
            return jnp.sum(
                vmap(lambda i: dpsi_dx_sym(x, y, i))(jnp.arange(Mp))
            )

        def dpsi_dy_sum(x, y):
            return jnp.sum(
                vmap(lambda i: dpsi_dy_sym(x, y, i))(jnp.arange(Mp))
            )

        psi_list = [
            lambda x, y, i=i: psi_sym(x, y, i) / psi_sum(x, y)
            for i in range(Mp)
        ]
        dpsi_dx_list = [
            lambda x, y, i=i: (
                (
                    dpsi_dx_sym(x, y, i) * psi_sum(x, y)
                    - dpsi_dx_sum(x, y) * psi_sym(x, y, i)
                )
                / psi_sum(x, y) ** 2
            )
            for i in range(Mp)
        ]
        dpsi_dy_list = [
            lambda x, y, i=i: (
                (
                    dpsi_dy_sym(x, y, i) * psi_sum(x, y)
                    - dpsi_dy_sum(x, y) * psi_sym(x, y, i)
                )
                / psi_sum(x, y) ** 2
            )
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
        self._center_j1 = self._mesh_j1.center_of_cell
        self._radius_j1 = self._mesh_j1.radius_of_cell
        self._Mp_j1 = self._mesh_j1.number_of_cells
        self._mesh_r1 = self._model_r1._mesh
        self._center_r1 = self._mesh_r1.center_of_cell
        self._radius_r1 = self._mesh_r1.radius_of_cell
        self._Mp_r1 = self._mesh_r1.number_of_cells
        self._mesh_j2 = self._model_j2._mesh
        self._center_j2 = self._mesh_j2.center_of_cell
        self._radius_j2 = self._mesh_j2.radius_of_cell
        self._Mp_j2 = self._mesh_j2.number_of_cells
        self._mesh_r2 = self._model_r2._mesh
        self._center_r2 = self._mesh_r2.center_of_cell
        self._radius_r2 = self._mesh_r2.radius_of_cell
        self._Mp_r2 = self._mesh_r2.number_of_cells

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_y: jnp.ndarray,
        invar_theta: jnp.ndarray,
    ) -> jnp.ndarray:
        x, y, theta_1 = invar_x, invar_y, invar_theta
        theta_2 = theta_1 + jnp.pi * 0.5
        theta_3 = theta_1 + jnp.pi
        theta_4 = theta_1 + jnp.pi * 1.5

        feats_j1_pos = self._feats_fn_j1(x, y, theta_1)
        feats_j1_neg = self._feats_fn_j1(x, y, theta_3)
        feats_r1_pos = self._feats_fn_r1(x, y, theta_1)
        feats_r1_neg = self._feats_fn_r1(x, y, theta_3)
        feats_j1 = 0.5 * (feats_j1_pos - feats_j1_neg)
        feats_r1 = 0.5 * (feats_r1_pos + feats_r1_neg)

        feats_j2_pos = self._feats_fn_j2(x, y, theta_2)
        feats_j2_neg = self._feats_fn_j2(x, y, theta_4)
        feats_r2_pos = self._feats_fn_r2(x, y, theta_2)
        feats_r2_neg = self._feats_fn_r2(x, y, theta_4)
        feats_j2 = 0.5 * (feats_j2_pos - feats_j2_neg)
        feats_r2 = 0.5 * (feats_r2_pos + feats_r2_neg)

        vector_j1 = (
            jnp.array([feats_j1[i] for i in range(self._Mp_j1)]).flatten()
            if self._Mp_j1 > 1
            else feats_j1.reshape(-1)
        )
        vector_r1 = (
            jnp.array([feats_r1[i] for i in range(self._Mp_r1)]).flatten()
            if self._Mp_r1 > 1
            else feats_r1.reshape(-1)
        )
        vector_j2 = (
            jnp.array([feats_j2[i] for i in range(self._Mp_j2)]).flatten()
            if self._Mp_j2 > 1
            else feats_j2.reshape(-1)
        )
        vector_r2 = (
            jnp.array([feats_r2[i] for i in range(self._Mp_r2)]).flatten()
            if self._Mp_r2 > 1
            else feats_r2.reshape(-1)
        )
        vector_f = jnp.concatenate(
            [
                self.kn * vector_j1,
                vector_r1,
                self.kn * vector_j2,
                vector_r2,
            ]
        )  # (Mp_j1*Jn_j1 + Mp_r1*Jn_r1 + Mp_j2*Jn_j2 + Mp_r2*Jn_r2, )
        return vector_f
