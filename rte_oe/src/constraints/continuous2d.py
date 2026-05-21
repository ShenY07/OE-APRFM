import jax.numpy as jnp
from jax import random, vmap, jacrev
from utils.integrate import leggauss
from utils.parallel import tree_map_funcs
from collections.abc import Callable
from typing import Dict, Tuple
from modules.pou_func import (psi_b as psi, dpsi_b as dpsi)
from modules.func_space import (
    RandomFeatureSpaceXY,
    RandomFeatureSpaceXYV,
)
from functools import partial
from jax.tree_util import tree_map


class MicroMacroPointwiseInteriorConstraint2D(
    RandomFeatureSpaceXY, RandomFeatureSpaceXYV
):
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
        self._quadratures = leggauss(
            self.num_quads, interval=(0.0, 2.0 * jnp.pi)
        )
        self._pts, self._ws = self._quadratures
        self._model_rho = RandomFeatureSpaceXY(
            self.domain,
            self.strides,
            self.Jn["rho"],
            self.scale,
            self.activation,
        )
        self._model_g = RandomFeatureSpaceXYV(
            self.domain,
            self.strides,
            self.Jn["g"],
            self.scale,
            self.activation,
        )
        self._dim = 3
        self._dummy_x, self._dummy_y, self._dummy_theta = jnp.split(
            jnp.empty((self._dim,)), self._dim, axis=0
        )
        self._params_rho = self._model_rho.init(
            self.init_rng, self._dummy_x, self._dummy_y
        )
        self._params_g = self._model_g.init(
            self.init_rng, self._dummy_x, self._dummy_y, self._dummy_theta
        )
        self._feats_fn_rho = partial(
            self._model_rho.apply, self._params_rho, has_aux=True
        )
        self._feats_fn_g = partial(
            self._model_g.apply, self._params_g, has_aux=True
        )
        self._mesh_rho = self._model_rho._mesh
        self._center_rho = self._mesh_rho.center_of_cell
        self._radius_rho = self._mesh_rho.radius_of_cell
        self._Mp_rho = self._mesh_rho.number_of_cells
        self._mesh_g = self._model_g._mesh
        self._center_g = self._mesh_g.center_of_cell
        self._radius_g = self._mesh_g.radius_of_cell
        self._Mp_g = self._mesh_g.number_of_cells
        self._Jn_rho = self.Jn["rho"]
        self._Jn_g = self.Jn["g"]
        self._sigma_s = self.coeff_fns["scattering"]
        self._sigma_a = self.coeff_fns["absorption"]

        self._psi = psi
        self._dpsi = dpsi
        self._psiXY, self._dpsiX, self._dpsiY = self.normalize_rho_pou_fn()
        self._psiXYV, self._dpsiXV, self._dpsiYV = self.normalize_g_pou_fn()

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_y: jnp.ndarray,
        invar_theta: jnp.ndarray,
    ) -> jnp.ndarray:
        _, feat_operators_rho = self._feats_fn_rho(invar_x, invar_y)
        _, feat_operators_g = self._feats_fn_g(invar_x, invar_y, invar_theta)
        x, y, theta = invar_x, invar_y, invar_theta
        v_x, v_y = (
            jnp.cos(theta),
            jnp.sin(theta),
        )
        feats_rho = feat_operators_rho
        feats_g = feat_operators_g
        # print("=============================")
        # funcs_pou = {
        #     f"pou_{i}": lambda x, y, i=i: self._psiXY[i](x, y)
        #     for i in range(self._Mp_rho)
        # }
        # vector_pou = jnp.array(tree_map_funcs(funcs_pou, (x, y))).flatten()
        # print("test: ", vector_pou)
        # print("=============================")
        # funcs_pou = {
        #     f"pou_{i}": lambda x, y, theta, i=i: self._psiXYV[i](x, y, theta)
        #     for i in range(self._Mp_g)
        # }
        # vector_pou = jnp.array(
        #     tree_map_funcs(funcs_pou, (x, y, theta))
        # ).flatten()
        # print("test: ", vector_pou)
        # print("=============================")

        funcs_drho_dx = [
            lambda x, y, i=i: self._dpsiX[i](x, y) * feats_rho[i](x, y)
            + self._psiXY[i](x, y) * jacrev(feats_rho[i], 0)(x, y).squeeze()
            for i in range(self._Mp_rho)
        ]

        funcs_drho_dy = [
            lambda x, y, i=i: self._dpsiY[i](x, y) * feats_rho[i](x, y)
            + self._psiXY[i](x, y) * jacrev(feats_rho[i], 1)(x, y).squeeze()
            for i in range(self._Mp_rho)
        ]

        funcs_dvrho = {
            f"dvrho_{i}": lambda x, y, i=i: v_x * funcs_drho_dx[i](x, y)
            + v_y * funcs_drho_dy[i](x, y)
            for i in range(self._Mp_rho)
        }

        funcs_rho = {
            f"rho_{i}": lambda x, y, i=i: self._psiXY[i](x, y)
            * feats_rho[i](x, y)
            for i in range(self._Mp_rho)
        }
        funcs_dg_dx = [
            lambda x, y, theta, i=i: self._dpsiXV[i](x, y, theta)
            * feats_g[i](x, y, theta)
            + self._psiXYV[i](x, y, theta)
            * jacrev(feats_g[i], 0)(x, y, theta).squeeze()
            for i in range(self._Mp_g)
        ]

        funcs_dg_dy = [
            lambda x, y, theta, i=i: self._dpsiYV[i](x, y, theta)
            * feats_g[i](x, y, theta)
            + self._psiXYV[i](x, y, theta)
            * jacrev(feats_g[i], 1)(x, y, theta).squeeze()
            for i in range(self._Mp_g)
        ]

        funcs_dvg = {
            f"dvg_{i}": lambda x, y, theta, i=i: jnp.cos(theta)
            * funcs_dg_dx[i](x, y, theta)
            + jnp.sin(theta) * funcs_dg_dy[i](x, y, theta)
            for i in range(self._Mp_g)
        }

        funcs_g = {
            f"g_{i}": lambda x, y, theta, i=i: self._psiXYV[i](x, y, theta)
            * feats_g[i](x, y, theta)
            for i in range(self._Mp_g)
        }

        vector_rho = jnp.array(tree_map_funcs(funcs_rho, (x, y))).flatten()
        vector_g = jnp.array(tree_map_funcs(funcs_g, (x, y, theta))).flatten()
        vector_dvrho = jnp.array(tree_map_funcs(funcs_dvrho, (x, y))).flatten()
        vector_dvg = jnp.array(
            tree_map_funcs(funcs_dvg, (x, y, theta))
        ).flatten()
        vector_aver_dvg = (
            jnp.array(
                list(
                    tree_map(
                        lambda fn: self._integrator(
                            fn=fn,
                            quadratures=(self._pts, self._ws),
                            argnum=2,
                        )(x, y),
                        funcs_dvg,
                    ).values()
                )
            )
            .squeeze()
            .flatten()
            / 2.0
            / jnp.pi
        )
        vector_aver_g = (
            jnp.array(
                list(
                    tree_map(
                        lambda fn: self._integrator(
                            fn=fn,
                            quadratures=(self._pts, self._ws),
                            argnum=2,
                        )(x, y),
                        funcs_g,
                    ).values()
                )
            )
            .squeeze()
            .flatten()
            / 2.0
            / jnp.pi
        )
        eqn_residual = jnp.zeros(
            (3, self._Mp_rho * self._Jn_rho + self._Mp_g * self._Jn_g)
        )
        eqn_residual = eqn_residual.at[0, : self._Mp_rho * self._Jn_rho].set(
            self._sigma_a(x, y) * vector_rho
        )
        eqn_residual = eqn_residual.at[0, self._Mp_rho * self._Jn_rho:].set(
            vector_aver_dvg
        )
        eqn_residual = eqn_residual.at[1, : self._Mp_rho * self._Jn_rho].set(
            vector_dvrho
        )
        eqn_residual = eqn_residual.at[1, self._Mp_rho * self._Jn_rho:].set(
            self.kn * (vector_dvg - vector_aver_dvg)
            + (self._sigma_s(x, y) + self.kn**2 * self._sigma_a(x, y))
            * vector_g
        )
        eqn_residual = eqn_residual.at[2, self._Mp_rho * self._Jn_rho:].set(
            vector_aver_g
        )
        return eqn_residual

    def _integrator(
        self,
        fn: Callable,
        quadratures: Tuple[jnp.ndarray, jnp.ndarray],
        argnum: int,
    ) -> Callable:
        """Integrate with respect to theta."""

        pts, ws = quadratures

        def integral_fn(*args):
            args = list(args)
            in_axes_ = [None] * len(args)
            args.insert(argnum, pts)
            in_axes_.insert(argnum, int(0))
            vmap_fns = vmap(fn, in_axes=in_axes_, out_axes=-1)(*args)
            out = [
                jnp.dot(vmap_fns[i], ws).squeeze() for i in range(self._Jn_g)
            ]
            return jnp.array(out)

        return integral_fn

    def normalize_rho_pou_fn(self):
        funcs_psi = {
            f"psi_{i}": lambda x, y, i=i: self._psi(
                (x - self._center_rho(i)[0]) / self._radius_rho[0]
            )
            * self._psi((y - self._center_rho(i)[1]) / self._radius_rho[1])
            for i in range(self._Mp_rho)
        }
        funcs_dpsi_dx = {
            f"dpsi_{i}": lambda x, y, i=i: self._dpsi(
                (x - self._center_rho(i)[0]) / self._radius_rho[0]
            )
            / self._radius_rho[0]
            * self._psi((y - self._center_rho(i)[1]) / self._radius_rho[1])
            for i in range(self._Mp_rho)
        }
        funcs_dpsi_dy = {
            f"dpsi_{i}": lambda x, y, i=i: self._psi(
                (x - self._center_rho(i)[0]) / self._radius_rho[0]
            )
            * self._dpsi((y - self._center_rho(i)[1]) / self._radius_rho[1])
            / self._radius_rho[1]
            for i in range(self._Mp_rho)
        }

        def psi_sum(x, y):
            return jnp.sum(jnp.array(tree_map_funcs(funcs_psi, (x, y))))

        def dpsi_dx_sum(x, y):
            return jnp.sum(jnp.array(tree_map_funcs(funcs_dpsi_dx, (x, y))))

        def dpsi_dy_sum(x, y):
            return jnp.sum(jnp.array(tree_map_funcs(funcs_dpsi_dy, (x, y))))

        psi_list = [
            lambda x, y, i=i: self._psi(
                (x - self._center_rho(i)[0]) / self._radius_rho[0]
            )
            * self._psi((y - self._center_rho(i)[1]) / self._radius_rho[1])
            / psi_sum(x, y)
            for i in range(self._Mp_rho)
        ]

        dpsi_dx_list = [
            lambda x, y, i=i: (
                self._dpsi((x - self._center_rho(i)[0]) / self._radius_rho[0])
                / self._radius_rho[0]
                * self._psi((y - self._center_rho(i)[1]) / self._radius_rho[1])
                * psi_sum(x, y)
                - dpsi_dx_sum(x, y)
                * self._psi((x - self._center_rho(i)[0]) / self._radius_rho[0])
                * self._psi((y - self._center_rho(i)[1]) / self._radius_rho[1])
            )
            / psi_sum(x, y) ** 2
            for i in range(self._Mp_rho)
        ]
        dpsi_dy_list = [
            lambda x, y, i=i: (
                self._dpsi((y - self._center_rho(i)[1]) / self._radius_rho[1])
                / self._radius_rho[1]
                * self._psi((x - self._center_rho(i)[0]) / self._radius_rho[0])
                * psi_sum(x, y)
                - dpsi_dy_sum(x, y)
                * self._psi((x - self._center_rho(i)[0]) / self._radius_rho[0])
                * self._psi((y - self._center_rho(i)[1]) / self._radius_rho[1])
            )
            / psi_sum(x, y) ** 2
            for i in range(self._Mp_rho)
        ]

        return psi_list, dpsi_dx_list, dpsi_dy_list

    def normalize_g_pou_fn(self):
        funcs_psi = {
            f"psi_{i}": lambda x, y, theta, i=i: self._psi(
                (x - self._center_g(i)[0]) / self._radius_g[0]
            )
            * self._psi((y - self._center_g(i)[1]) / self._radius_g[1])
            * self._psi((theta - self._center_g(i)[2]) / self._radius_g[2])
            for i in range(self._Mp_g)
        }
        funcs_dpsi_dx = {
            f"dpsi_{i}": lambda x, y, theta, i=i: self._dpsi(
                (x - self._center_g(i)[0]) / self._radius_g[0]
            )
            / self._radius_g[0]
            * self._psi((y - self._center_g(i)[1]) / self._radius_g[1])
            * self._psi((theta - self._center_g(i)[2]) / self._radius_g[2])
            for i in range(self._Mp_g)
        }
        funcs_dpsi_dy = {
            f"dpsi_{i}": lambda x, y, theta, i=i: self._dpsi(
                (y - self._center_g(i)[1]) / self._radius_g[1]
            )
            / self._radius_g[1]
            * self._psi((x - self._center_g(i)[0]) / self._radius_g[0])
            * self._psi((theta - self._center_g(i)[2]) / self._radius_g[2])
            for i in range(self._Mp_g)
        }

        def psi_sum(x, y, theta):
            return jnp.sum(jnp.array(tree_map_funcs(funcs_psi, (x, y, theta))))

        def dpsi_dx_sum(x, y, theta):
            return jnp.sum(
                jnp.array(tree_map_funcs(funcs_dpsi_dx, (x, y, theta)))
            )

        def dpsi_dy_sum(x, y, theta):
            return jnp.sum(
                jnp.array(tree_map_funcs(funcs_dpsi_dy, (x, y, theta)))
            )

        psi_list = [
            lambda x, y, theta, i=i: self._psi(
                (x - self._center_g(i)[0]) / self._radius_g[0]
            )
            * self._psi((y - self._center_g(i)[1]) / self._radius_g[1])
            * self._psi((theta - self._center_g(i)[2]) / self._radius_g[2])
            / psi_sum(x, y, theta)
            for i in range(self._Mp_g)
        ]

        dpsi_dx_list = [
            lambda x, y, theta, i=i: (
                self._dpsi((x - self._center_g(i)[0]) / self._radius_g[0])
                / self._radius_g[0]
                * self._psi((y - self._center_g(i)[1]) / self._radius_g[1])
                * self._psi((theta - self._center_g(i)[2]) / self._radius_g[2])
                * psi_sum(x, y, theta)
                - dpsi_dx_sum(x, y, theta)
                * self._psi((x - self._center_g(i)[0]) / self._radius_g[0])
                * self._psi((y - self._center_g(i)[1]) / self._radius_g[1])
                * self._psi((theta - self._center_g(i)[2]) / self._radius_g[2])
            )
            / psi_sum(x, y, theta) ** 2
            for i in range(self._Mp_g)
        ]

        dpsi_dy_list = [
            lambda x, y, theta, i=i: (
                self._dpsi((y - self._center_g(i)[1]) / self._radius_g[1])
                / self._radius_g[1]
                * self._psi((x - self._center_g(i)[0]) / self._radius_g[0])
                * self._psi((theta - self._center_g(i)[2]) / self._radius_g[2])
                * psi_sum(x, y, theta)
                - dpsi_dy_sum(x, y, theta)
                * self._psi((x - self._center_g(i)[0]) / self._radius_g[0])
                * self._psi((y - self._center_g(i)[1]) / self._radius_g[1])
                * self._psi((theta - self._center_g(i)[2]) / self._radius_g[2])
            )
            / psi_sum(x, y, theta) ** 2
            for i in range(self._Mp_g)
        ]
        return psi_list, dpsi_dx_list, dpsi_dy_list


class MicroMacroPointwiseBoundaryConstraint2D(
    RandomFeatureSpaceXY, RandomFeatureSpaceXYV
):
    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: int
    scale: float
    init_rng: random.PRNGKey
    kn: float
    activation: Callable

    def setup(self):
        self._model_rho = RandomFeatureSpaceXY(
            self.domain,
            self.strides,
            self.Jn["rho"],
            self.scale,
            self.activation,
        )
        self._model_g = RandomFeatureSpaceXYV(
            self.domain,
            self.strides,
            self.Jn["g"],
            self.scale,
            self.activation,
        )
        self._dim = 3
        self._dummy_x, self._dummy_y, self._dummy_theta = jnp.split(
            jnp.empty((self._dim,)), self._dim, axis=0
        )
        self._params_rho = self._model_rho.init(
            self.init_rng, self._dummy_x, self._dummy_y
        )
        self._params_g = self._model_g.init(
            self.init_rng, self._dummy_x, self._dummy_y, self._dummy_theta
        )
        self._feats_fn_rho = partial(self._model_rho.apply, self._params_rho)
        self._feats_fn_g = partial(self._model_g.apply, self._params_g)
        self._mesh_rho = self._model_rho._mesh
        self._center_rho = self._mesh_rho.center_of_cell
        self._radius_rho = self._mesh_rho.radius_of_cell
        self._Mp_rho = self._mesh_rho.number_of_cells
        self._mesh_g = self._model_g._mesh
        self._center_g = self._mesh_g.center_of_cell
        self._radius_g = self._mesh_g.radius_of_cell
        self._Mp_g = self._mesh_g.number_of_cells

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_y: jnp.ndarray,
        invar_theta: jnp.ndarray,
    ) -> jnp.ndarray:
        x, y, theta = invar_x, invar_y, invar_theta
        feats_rho = self._feats_fn_rho(x, y)
        feats_g = self._feats_fn_g(x, y, theta)
        vector_rho = (
            jnp.array([feats_rho[i] for i in range(self._Mp_rho)]).flatten()
            if self._Mp_rho > 1
            else feats_rho.reshape(-1)
        )
        vector_g = (
            jnp.array([feats_g[i] for i in range(self._Mp_g)]).flatten()
            if self._Mp_g > 1
            else feats_g.reshape(-1)
        )
        vector_f = jnp.concatenate(
            [vector_rho, self.kn * vector_g]
        )  # (Mp_rho*Jn_rho + Mp_g*Jn_g, )
        return vector_f
