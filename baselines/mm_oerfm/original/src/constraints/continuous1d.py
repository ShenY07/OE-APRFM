import jax.numpy as jnp
from jax import random, vmap, jacrev
from utils.integrate import leggauss
from utils.parallel import tree_map_funcs
from collections.abc import Callable
from typing import Dict, Tuple
from modules.pou_func import psi, dpsi
from modules.func_space import (
    RandomFeatureSpaceX,
    RandomFeatureSpaceXV,
)
from functools import partial
from jax.tree_util import tree_map


class MicroMacroPointwiseInteriorConstraint1D(
    RandomFeatureSpaceX, RandomFeatureSpaceXV
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
        self._quadratures = leggauss(self.num_quads)
        self._pts, self._ws = self._quadratures
        self._model_rho = RandomFeatureSpaceX(
            self.domain,
            self.strides,
            self.Jn["rho"],
            self.scale,
            self.activation,
        )
        self._model_g = RandomFeatureSpaceXV(
            self.domain,
            self.strides,
            self.Jn["g"],
            self.scale,
            self.activation,
        )
        self._dim = 2
        self._dummy_x, self._dummy_v = jnp.split(
            jnp.empty((self._dim,)), self._dim, axis=0
        )
        self._params_rho = self._model_rho.init(self.init_rng, self._dummy_x)
        self._params_g = self._model_g.init(
            self.init_rng, self._dummy_x, self._dummy_v
        )
        self._feats_fn_rho = partial(
            self._model_rho.apply, self._params_rho, has_aux=True
        )
        self._feats_fn_g = partial(
            self._model_g.apply, self._params_g, has_aux=True
        )
        self._mesh_rho = self._model_rho._mesh
        self._mesh_g = self._model_g._mesh
        self._center_rho = self._mesh_rho.center_of_cell
        self._center_g = self._mesh_g.center_of_cell
        self._radius_rho = self._mesh_rho.radius_of_cell
        self._radius_g = self._mesh_g.radius_of_cell
        self._Mp_rho = self._mesh_rho.number_of_cells
        self._Mp_g = self._mesh_g.number_of_cells
        self._Jn_rho = self.Jn["rho"]
        self._Jn_g = self.Jn["g"]
        self._sigma_s = self.coeff_fns["scattering"]
        self._sigma_a = self.coeff_fns["absorption"]

        self._psi = psi
        self._dpsi = dpsi
        self._psiX, self._dpsiX = self.normalize_rho_pou_fn()
        self._psiXV, self._dpsiXV = self.normalize_g_pou_fn()

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_v: jnp.ndarray,
    ) -> jnp.ndarray:
        _, feat_operators_rho = self._feats_fn_rho(invar_x)
        _, feat_operators_g = self._feats_fn_g(invar_x, invar_v)
        x, v = invar_x, invar_v  # (1, ) (1, )
        feats_rho = feat_operators_rho
        feats_g = feat_operators_g
        # funcs_pou = {
        #     f"pou_{i}": lambda x, v, i=i: self._psiXV[i](x, v)
        #     for i in range(self._Mp_g)
        # }
        # vector_pou = jnp.array(tree_map_funcs(funcs_pou, (x, v))).flatten()
        funcs_drho_dx = {
            f"drho_dx_{i}": lambda x, i=i: self._dpsiX[i](x) * feats_rho[i](x)
            + self._psiX[i](x) * jacrev(feats_rho[i], 0)(x).squeeze()
            for i in range(self._Mp_rho)
        }
        funcs_rho = {
            f"rho_{i}": lambda x, i=i: self._psiX[i](x) * feats_rho[i](x)
            for i in range(self._Mp_rho)
        }
        funcs_dg_dx = {
            f"dg_dx_{i}": lambda x, v, i=i: self._dpsiXV[i](x, v)
            * feats_g[i](x, v)
            + self._psiXV[i](x, v) * jacrev(feats_g[i], 0)(x, v).squeeze()
            for i in range(self._Mp_g)
        }
        funcs_g = {
            f"g_{i}": lambda x, v, i=i: self._psiXV[i](x, v) * feats_g[i](x, v)
            for i in range(self._Mp_g)
        }
        # print(f"{jnp.array(tree_map_funcs(funcs_rho, (x,))).shape}")
        # print(f"{jnp.array(tree_map_funcs(funcs_g, (x, v))).shape}")
        vector_rho = jnp.array(tree_map_funcs(funcs_rho, (x,))).flatten()
        vector_g = jnp.array(tree_map_funcs(funcs_g, (x, v))).flatten()
        vector_drho_dx = jnp.array(
            tree_map_funcs(funcs_drho_dx, (x,))
        ).flatten()
        vector_dg_dx = jnp.array(tree_map_funcs(funcs_dg_dx, (x, v))).flatten()

        # Integral term - Integrate with respect to v.
        vector_aver_dvg_dx = (
            jnp.array(
                list(
                    tree_map(
                        lambda fn: self._integrator(
                            fn=fn,
                            quadratures=(self._pts, self._ws * self._pts),
                            argnum=1,
                        )(x),
                        funcs_dg_dx,
                    ).values()
                )
            )
            .squeeze()
            .flatten()
            / 2.0
        )
        vector_aver_g = (
            jnp.array(
                list(
                    tree_map(
                        lambda fn: self._integrator(
                            fn=fn,
                            quadratures=(self._pts, self._ws),
                            argnum=1,
                        )(x),
                        funcs_g,
                    ).values()
                )
            )
            .squeeze()
            .flatten()
            / 2.0
        )

        eqn_residual = jnp.zeros(
            (3, self._Mp_rho * self._Jn_rho + self._Mp_g * self._Jn_g)
        )
        eqn_residual = eqn_residual.at[0, : self._Mp_rho * self._Jn_rho].set(
            self._sigma_a(x) * vector_rho
        )
        eqn_residual = eqn_residual.at[0, self._Mp_rho * self._Jn_rho :].set(
            vector_aver_dvg_dx
        )
        eqn_residual = eqn_residual.at[1, : self._Mp_rho * self._Jn_rho].set(
            v * vector_drho_dx
        )
        eqn_residual = eqn_residual.at[1, self._Mp_rho * self._Jn_rho :].set(
            self.kn * (v * vector_dg_dx - vector_aver_dvg_dx)
            - self._sigma_s(x) * (0.0 - vector_g)
            + self.kn**2 * self._sigma_a(x) * vector_g
        )
        eqn_residual = eqn_residual.at[2, self._Mp_rho * self._Jn_rho :].set(
            vector_aver_g
        )  # (3, Mp_rho*Jn_rho + Mp_g*Jn_g)

        return eqn_residual

    def _integrator(
        self,
        fn: Callable,
        quadratures: Tuple[jnp.ndarray, jnp.ndarray],
        argnum: int,
    ) -> Callable:
        """Integrate with respect to v."""

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
            f"psi_{i}": lambda x, i=i: self._psi(
                (x - self._center_rho(i)) / self._radius_rho
            )
            for i in range(self._Mp_rho)
        }
        funcs_dpsi = {
            f"dpsi_{i}": lambda x, i=i: self._dpsi(
                (x - self._center_rho(i)) / self._radius_rho
            )
            / self._radius_rho
            for i in range(self._Mp_rho)
        }

        def psi_sum(x):
            return jnp.sum(jnp.array(tree_map_funcs(funcs_psi, x)))

        def dpsi_sum(x):
            return jnp.sum(jnp.array(tree_map_funcs(funcs_dpsi, x)))

        psi_list = [
            lambda x, i=i: self._psi(
                (x - self._center_rho(i)) / self._radius_rho
            )
            / psi_sum(x)
            for i in range(self._Mp_rho)
        ]
        dpsi_list = [
            lambda x, i=i: (
                self._dpsi((x - self._center_rho(i)) / self._radius_rho)
                / self._radius_rho
                * psi_sum(x)
                - dpsi_sum(x)
                * self._psi((x - self._center_rho(i)) / self._radius_rho)
            )
            / psi_sum(x) ** 2
            for i in range(self._Mp_rho)
        ]
        return psi_list, dpsi_list

    def normalize_g_pou_fn(self):
        funcs_psi = {
            f"psi_{i}": lambda x, v, i=i: self._psi(
                (x - self._center_g(i)[0]) / self._radius_g[0]
            )
            * self._psi((v - self._center_g(i)[-1]) / self._radius_g[-1])
            for i in range(self._Mp_g)
        }
        funcs_dpsi = {
            f"dpsi_{i}": lambda x, v, i=i: self._dpsi(
                (x - self._center_g(i)[0]) / self._radius_g[0]
            )
            / self._radius_g[0]
            * self._psi((v - self._center_g(i)[-1]) / self._radius_g[-1])
            for i in range(self._Mp_g)
        }

        def psi_sum(x, v):
            return jnp.sum(jnp.array(tree_map_funcs(funcs_psi, (x, v))))

        def dpsi_sum(x, v):
            return jnp.sum(jnp.array(tree_map_funcs(funcs_dpsi, (x, v))))

        psi_list = [
            lambda x, v, i=i: self._psi(
                (x - self._center_g(i)[0]) / self._radius_g[0]
            )
            * self._psi((v - self._center_g(i)[-1]) / self._radius_g[-1])
            / psi_sum(x, v)
            for i in range(self._Mp_g)
        ]
        dpsi_list = [
            lambda x, v, i=i: (
                self._dpsi((x - self._center_g(i)[0]) / self._radius_g[0])
                / self._radius_g[0]
                * self._psi((v - self._center_g(i)[-1]) / self._radius_g[-1])
                * psi_sum(x, v)
                - dpsi_sum(x, v)
                * self._psi((x - self._center_g(i)[0]) / self._radius_g[0])
                * self._psi((v - self._center_g(i)[-1]) / self._radius_g[-1])
            )
            / psi_sum(x, v) ** 2
            for i in range(self._Mp_g)
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
        self._quadratures = leggauss(self.num_quads)
        self._pts, self._ws = self._quadratures
        self._model_f = RandomFeatureSpaceXV(
            self.domain,
            self.strides,
            self.Jn["f"],
            self.scale,
            self.activation,
        )
        self._dim = 2
        self._dummy_x, self._dummy_v = jnp.split(
            jnp.empty((self._dim,)), self._dim, axis=0
        )
        self._params_f = self._model_f.init(
            self.init_rng, self._dummy_x, self._dummy_v
        )
        self._feats_fn_f = partial(
            self._model_f.apply, self._params_f, has_aux=True
        )
        self._mesh_f = self._model_f._mesh
        self._center_f = self._mesh_f.center_of_cell
        self._radius_f = self._mesh_f.radius_of_cell
        self._Mp_f = self._mesh_f.number_of_cells
        self._Jn_f = self.Jn["f"]
        self._sigma_s = self.coeff_fns["scattering"]
        self._sigma_a = self.coeff_fns["absorption"]

        self._psi = psi
        self._dpsi = dpsi
        self._psiXV, self._dpsiXV = self.normalize_f_pou_fn()

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_v: jnp.ndarray,
    ) -> jnp.ndarray:
        _, feat_operators_f = self._feats_fn_f(invar_x, invar_v)
        x, v = invar_x, invar_v  # (1, ) (1, )
        feats_f = feat_operators_f
        funcs_df_dx = {
            f"df_dx_{i}": lambda x, v, i=i: self._dpsiXV[i](x, v)
            * feats_f[i](x, v)
            + self._psiXV[i](x, v) * jacrev(feats_f[i], 0)(x, v).squeeze()
            for i in range(self._Mp_f)
        }
        funcs_f = {
            f"f_{i}": lambda x, v, i=i: self._psiXV[i](x, v) * feats_f[i](x, v)
            for i in range(self._Mp_f)
        }
        vector_f = jnp.array(tree_map_funcs(funcs_f, (x, v))).flatten()
        vector_df_dx = jnp.array(tree_map_funcs(funcs_df_dx, (x, v))).flatten()
        vector_aver_f = (
            jnp.array(
                list(
                    tree_map(
                        lambda fn: self._integrator(
                            fn=fn,
                            quadratures=(self._pts, self._ws),
                            argnum=1,
                        )(x),
                        funcs_f,
                    ).values()
                )
            )
            .squeeze()
            .flatten()
            / 2.0
        )
        eqn_residual = jnp.zeros((1, self._Mp_f * self._Jn_f))
        eqn_residual = eqn_residual.at[0, : self._Mp_f * self._Jn_f].set(
            self.kn * v * vector_df_dx
            - (
                self._sigma_s(x) * (vector_aver_f - vector_f)
                - self.kn**2 * self._sigma_a(x) * vector_f
            )
        )
        return eqn_residual

    def _integrator(
        self,
        fn: Callable,
        quadratures: Tuple[jnp.ndarray, jnp.ndarray],
        argnum: int,
    ) -> Callable:
        """Integrate with respect to v."""

        pts, ws = quadratures

        def integral_fn(*args):
            args = list(args)
            in_axes_ = [None] * len(args)
            args.insert(argnum, pts)
            in_axes_.insert(argnum, int(0))
            vmap_fns = vmap(fn, in_axes=in_axes_, out_axes=-1)(*args)
            out = [
                jnp.dot(vmap_fns[i], ws).squeeze() for i in range(self._Jn_f)
            ]
            return jnp.array(out)

        return integral_fn

    def normalize_f_pou_fn(self):
        funcs_psi = {
            f"psi_{i}": lambda x, v, i=i: self._psi(
                (x - self._center_f(i)[0]) / self._radius_f[0]
            )
            * self._psi((v - self._center_f(i)[-1]) / self._radius_f[-1])
            for i in range(self._Mp_f)
        }
        funcs_dpsi = {
            f"dpsi_{i}": lambda x, v, i=i: self._dpsi(
                (x - self._center_f(i)[0]) / self._radius_f[0]
            )
            / self._radius_f[0]
            * self._psi((v - self._center_f(i)[-1]) / self._radius_f[-1])
            for i in range(self._Mp_f)
        }

        def psi_sum(x, v):
            return jnp.sum(jnp.array(tree_map_funcs(funcs_psi, (x, v))))

        def dpsi_sum(x, v):
            return jnp.sum(jnp.array(tree_map_funcs(funcs_dpsi, (x, v))))

        psi_list = [
            lambda x, v, i=i: self._psi(
                (x - self._center_f(i)[0]) / self._radius_f[0]
            )
            * self._psi((v - self._center_f(i)[-1]) / self._radius_f[-1])
            / psi_sum(x, v)
            for i in range(self._Mp_f)
        ]
        dpsi_list = [
            lambda x, v, i=i: (
                self._dpsi((x - self._center_f(i)[0]) / self._radius_f[0])
                / self._radius_f[0]
                * self._psi((v - self._center_f(i)[-1]) / self._radius_f[-1])
                * psi_sum(x, v)
                - dpsi_sum(x, v)
                * self._psi((x - self._center_f(i)[0]) / self._radius_f[0])
                * self._psi((v - self._center_f(i)[-1]) / self._radius_f[-1])
            )
            / psi_sum(x, v) ** 2
            for i in range(self._Mp_f)
        ]
        return psi_list, dpsi_list


class MicroMacroPointwiseBoundaryConstraint1D(
    RandomFeatureSpaceX, RandomFeatureSpaceXV
):
    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: int
    scale: float
    init_rng: random.PRNGKey
    kn: float
    activation: Callable

    def setup(self):
        self._model_rho = RandomFeatureSpaceX(
            self.domain,
            self.strides,
            self.Jn["rho"],
            self.scale,
            self.activation,
        )
        self._model_g = RandomFeatureSpaceXV(
            self.domain,
            self.strides,
            self.Jn["g"],
            self.scale,
            self.activation,
        )
        self._dim = 2
        self._dummy_x, self._dummy_v = jnp.split(
            jnp.empty((self._dim,)), self._dim, axis=0
        )
        self._params_rho = self._model_rho.init(self.init_rng, self._dummy_x)
        self._params_g = self._model_g.init(
            self.init_rng, self._dummy_x, self._dummy_v
        )
        self._feats_fn_rho = partial(self._model_rho.apply, self._params_rho)
        self._feats_fn_g = partial(self._model_g.apply, self._params_g)
        self._mesh_rho = self._model_rho._mesh
        self._mesh_g = self._model_g._mesh
        self._center_rho = self._mesh_rho.center_of_cell
        self._center_g = self._mesh_g.center_of_cell
        self._radius_rho = self._mesh_rho.radius_of_cell
        self._radius_g = self._mesh_g.radius_of_cell
        self._Mp_rho = self._mesh_rho.number_of_cells
        self._Mp_g = self._mesh_g.number_of_cells

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_v: jnp.ndarray,
    ) -> jnp.ndarray:
        x, v = invar_x, invar_v
        feats_rho = self._feats_fn_rho(x)
        feats_g = self._feats_fn_g(x, v)
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


class PointwiseBoundaryConstraint1D(RandomFeatureSpaceXV):
    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: int
    scale: float
    init_rng: random.PRNGKey
    kn: float
    activation: Callable

    def setup(self):
        self._model_f = RandomFeatureSpaceXV(
            self.domain,
            self.strides,
            self.Jn["f"],
            self.scale,
            self.activation,
        )
        self._dim = 2
        self._dummy_x, self._dummy_v = jnp.split(
            jnp.empty((self._dim,)), self._dim, axis=0
        )
        self._params_f = self._model_f.init(
            self.init_rng, self._dummy_x, self._dummy_v
        )
        self._feats_fn_f = partial(self._model_f.apply, self._params_f)
        self._mesh_f = self._model_f._mesh
        self._center_f = self._mesh_f.center_of_cell
        self._radius_f = self._mesh_f.radius_of_cell
        self._Mp_f = self._mesh_f.number_of_cells

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_v: jnp.ndarray,
    ) -> jnp.ndarray:
        x, v = invar_x, invar_v
        feats_f = self._feats_fn_f(x, v)
        vector_f = (
            jnp.array([feats_f[i] for i in range(self._Mp_f)]).flatten()
            if self._Mp_f > 1
            else feats_f.reshape(-1)
        )  # (Mp_rho*Jn_rho + Mp_g*Jn_g, )
        return vector_f
