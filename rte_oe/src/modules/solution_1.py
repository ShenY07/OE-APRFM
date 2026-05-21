from collections.abc import Callable
from functools import partial
from typing import Dict, Tuple

import jax.numpy as jnp
from jax import random

from modules.func_space_2 import (
    RandomFeatureSpaceXYV,
)


class OddEvenDecompositionConstructor2D(RandomFeatureSpaceXYV):
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
        feat_j_v = self._feats_fn_j(invar_x, invar_y, invar_theta)  # (Mp_j, Jn_j)
        feat_r_v = self._feats_fn_r(invar_x, invar_y, invar_theta)  # (Mp_r, Jn_r)

        feat_j_negv = self._feats_fn_j(invar_x, invar_y, -invar_theta)  # (Mp_j, Jn_j)
        feat_r_negv = self._feats_fn_r(invar_x, invar_y, -invar_theta)  # (Mp_r, Jn_r)

        feat_j_outvar = 0.5 * (feat_j_v - feat_j_negv)  # j_new
        feat_r_outvar = 0.5 * (feat_r_v + feat_r_negv)  # r_new

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


class OddEvenDecompositionConstructor2D_1(RandomFeatureSpaceXYV):
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
        feat_j_v = self._feats_fn_j(invar_x, invar_y, invar_theta)  # (Mp_j, Jn_j)
        feat_r_v = self._feats_fn_r(invar_x, invar_y, invar_theta)  # (Mp_r, Jn_r)

        feat_j_negv = self._feats_fn_j(
            invar_x, invar_y, -jnp.pi + invar_theta
        )  # (Mp_j, Jn_j)
        feat_r_negv = self._feats_fn_r(
            invar_x, invar_y, -jnp.pi + invar_theta
        )  # (Mp_r, Jn_r)

        feat_j_outvar = 0.5 * (feat_j_v - feat_j_negv)  # j_new
        feat_r_outvar = 0.5 * (feat_r_v + feat_r_negv)  # r_new

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


# inputs: (x, v):  x = xl, v > 0 or x = xr, v < 0
def bdy_fn(
    x: jnp.ndarray, v: jnp.ndarray, bdy_value: Tuple[jnp.ndarray, jnp.ndarray]
) -> jnp.ndarray:
    """Inflow boundary condition for x = xl (v>0) or x = xr (v<0)."""

    inflow_value = jnp.where(v > 0, bdy_value[0], jnp.where(v < 0, bdy_value[-1], 0.0))
    return inflow_value


# import jax.numpy as jnp
# from jax import random
# from collections.abc import Callable
# from typing import Dict, Tuple
# from modules.func_space import (
#     RandomFeatureSpaceX,
#     RandomFeatureSpaceXV,
#     RandomFeatureSpaceXY,
#     RandomFeatureSpaceXYV,
# )
# from functools import partial


# class MicroMacroConstructor1D(RandomFeatureSpaceX, RandomFeatureSpaceXV):
#     domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
#     strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
#     Jn: int
#     scale: float
#     init_rng: random.PRNGKey
#     kn: float
#     activation: Callable
#     coefficients: jnp.ndarray

#     def setup(self):
#         self._model_rho = RandomFeatureSpaceX(
#             self.domain,
#             self.strides,
#             self.Jn["rho"],
#             self.scale,
#             self.activation,
#         )
#         self._model_g = RandomFeatureSpaceXV(
#             self.domain,
#             self.strides,
#             self.Jn["g"],
#             self.scale,
#             self.activation,
#         )
#         self._dim = 2
#         self._dummy_x, self._dummy_v = jnp.split(
#             jnp.empty((self._dim,)), self._dim, axis=0
#         )
#         self._params_rho = self._model_rho.init(self.init_rng, self._dummy_x)
#         self._params_g = self._model_g.init(
#             self.init_rng, self._dummy_x, self._dummy_v
#         )
#         self._feats_fn_rho = partial(self._model_rho.apply, self._params_rho)
#         self._feats_fn_g = partial(self._model_g.apply, self._params_g)
#         self._mesh_rho = self._model_rho._mesh
#         self._mesh_g = self._model_g._mesh
#         self._Mp_rho = self._mesh_rho.number_of_cells
#         self._Mp_g = self._mesh_g.number_of_cells
#         self._Jn_rho = self.Jn["rho"]
#         self._Jn_g = self.Jn["g"]

#     def __call__(
#         self,
#         invar_x: jnp.ndarray,
#         invar_v: jnp.ndarray,
#     ) -> jnp.ndarray:
#         feat_rho_outvar = self._feats_fn_rho(invar_x)  # (Mp_rho, Jn_rho)
#         feat_g_outvar = self._feats_fn_g(invar_x, invar_v)  # (Mp_g, Jn_g)
#         coeff_rho = self.coefficients[: self._Mp_rho * self._Jn_rho].reshape(
#             self._Mp_rho, self._Jn_rho
#         )
#         coeff_g = self.coefficients[self._Mp_rho * self._Jn_rho :].reshape(
#             self._Mp_g, self._Jn_g
#         )
#         approx_rho = jnp.einsum(
#             "ij,ij->",
#             feat_rho_outvar
#             if self._Mp_rho > 1
#             else feat_rho_outvar.reshape(1, self._Jn_rho),
#             coeff_rho,
#         )
#         approx_g = jnp.einsum(
#             "ij,ij->",
#             feat_g_outvar
#             if self._Mp_g > 1
#             else feat_g_outvar.reshape(1, self._Jn_g),
#             coeff_g,
#         )
#         approx_solution = approx_rho + self.kn * approx_g
#         return approx_solution


# class Constructor1D(RandomFeatureSpaceXV):
#     domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
#     strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
#     Jn: int
#     scale: float
#     init_rng: random.PRNGKey
#     kn: float
#     activation: Callable
#     coefficients: jnp.ndarray

#     def setup(self):
#         self._model_f = RandomFeatureSpaceXV(
#             self.domain,
#             self.strides,
#             self.Jn["f"],
#             self.scale,
#             self.activation,
#         )
#         self._dim = 2
#         self._dummy_x, self._dummy_v = jnp.split(
#             jnp.empty((self._dim,)), self._dim, axis=0
#         )
#         self._params_f = self._model_f.init(
#             self.init_rng, self._dummy_x, self._dummy_v
#         )
#         self._feats_fn_f = partial(self._model_f.apply, self._params_f)
#         self._mesh_f = self._model_f._mesh
#         self._Mp_f = self._mesh_f.number_of_cells
#         self._Jn_f = self.Jn["f"]

#     def __call__(
#         self,
#         invar_x: jnp.ndarray,
#         invar_v: jnp.ndarray,
#     ) -> jnp.ndarray:
#         feat_f_outvar = self._feats_fn_f(invar_x, invar_v)
#         coeff_f = self.coefficients[: self._Mp_f * self._Jn_f].reshape(
#             self._Mp_f, self._Jn_f
#         )
#         approx_f = jnp.einsum(
#             "ij,ij->",
#             feat_f_outvar
#             if self._Mp_f > 1
#             else feat_f_outvar.reshape(1, self._Jn_f),
#             coeff_f,
#         )
#         approx_solution = approx_f
#         return approx_solution


# class MicroMacroConstructor2D(RandomFeatureSpaceXY, RandomFeatureSpaceXYV):
#     domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
#     strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
#     Jn: int
#     scale: float
#     init_rng: random.PRNGKey
#     kn: float
#     activation: Callable
#     coefficients: jnp.ndarray

#     def setup(self):
#         self._model_rho = RandomFeatureSpaceXY(
#             self.domain,
#             self.strides,
#             self.Jn["rho"],
#             self.scale,
#             self.activation,
#         )
#         self._model_g = RandomFeatureSpaceXYV(
#             self.domain,
#             self.strides,
#             self.Jn["g"],
#             self.scale,
#             self.activation,
#         )
#         self._dim = 3
#         self._dummy_x, self._dummy_y, self._dummy_theta = jnp.split(
#             jnp.empty((self._dim,)), self._dim, axis=0
#         )
#         self._params_rho = self._model_rho.init(
#             self.init_rng, self._dummy_x, self._dummy_y
#         )
#         self._params_g = self._model_g.init(
#             self.init_rng, self._dummy_x, self._dummy_y, self._dummy_theta
#         )
#         self._feats_fn_rho = partial(self._model_rho.apply, self._params_rho)
#         self._feats_fn_g = partial(self._model_g.apply, self._params_g)
#         self._mesh_rho = self._model_rho._mesh
#         self._mesh_g = self._model_g._mesh
#         self._Mp_rho = self._mesh_rho.number_of_cells
#         self._Mp_g = self._mesh_g.number_of_cells
#         self._Jn_rho = self.Jn["rho"]
#         self._Jn_g = self.Jn["g"]

#     def __call__(
#         self,
#         invar_x: jnp.ndarray,
#         invar_y: jnp.ndarray,
#         invar_theta: jnp.ndarray,
#     ) -> jnp.ndarray:
#         feat_rho_outvar = self._feats_fn_rho(
#             invar_x, invar_y
#         )  # (Mp_rho, Jn_rho)
#         feat_g_outvar = self._feats_fn_g(
#             invar_x, invar_y, invar_theta
#         )  # (Mp_g, Jn_g)
#         coeff_rho = self.coefficients[: self._Mp_rho * self._Jn_rho].reshape(
#             self._Mp_rho, self._Jn_rho
#         )
#         coeff_g = self.coefficients[self._Mp_rho * self._Jn_rho :].reshape(
#             self._Mp_g, self._Jn_g
#         )
#         approx_rho = jnp.einsum(
#             "ij,ij->",
#             feat_rho_outvar
#             if self._Mp_rho > 1
#             else feat_rho_outvar.reshape(1, self._Jn_rho),
#             coeff_rho,
#         )
#         approx_g = jnp.einsum(
#             "ij,ij->",
#             feat_g_outvar
#             if self._Mp_g > 1
#             else feat_g_outvar.reshape(1, self._Jn_g),
#             coeff_g,
#         )
#         approx_solution = approx_rho + self.kn * approx_g
#         return approx_solution


# # inputs: (x, v):  x = xl, v > 0 or x = xr, v < 0
# def bdy_fn(
#     x: jnp.ndarray, v: jnp.ndarray, bdy_value: Tuple[jnp.ndarray, jnp.ndarray]
# ) -> jnp.ndarray:
#     inflow_value = jnp.where(
#         v > 0, bdy_value[0], jnp.where(v < 0, bdy_value[-1], 0.0)
#     )
#     return inflow_value


# # def ex_solution(x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
# #     return 1.0 - x


# # def ex_solution_2d(
# #     x: jnp.ndarray, y: jnp.ndarray, theta: jnp.ndarray
# # ) -> jnp.ndarray:
# #     return jnp.exp(-x) * jnp.exp(-y)


# # def rhs_fn_2d(
# #     x: jnp.ndarray, y: jnp.ndarray, theta: jnp.ndarray
# # ) -> jnp.ndarray:
# #     return jnp.array(
# #         [0.0, jnp.exp(-x) * jnp.exp(-y) * (-jnp.cos(theta) - jnp.sin(theta))]
# #     )


# ## with <g> = 0
# # def rhs_fn_2d(
# #     x: jnp.ndarray, y: jnp.ndarray, theta: jnp.ndarray
# # ) -> jnp.ndarray:
# #     return jnp.array(
# #         [
# #             0.0,
# #             jnp.exp(-x) * jnp.exp(-y) * (-jnp.cos(theta) - jnp.sin(theta)),
# #             0.0,
# #         ]
# #     )


# # def rhs_fn_2d_squeeze(
# #     x: jnp.ndarray, y: jnp.ndarray, theta: jnp.ndarray
# # ) -> jnp.ndarray:
# #     return jnp.exp(-x) * jnp.exp(-y) * (-jnp.cos(theta) - jnp.sin(theta))
