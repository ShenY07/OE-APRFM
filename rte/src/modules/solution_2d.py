import jax.numpy as jnp
from jax import random
from collections.abc import Callable
from typing import Dict, Tuple
from modules.func_space_2d import (
    RandomFeatureSpaceXYV,
    RandomFeatureSpaceXY,
)
from functools import partial


class MicroMacroConstructor2D(RandomFeatureSpaceXY, RandomFeatureSpaceXYV):
    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: int
    scale: float
    init_rng: random.PRNGKey
    kn: float
    activation: Callable
    coefficients: jnp.ndarray

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
        self._mesh_g = self._model_g._mesh
        self._Mp_rho = self._mesh_rho.number_of_cells
        self._Mp_g = self._mesh_g.number_of_cells
        self._Jn_rho = self.Jn["rho"]
        self._Jn_g = self.Jn["g"]

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_y: jnp.ndarray,
        invar_theta: jnp.ndarray,
    ) -> jnp.ndarray:
        feat_rho_outvar = self._feats_fn_rho(
            invar_x, invar_y
        )  # (Mp_rho, Jn_rho)
        feat_g_outvar = self._feats_fn_g(
            invar_x, invar_y, invar_theta
        )  # (Mp_g, Jn_g)
        coeff_rho = self.coefficients[: self._Mp_rho * self._Jn_rho].reshape(
            self._Mp_rho, self._Jn_rho
        )
        coeff_g = self.coefficients[self._Mp_rho * self._Jn_rho :].reshape(
            self._Mp_g, self._Jn_g
        )
        approx_rho = jnp.einsum(
            "ij,ij->",
            feat_rho_outvar
            if self._Mp_rho > 1
            else feat_rho_outvar.reshape(1, self._Jn_rho),
            coeff_rho,
        )
        approx_g = jnp.einsum(
            "ij,ij->",
            feat_g_outvar
            if self._Mp_g > 1
            else feat_g_outvar.reshape(1, self._Jn_g),
            coeff_g,
        )
        approx_solution = approx_rho + self.kn * approx_g
        return approx_solution


# inputs: (x, v):  x = xl, v > 0 or x = xr, v < 0
def bdy_fn(
    x: jnp.ndarray, v: jnp.ndarray, bdy_value: Tuple[jnp.ndarray, jnp.ndarray]
) -> jnp.ndarray:
    inflow_value = jnp.where(
        v > 0, bdy_value[0], jnp.where(v < 0, bdy_value[-1], 0.0)
    )
    return inflow_value


# def ex_solution(x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
#     return 1.0 - x


# def ex_solution_2d(
#     x: jnp.ndarray, y: jnp.ndarray, theta: jnp.ndarray
# ) -> jnp.ndarray:
#     return jnp.exp(-x) * jnp.exp(-y)


# def rhs_fn_2d(
#     x: jnp.ndarray, y: jnp.ndarray, theta: jnp.ndarray
# ) -> jnp.ndarray:
#     return jnp.array(
#         [0.0, jnp.exp(-x) * jnp.exp(-y) * (-jnp.cos(theta) - jnp.sin(theta))]
#     )


## with <g> = 0
# def rhs_fn_2d(
#     x: jnp.ndarray, y: jnp.ndarray, theta: jnp.ndarray
# ) -> jnp.ndarray:
#     return jnp.array(
#         [
#             0.0,
#             jnp.exp(-x) * jnp.exp(-y) * (-jnp.cos(theta) - jnp.sin(theta)),
#             0.0,
#         ]
#     )


# def rhs_fn_2d_squeeze(
#     x: jnp.ndarray, y: jnp.ndarray, theta: jnp.ndarray
# ) -> jnp.ndarray:
#     return jnp.exp(-x) * jnp.exp(-y) * (-jnp.cos(theta) - jnp.sin(theta))
