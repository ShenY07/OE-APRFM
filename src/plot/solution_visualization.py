"""Provide solution visualization functionality for the plot layer.

Distinction: This file is the canonical implementation in this module family; numerical logic is preserved.
"""

import sys
from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit, random, vmap

sys.path.append("../")

from configuration.rte_1d_settings import get_config
from geometry.uniform_mesh import UniformMeshX, UniformMeshXV
from modules.collocation_sampling import Sample1D
from modules.solution_construction import MicroMacroConstructor1D, ex_solution
from plot import plot_fns
from utils.coordinate_generation import cartesian_product

rte_config = get_config()
domain = dict(rte_config.mesh["domain"])
strides = rte_config.mesh["strides"]
kn = rte_config.model.knudsen_number
Jn = rte_config.model.Jn
num_unknowns = rte_config.model.num_unknowns["rho/g"]
scale = rte_config.model.scale
activation = rte_config.model.activation
mode = rte_config.model.sample_mode
collocation_sizes = rte_config.model.collocation_sizes

key_num = rte_config.model.random_seed
rng_key = random.key(key_num)
UniformMeshX(domain=domain, strides=strides)
UniformMeshXV(domain=domain, strides=strides)

data = np.load("../data/aprfm_weights_1.0e+00.npz")
vector_U = data["coefficients"]
coefficients = jnp.array(vector_U).reshape(
    num_unknowns,
)


constructor = MicroMacroConstructor1D(
    domain=domain,
    strides=strides,
    Jn=Jn,
    scale=scale,
    init_rng=rng_key,
    kn=kn,
    activation=activation,
    coefficients=coefficients,
)

params = {}


def approx_fn(x, v):
    return partial(constructor.apply, params)(x, v)


dummy_z = jnp.array([0.7, -0.5])
dummy_x, dummy_v = jnp.split(dummy_z, 2, axis=0)
# print(f"f({dummy_z}) = {jit(approx_fn)(dummy_x, dummy_v)}")
_ = jit(approx_fn)(dummy_x, dummy_v)

vmap_approx_fn = vmap(approx_fn)

grid_sizes = collocation_sizes["interior"]
sampler = Sample1D(
    domain=domain, collocation_sizes=collocation_sizes, mode=mode
)
xv_int = sampler.grids
F = ex_solution(*xv_int).reshape(grid_sizes[0] - 2, grid_sizes[1] - 2)
F_hat = vmap_approx_fn(*xv_int).reshape(grid_sizes[0] - 2, grid_sizes[1] - 2)
X, V = jnp.split(
    cartesian_product([sampler.grid_x[1:-1, :], sampler.grid_v[1:-1, :]]),
    2,
    axis=-1,
)
X, V = X.squeeze(), V.squeeze()

np.savez(f"../data/aprfm1d_solutions_{kn:.1e}.npz", X=X, V=V, F=F, F_hat=F_hat)

plot_fns.contourf(kn=f"{kn:.1e}", method="aprfm")
