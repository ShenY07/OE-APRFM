"""Sample transport domains with rectangular or circular holes."""

import jax
from jax import random
import jax.numpy as jnp
from typing import Dict, Tuple
from oeaprfm.utils.coordinates import cartesian_product
from abc import ABCMeta


class SampleCircleHole(metaclass=ABCMeta):
    def __init__(
        self,
        domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
        collocation_sizes: Dict[str, Tuple[int, int, int]],
        mode: str = "uniform",
    ):
        self.domain = domain
        self.hole_center = domain["hole"]["center"]  # (x0, y0)
        self.hole_radius = domain["hole"]["radius"]  # r
        self.collocation_sizes = collocation_sizes
        self._interior_sizes = collocation_sizes["interior"]
        self._boundary_sizes = collocation_sizes["boundary"]
        self.mode = mode

        self.grid_x, self.grid_y, self.theta = self._get_mesh()
        self.pts_int = self._interior(mode=self.mode)
        self.pts_left, self.pts_right, self.pts_lower, self.pts_upper = (
            self._boundary()
        )
        self.pts_hole = self._hole_boundary()
        self.grids = self._grids()

    def _get_mesh(self):
        grid_x = jnp.linspace(*self.domain["x"], self._interior_sizes[0] + 1)[
            :, None
        ]
        grid_y = jnp.linspace(*self.domain["y"], self._interior_sizes[1] + 1)[
            :, None
        ]
        theta = jnp.linspace(
            *self.domain["theta"], self._interior_sizes[-1] + 1
        )[:, None]
        return grid_x, grid_y, theta

    def _grids(self):
        pts_int_ = cartesian_product(
            [
                self.grid_x[1:-1, :],
                self.grid_y[1:-1, :],
            ]
        )
        pts_int = jnp.split(pts_int_.reshape(-1, 2), 2, axis=1)
        return pts_int

    def _interior(self, mode):
        rx = sample_range(
            rng_seed=0,
            num_samples=self._interior_sizes[0],
            mode=self.mode,
            bound=self.domain["x"],
        )[:, None]
        ry = sample_range(
            rng_seed=1,
            num_samples=self._interior_sizes[1],
            mode=self.mode,
            bound=self.domain["y"],
        )[:, None]
        rtheta = sample_range(
            rng_seed=2,
            num_samples=self._interior_sizes[-1],
            mode=self.mode,
            bound=self.domain["theta"],
        )[:, None]

        pts_int_ = cartesian_product(
            [rx[1:-1, :], ry[1:-1, :], rtheta[1:-1, :]]
        )
        pts_int = jnp.split(pts_int_.reshape(-1, 3), 3, axis=1)
        pts_int = list(filter(self._condition, zip(*pts_int)))
        pts_int = jnp.split(jnp.array(pts_int).squeeze(), 3, axis=-1)
        return pts_int

    def _boundary(self):
        mesh_x = jnp.linspace(*self.domain["x"], self._boundary_sizes[0] + 2)[
            1:-1, None
        ]
        mesh_y = jnp.linspace(*self.domain["y"], self._boundary_sizes[1] + 2)[
            1:-1, None
        ]
        mesh_theta = jnp.linspace(
            *self.domain["theta"], self._boundary_sizes[-1]
        )[:, None]

        # 左右边界
        x_left = self.domain["x"][0] * jnp.ones(
            (self._boundary_sizes[1] * self._boundary_sizes[-1], 1)
        )
        y_left, theta_left = jnp.split(
            cartesian_product([mesh_y, mesh_theta]).reshape((-1, 2)), 2, axis=1
        )
        pts_left = x_left, y_left, theta_left

        x_right = self.domain["x"][-1] * jnp.ones(
            (self._boundary_sizes[1] * self._boundary_sizes[-1], 1)
        )
        y_right, theta_right = jnp.split(
            cartesian_product([mesh_y, mesh_theta]).reshape((-1, 2)), 2, axis=1
        )
        pts_right = x_right, y_right, theta_right

        # 上下边界
        y_lower = self.domain["y"][0] * jnp.ones(
            (self._boundary_sizes[0] * self._boundary_sizes[-1], 1)
        )
        x_lower, theta_lower = jnp.split(
            cartesian_product([mesh_x, mesh_theta]).reshape((-1, 2)), 2, axis=1
        )
        pts_lower = x_lower, y_lower, theta_lower

        y_upper = self.domain["y"][-1] * jnp.ones(
            (self._boundary_sizes[0] * self._boundary_sizes[-1], 1)
        )
        x_upper, theta_upper = jnp.split(
            cartesian_product([mesh_x, mesh_theta]).reshape((-1, 2)), 2, axis=1
        )
        pts_upper = x_upper, y_upper, theta_upper

        return pts_left, pts_right, pts_lower, pts_upper

    def _hole_boundary(self):
        """生成圆形洞的边界点，只返回圆周 (x, y)，不和 theta 做笛卡尔积"""
        # A one-dimensional boundary needs O(N), not O(N^2), spatial points.
        # The old product over-sampled the circle and implicitly gave it a
        # much larger least-squares weight than the four outer sides.
        N = 4 * max(self._boundary_sizes[0], self._boundary_sizes[1])
        cx, cy = self.hole_center
        r = self.hole_radius

        # 均匀角度生成圆周点
        t = jnp.linspace(0, 2 * jnp.pi, N, endpoint=False)[:, None]
        x_circle = cx + r * jnp.cos(t)
        y_circle = cy + r * jnp.sin(t)

        pts_hole = x_circle, y_circle, t

        return pts_hole

    def _condition(self, pts):
        """排除圆形洞内部"""
        x, y, _ = pts
        dx = x[0] - self.hole_center[0]
        dy = y[0] - self.hole_center[1]
        return dx**2 + dy**2 > self.hole_radius**2


def sample_range(
    rng_seed: int,
    num_samples: int,
    mode: str,
    bound: Tuple[float, float],
) -> jnp.ndarray:
    rng_key = random.PRNGKey(rng_seed)
    if mode == "uniform":
        pts = jnp.linspace(0, 1, num_samples)
    elif mode == "random":
        pts = random.uniform(key=rng_key, shape=(num_samples,))
    elif mode == "lhs":
        pts = sample_latin(rng_key, num_samples)
    else:
        raise ValueError("Invalid mode")
    pts = pts.reshape(-1, 1)
    pts *= bound[1] - bound[0]
    pts += bound[0]
    return pts


def sample_latin(key, num_samples, dim=1):
    rng_keys = random.split(key, dim)
    samples = []
    for i in range(dim):
        indices = jax.random.permutation(rng_keys[i], num_samples)
        samples.append(
            (indices + jax.random.uniform(rng_keys[i], (num_samples,)))
            / num_samples
        )
    pts = jnp.column_stack(samples)
    return pts
