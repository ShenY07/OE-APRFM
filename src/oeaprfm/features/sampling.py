"""Sample interior and inflow points on intervals and planar domains."""

import jax
from jax import random
import jax.numpy as jnp
from typing import Dict, Tuple
from oeaprfm.utils.coordinates import cartesian_product
from abc import ABCMeta


class Sample1D(metaclass=ABCMeta):
    """1D 辐射传输方程的采样点生成器。

    生成内部配置点和边界条件采样点，定义域为 x∈[xmin, xmax], v∈[vmin, vmax]。
    """

    def __init__(
        self,
        domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
        collocation_sizes: Dict[str, Tuple[int, int] | int],
        mode: str,
    ):
        """初始化采样点生成器。

        Args:
            domain: 定义域字典，包含 'x' 和 'v' 的边界
            collocation_sizes: 配置点数量，包含 'interior' 和 'boundary'
            mode: 采样模式，'uniform', 'random', 或 'lhs'
        """
        self.domain = domain
        self.collocation_sizes = collocation_sizes
        self._nx, self._nv = collocation_sizes["interior"]
        self._nb = collocation_sizes["boundary"]
        self.mode = mode

        self.grid_x, self.grid_v = self._get_mesh()
        self.pts_int = self._interior()
        self.pts_left, self.pts_right = self._boundary()
        self.grids = self._grids()

    def _get_mesh(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """生成均匀网格（用于 POU 函数）。"""
        grid_x = jnp.linspace(*self.domain["x"], self._nx)[:, None]
        grid_v = jnp.linspace(*self.domain["v"], self._nv)[:, None]
        return grid_x, grid_v

    def _grids(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """返回内部网格点（去除边界）的笛卡尔积。"""
        pts = cartesian_product(
            [
                self.grid_x[1:-1, :],
                self.grid_v[1:-1, :],
            ]
        )
        return jnp.split(pts.reshape(-1, 2), 2, axis=1)

    def _interior(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """生成内部配置点（去除边界）。"""
        x_samples = sample_range(
            rng_seed=0,
            num_samples=self._nx,
            mode=self.mode,
            bound=self.domain["x"],
        )[1:-1]

        if self.mode == "uniform":
            # 半格点：取相邻网格的中心点，避免落在边界
            v_edges = jnp.linspace(*self.domain["v"], self._nv + 1)[:, None]
            v_samples = 0.5 * (v_edges[:-1] + v_edges[1:])
        else:
            v_samples = sample_range(
                rng_seed=1,
                num_samples=self._nv,
                mode=self.mode,
                bound=self.domain["v"],
            )[1:-1]

        pts = cartesian_product([x_samples, v_samples])
        return jnp.split(pts.reshape(-1, 2), 2, axis=1)

    def _boundary(
        self,
    ) -> Tuple[
        Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]
    ]:
        """在两端采样流入点，用正速度大小表示，右端方向由奇偶约束处理。"""
        xmin, xmax = self.domain["x"]
        vmin, vmax = self.domain["v"]
        # 排除零速度，避免取到与边界相切的方向。
        v_pos = jnp.linspace(0, vmax, self._nb + 1)[1:, None]

        pts_left = (jnp.full_like(v_pos, xmin), v_pos)
        pts_right = (jnp.full_like(v_pos, xmax), v_pos)

        return pts_left, pts_right


class Sample2D(metaclass=ABCMeta):
    def __init__(
        self,
        domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
        collocation_sizes: Dict[str, Tuple[int, int, int]],
        mode: str = "uniform",
    ):
        self.domain = domain
        self.collocation_sizes = collocation_sizes
        self._interior_sizes = collocation_sizes["interior"]
        self._boundary_sizes = collocation_sizes["boundary"]
        self.mode = mode
        self.grid_x, self.grid_y, self.theta = self._get_mesh()
        self.pts_int = self._interior(mode=self.mode)
        self.pts_left, self.pts_right, self.pts_lower, self.pts_upper = (
            self._boundary()
        )
        self.grids = self._grids()

    def _grids(self):
        pts_int_ = cartesian_product(
            [
                self.grid_x[1:-1, :],
                self.grid_y[1:-1, :],
            ]
        )
        pts_int = jnp.split(pts_int_.reshape(-1, 2), 2, axis=1)
        return pts_int

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
            num_samples=self._interior_sizes[-1] - 2,
            mode=self.mode,
            bound=self.domain["theta"],
        )[:, None]
        pts_int_ = cartesian_product(
            [
                rx[1:-1, :],
                ry[1:-1, :],
                rtheta[1:-1, :],
            ]
        )
        pts_int = jnp.split(pts_int_.reshape(-1, 3), 3, axis=1)
        return pts_int

    def _boundary(self):
        mesh_x = jnp.linspace(*self.domain["x"], self._boundary_sizes[0])[
            :, None
        ]
        mesh_y = jnp.linspace(*self.domain["y"], self._boundary_sizes[1])[
            :, None
        ]
        mesh_theta = jnp.linspace(
            -jnp.pi, jnp.pi, self._boundary_sizes[-1] * 4 + 2
        )[1:-1, None]

        # 左边界：cosθ > 0
        mesh_theta_xmin = mesh_theta[(jnp.cos(mesh_theta) > 0)][:, None]

        # 右边界：cosθ < 0
        mesh_theta_xmax = mesh_theta[(jnp.cos(mesh_theta) < 0)][:, None]

        # 下边界：sinθ > 0
        mesh_theta_ymin = mesh_theta[(jnp.sin(mesh_theta) > 0)][:, None]

        # 上边界：sinθ < 0
        mesh_theta_ymax = mesh_theta[(jnp.sin(mesh_theta) < 0)][:, None]

        x_left = self.domain["x"][0] * jnp.ones(
            (
                2 * self._boundary_sizes[1] * self._boundary_sizes[-1],
                1,
            )
        )
        y_left, theta_left = jnp.split(
            cartesian_product([mesh_y, mesh_theta_xmin]).reshape((-1, 2)),
            2,
            axis=1,
        )
        pts_left = x_left, y_left, theta_left

        x_right = self.domain["x"][-1] * jnp.ones(
            (
                2 * self._boundary_sizes[1] * self._boundary_sizes[-1],
                1,
            )
        )
        y_right, theta_right = jnp.split(
            cartesian_product([mesh_y, mesh_theta_xmax]).reshape((-1, 2)),
            2,
            axis=1,
        )
        pts_right = x_right, y_right, theta_right

        y_lower = self.domain["y"][0] * jnp.ones(
            (
                2 * self._boundary_sizes[0] * self._boundary_sizes[-1],
                1,
            )
        )
        x_lower, theta_lower = jnp.split(
            cartesian_product([mesh_x, mesh_theta_ymin]).reshape((-1, 2)),
            2,
            axis=1,
        )
        pts_lower = x_lower, y_lower, theta_lower

        y_upper = self.domain["y"][-1] * jnp.ones(
            (
                2 * self._boundary_sizes[0] * self._boundary_sizes[-1],
                1,
            )
        )
        x_upper, theta_upper = jnp.split(
            cartesian_product([mesh_x, mesh_theta_ymax]).reshape((-1, 2)),
            2,
            axis=1,
        )
        pts_upper = x_upper, y_upper, theta_upper

        return pts_left, pts_right, pts_lower, pts_upper


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
