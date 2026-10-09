"""Provide collocation sampling functionality for the modules layer.

Distinction: This file is the canonical implementation in this module family; numerical logic is preserved.
"""

# from itertools import product
import jax
from jax import random
import jax.numpy as jnp
from typing import Dict, Tuple
from utils.coordinate_generation import cartesian_product
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

        # 生成网格和采样点
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
        """生成边界条件采样点。

        左边界(x = 0) : v > 0, 流入边界
        右边界(x = 1) : v < 0, 流出边界

        注: v < 0 的情况在物理模型层面通过对称性处理。
        """
        xmin, xmax = self.domain["x"]
        vmin, vmax = self.domain["v"]
        # 速度不取 0：正负各取 nb 个点，排除 0
        v_pos = jnp.linspace(0, vmax, self._nb + 1)[1:, None]
        # v_neg = jnp.linspace(-vmax, -vmin, self._nb + 1)[:-1, None]

        pts_left = (jnp.full_like(v_pos, xmin), v_pos)
        pts_right = (jnp.full_like(v_pos, xmax), v_pos)
        # pts_right = (jnp.full_like(v_neg, xmax), v_neg)
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

        # 组装点
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


class SampleHole(metaclass=ABCMeta):
    def __init__(
        self,
        domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
        collocation_sizes: Dict[str, Tuple[int, int, int]],
        mode: str = "uniform",
    ):
        self.domain = domain
        self.hole = domain["hole"]
        self.collocation_sizes = collocation_sizes
        self._interior_sizes = collocation_sizes["interior"]
        self._boundary_sizes = collocation_sizes["boundary"]
        self.mode = mode
        self.grid_x, self.grid_y, self.theta = self._get_mesh()
        self.pts_int = self._interior(mode=self.mode)
        self.pts_left, self.pts_right, self.pts_lower, self.pts_upper = (
            self._boundary()
        )
        (
            self.pts_left_hole,
            self.pts_right_hole,
            self.pts_lower_hole,
            self.pts_upper_hole,
        ) = self._hole_boundary()
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
            num_samples=self._interior_sizes[-1],
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
        pts_int = list(filter(self._condition, zip(*pts_int)))
        pts_int = jnp.split(jnp.array(pts_int).squeeze(), 3, axis=-1)
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
        mesh_theta_xmin = mesh_theta[jnp.cos(mesh_theta) > 0][:, None]

        # 右边界：cosθ < 0
        mesh_theta_xmax = mesh_theta[jnp.cos(mesh_theta) < 0][:, None]

        # 下边界：sinθ > 0
        mesh_theta_ymin = mesh_theta[jnp.sin(mesh_theta) > 0][:, None]

        # 上边界：sinθ < 0
        mesh_theta_ymax = mesh_theta[jnp.sin(mesh_theta) < 0][:, None]

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

    def _hole_boundary(self):
        mesh_x = jnp.linspace(*self.hole["x"], self._boundary_sizes[0])[
            :, None
        ]
        mesh_y = jnp.linspace(*self.hole["y"], self._boundary_sizes[1])[
            :, None
        ]
        mesh_theta = jnp.linspace(
            -jnp.pi, jnp.pi, self._boundary_sizes[-1] * 4 + 2
        )[1:-1, None]

        mesh_theta_xmax = mesh_theta[jnp.cos(mesh_theta) < 0][:, None]
        mesh_theta_xmin = mesh_theta[jnp.cos(mesh_theta) > 0][:, None]
        mesh_theta_ymax = mesh_theta[jnp.sin(mesh_theta) < 0][:, None]
        mesh_theta_ymin = mesh_theta[jnp.sin(mesh_theta) > 0][:, None]
        x_left = self.hole["x"][0] * jnp.ones(
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

        x_right = self.hole["x"][-1] * jnp.ones(
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

        y_lower = self.hole["y"][0] * jnp.ones(
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

        y_upper = self.hole["y"][-1] * jnp.ones(
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

    def _condition(self, pts):
        x, y, _ = pts
        xl, xr = self.hole["x"]
        yl, yr = self.hole["y"]
        cond = (x[0] >= xl and x[0] <= xr) and (y[0] >= yl and y[0] <= yr)
        return not cond


class SampleC(metaclass=ABCMeta):
    def __init__(
        self,
        domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
        collocation_sizes: Dict[str, Tuple[int, int, int]],
        mode: str = "uniform",
    ):
        self.domain = domain
        self.hole = domain["hole"]
        self.collocation_sizes = collocation_sizes
        self._interior_sizes = collocation_sizes["interior"]
        self._boundary_sizes = collocation_sizes["boundary"]
        self.xmin, self.xmax = self.domain["x"]
        self.ymin, self.ymax = self.domain["y"]
        self.hole_xmin, self.hole_xmax = self.hole["x"]
        self.hole_ymin, self.hole_ymax = self.hole["y"]
        self.cx, self.cy = self.xmin + self.ymax, (self.ymin + self.ymax) / 2
        self.R = self.ymax

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

    # def _interior(self, mode):
    #     Nx, Ny, Ntheta = self._interior_sizes

    #     # ============================
    #     # 面积比例
    #     # ============================
    #     area_top = (self.xmax - self.xmin) * (self.ymax - self.hole_ymax)
    #     area_bot = (self.xmax - self.xmin) * (self.hole_ymin - self.ymin)
    #     area_left = (self.hole_xmin - self.xmin) * \
    #         (self.hole_ymax - self.hole_ymin)
    #     area_sum = area_top + area_bot + area_left

    #     Nt_top = int(Nx * area_top / area_sum)
    #     Nt_bot = int(Nx * area_bot / area_sum)
    #     Nt_left = Nx - Nt_top - Nt_bot

    #     # ============================
    #     # theta 统一采样
    #     # ============================
    #     rtheta = sample_range(
    #         rng_seed=2,
    #         num_samples=Ntheta,
    #         mode=mode,
    #         bound=self.domain["theta"],
    #     )[:, None]
    #     rtheta = rtheta[1:-1, :]

    #     # ============================
    #     # 上横
    #     # ============================
    #     rx_top = sample_range(0, Nt_top, mode, (self.xmin, self.xmax))[:, None]
    #     ry_top = sample_range(
    #         1, Ny, mode, (self.hole_ymax, self.ymax))[:, None]

    #     pts_top = cartesian_product([
    #         rx_top[1:-1, :],
    #         ry_top[1:-1, :],
    #         rtheta,
    #     ])

    #     # ============================
    #     # 下横
    #     # ============================
    #     rx_bot = sample_range(3, Nt_bot, mode, (self.xmin, self.xmax))[:, None]
    #     ry_bot = sample_range(
    #         4, Ny, mode, (self.ymin, self.hole_ymin))[:, None]

    #     pts_bot = cartesian_product([
    #         rx_bot[1:-1, :],
    #         ry_bot[1:-1, :],
    #         rtheta,
    #     ])

    #     # ============================
    #     # 左竖
    #     # ============================
    #     rx_left = sample_range(
    #         5, Nt_left, mode, (self.xmin, self.hole_xmin))[:, None]
    #     ry_left = sample_range(
    #         6, Ny, mode, (self.hole_ymin, self.hole_ymax))[:, None]

    #     pts_left = cartesian_product([
    #         rx_left[1:-1, :],
    #         ry_left[1:-1, :],
    #         rtheta,
    #     ])

    #     # ============================
    #     # 拼接 & split
    #     # ============================
    #     pts_int_ = jnp.concatenate([pts_top, pts_bot, pts_left], axis=0)
    #     pts_int = jnp.split(pts_int_.reshape(-1, 3), 3, axis=1)

    #     return pts_int

    # def _boundary(self):
    #     Nb_x, Nb_y, Nb_theta = self._boundary_sizes

    #     mesh_theta = jnp.linspace(
    #         -jnp.pi, jnp.pi, Nb_theta * 4 + 2
    #     )[1:-1, None]

    #     # 方向筛选（与你现有逻辑一致）
    #     theta_xpos = mesh_theta[(jnp.cos(mesh_theta) > 0).squeeze(), :]
    #     theta_xneg = mesh_theta[(jnp.cos(mesh_theta) < 0).squeeze(), :]
    #     theta_ypos = mesh_theta[(jnp.sin(mesh_theta) > 0).squeeze(), :]
    #     theta_yneg = mesh_theta[(jnp.sin(mesh_theta) < 0).squeeze(), :]

    #     # ============================
    #     # Left
    #     # ============================
    #     y_left = jnp.linspace(self.ymin, self.ymax, Nb_y)[:, None]
    #     x_left = jnp.ones(
    #         (y_left.shape[0] * theta_xpos.shape[0], 1)) * self.xmin

    #     yl, tl = jnp.split(
    #         cartesian_product([y_left, theta_xpos]).reshape(-1, 2),
    #         2,
    #         axis=1,
    #     )
    #     pts_left = x_left, yl, tl

    #     # ============================
    #     # Right
    #     # ============================
    #     # y 方向三段
    #     y_r1 = jnp.linspace(self.ymin, self.hole_ymin, Nb_y // 3)[:, None]
    #     y_r2 = jnp.linspace(self.hole_ymax, self.ymax, Nb_y // 3)[:, None]
    #     y_r3 = jnp.linspace(self.hole_ymin, self.hole_ymax,
    #                         Nb_y - 2*Nb_y // 3)[:, None]

    #     # x 对应位置
    #     x_r1 = self.xmax * jnp.ones_like(y_r1)
    #     x_r2 = self.xmax * jnp.ones_like(y_r2)
    #     x_r3 = self.hole_xmin * jnp.ones_like(y_r3)

    #     # 汇总几何点
    #     xr = jnp.vstack([x_r1, x_r2, x_r3])
    #     yr = jnp.vstack([y_r1, y_r2, y_r3])

    #     # 与角度做 Cartesian product
    #     xr_, tr_ = jnp.split(
    #         cartesian_product([xr, theta_xneg]).reshape(-1, 2),
    #         2,
    #         axis=1,
    #     )

    #     pts_right = xr_, yr.repeat(theta_xneg.shape[0], axis=0), tr_

    #     # ============================
    #     # Bottom
    #     # ============================
    #     x_b1 = jnp.linspace(self.xmin, self.xmax, Nb_x // 2)[:, None]
    #     x_b2 = jnp.linspace(self.hole_xmin, self.hole_xmax, Nb_x // 2)[:, None]

    #     y_b1 = self.ymin * jnp.ones_like(x_b1)
    #     y_b2 = self.hole_ymax * jnp.ones_like(x_b2)
    #     xb = jnp.vstack([x_b1, x_b2])
    #     yb = jnp.vstack([y_b1, y_b2])

    #     xb_, tb_ = jnp.split(
    #         cartesian_product([xb, theta_ypos]).reshape(-1, 2),
    #         2,
    #         axis=1,
    #     )
    #     pts_lower = xb_, yb.repeat(theta_ypos.shape[0], axis=0), tb_

    #     # ============================
    #     # Upper
    #     # ============================
    #     x_u1 = jnp.linspace(self.xmin, self.xmax, Nb_x // 2)[:, None]
    #     x_u2 = jnp.linspace(self.hole_xmin, self.hole_xmax, Nb_x // 2)[:, None]

    #     y_u1 = self.ymax * jnp.ones_like(x_u1)
    #     y_u2 = self.hole_ymax * jnp.ones_like(x_u2)
    #     xu = jnp.vstack([x_u1, x_u2])
    #     yu = jnp.vstack([y_u1, y_u2])

    #     xu_, tu_ = jnp.split(
    #         cartesian_product([xu, theta_yneg]).reshape(-1, 2),
    #         2,
    #         axis=1,
    #     )
    #     pts_upper = xu_, yu.repeat(theta_yneg.shape[0], axis=0), tu_

    #     return pts_left, pts_right, pts_lower, pts_upper

    def _interior(self, mode):
        Nx, Ny, Ntheta = self._interior_sizes

        # ------------------
        # 统一采样角度 theta
        # ------------------
        rtheta = sample_range(
            rng_seed=2,
            num_samples=Ntheta,
            mode=mode,
            bound=self.domain["theta"],
        )
        rtheta = rtheta.flatten()  # shape (Ntheta,)

        # ------------------
        # 左半圆内部 (x <= 0)
        # ------------------
        r = jnp.linspace(self.ymax - 1, self.ymax, Nx)
        phi = jnp.linspace(jnp.pi / 2, 3 * jnp.pi / 2, Ny)
        R, PHI = jnp.meshgrid(r, phi, indexing="ij")
        Xc = R * jnp.cos(PHI) + self.cx
        Yc = R * jnp.sin(PHI) + self.cy
        pts_circle = jnp.stack([Xc.flatten(), Yc.flatten()], axis=1)

        # ------------------
        # 上下矩形
        # ------------------
        X_top = jnp.linspace(self.cx, self.xmax, Nx)
        # r 是数组，取最大值作为 Y_top 起点
        Y_top = jnp.linspace(float(r[-1]), self.ymax, Ny)
        XX_top, YY_top = jnp.meshgrid(X_top, Y_top, indexing="ij")
        pts_top = jnp.stack([XX_top.flatten(), YY_top.flatten()], axis=1)

        X_bot = jnp.linspace(self.cx, self.xmax, Nx)
        # r 是数组，取最小值作为 Y_bot 终点
        Y_bot = jnp.linspace(self.ymin, float(r[0]), Ny)
        XX_bot, YY_bot = jnp.meshgrid(X_bot, Y_bot, indexing="ij")
        pts_bot = jnp.stack([XX_bot.flatten(), YY_bot.flatten()], axis=1)

        # ------------------
        # 合并内部二维点
        # ------------------
        pts_xy = jnp.vstack([pts_circle, pts_top, pts_bot])  # (Nxy, 2)

        # ------------------
        # 与 theta 做笛卡尔积
        # ------------------
        Nxy = pts_xy.shape[0]
        Nth = rtheta.shape[0]
        # 扩展为 (Nxy*Nth, 1)
        x_col = jnp.repeat(pts_xy[:, 0], Nth)[:, None]
        y_col = jnp.repeat(pts_xy[:, 1], Nth)[:, None]
        theta_col = jnp.tile(rtheta, Nxy)[:, None]
        pts_xyz = jnp.concatenate([x_col, y_col, theta_col], axis=1)
        x_int, y_int, theta_int = jnp.split(pts_xyz, 3, axis=1)
        return x_int, y_int, theta_int

    def _boundary(self):
        Nb_x, Nb_y, Nb_theta = self._boundary_sizes
        mesh_theta = jnp.linspace(-jnp.pi, jnp.pi, Nb_theta * 4 + 2)[
            1:-1, None
        ]

        # 方向筛选
        theta_xpos = mesh_theta[(jnp.cos(mesh_theta) > 0).squeeze(), :]
        theta_xneg = mesh_theta[(jnp.cos(mesh_theta) < 0).squeeze(), :]
        theta_ypos = mesh_theta[(jnp.sin(mesh_theta) > 0).squeeze(), :]
        theta_yneg = mesh_theta[(jnp.sin(mesh_theta) < 0).squeeze(), :]

        # ------------------
        # 左半圆边界
        # ------------------
        r = self.ymax - 1
        phi = jnp.linspace(jnp.pi / 2, 3 * jnp.pi / 2, Nb_y)[:, None]
        x_left = self.R * jnp.cos(phi)
        y_left = self.R * jnp.sin(phi)
        x_left_, t_left = jnp.split(
            cartesian_product([x_left, theta_xpos]).reshape(-1, 2), 2, axis=1
        )
        pts_left = x_left_, y_left.repeat(theta_xpos.shape[0], axis=0), t_left

        # ------------------
        # 右边界：右半圆 + 两条竖线
        # ------------------
        # 右半圆
        phi_r = jnp.linspace(jnp.pi / 2, 3 * jnp.pi / 2, Nb_y - Nb_y // 4 * 2)[
            :, None
        ]
        x_r = r * jnp.cos(phi_r)
        y_r = r * jnp.sin(phi_r)
        x_r_, t_r = jnp.split(
            cartesian_product([x_r, theta_xneg]).reshape(-1, 2), 2, axis=1
        )

        # 两条竖线
        y_v1 = jnp.linspace(self.ymin, -r, Nb_y // 4)[:, None]
        y_v2 = jnp.linspace(r, self.ymax, Nb_y // 4)[:, None]
        x_v1 = self.xmax * jnp.ones_like(y_v1)
        x_v2 = self.xmax * jnp.ones_like(y_v2)
        x_vert = jnp.vstack([x_v1, x_v2])
        y_vert = jnp.vstack([y_v1, y_v2])
        x_vert_, t_vert = jnp.split(
            cartesian_product([x_vert, theta_xneg]).reshape(-1, 2), 2, axis=1
        )

        # 汇总右边界
        xr = jnp.vstack([x_r_, x_vert_])
        yr = jnp.vstack(
            [
                y_r.repeat(theta_xneg.shape[0], axis=0),
                y_vert.repeat(theta_xneg.shape[0], axis=0),
            ]
        )
        tr = jnp.vstack([t_r, t_vert])
        pts_right = xr, yr, tr

        # ------------------
        # 上边界：两条横线
        # ------------------
        x_top1 = jnp.linspace(self.cx, self.xmax, Nb_x // 2)[:, None]  # 左半段
        y_top1 = -r * jnp.ones_like(x_top1)

        x_top2 = jnp.linspace(self.cx, self.xmax, Nb_x - Nb_x // 2)[
            :, None
        ]  # 右半段
        y_top2 = self.ymax * jnp.ones_like(x_top2)

        xt = jnp.vstack([x_top1, x_top2])
        yt = jnp.vstack([y_top1, y_top2])

        xt_, tt = jnp.split(
            cartesian_product([xt, theta_yneg]).reshape(-1, 2),
            2,
            axis=1,
        )
        pts_upper = xt_, yt.repeat(theta_yneg.shape[0], axis=0), tt

        # ------------------
        # 下边界：两条横线
        # ------------------
        x_bot1 = jnp.linspace(self.cx, self.xmax, Nb_x // 2)[:, None]
        y_bot1 = r * jnp.ones_like(x_bot1)

        x_bot2 = jnp.linspace(self.cx, self.xmax, Nb_x // 2)[:, None]
        y_bot2 = self.ymin * jnp.ones_like(x_bot2)

        xb = jnp.vstack([x_bot1, x_bot2])
        yb = jnp.vstack([y_bot1, y_bot2])

        xb_, tb = jnp.split(
            cartesian_product([xb, theta_ypos]).reshape(-1, 2),
            2,
            axis=1,
        )
        pts_lower = xb_, yb.repeat(theta_ypos.shape[0], axis=0), tb

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
        pts = latin_hypercube_sample(rng_key, num_samples)
    else:
        raise ValueError("Invalid mode")
    pts = pts.reshape(-1, 1)
    pts *= bound[1] - bound[0]
    pts += bound[0]
    return pts


def latin_hypercube_sample(key, num_samples, dim=1):
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
