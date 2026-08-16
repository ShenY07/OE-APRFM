"""Provide collocation sampling variant 02 alternative functionality for the modules layer.

Distinction: This file is an explicitly retained alternative to the numbered variant; numerical logic is preserved.
"""

# from itertools import product
import jax
from jax import random
import jax.numpy as jnp
from typing import Dict, Tuple
from utils.coordinate_generation import cartesian_product
from abc import ABCMeta


class Sample1D(metaclass=ABCMeta):
    def __init__(
        self,
        domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
        collocation_sizes: Dict[str, Tuple[int, int] | int],
        mode: str,
    ):
        self.domain = domain
        self.collocation_sizes = collocation_sizes
        self._interior_sizes = collocation_sizes["interior"]
        self._boundary_sizes = collocation_sizes["boundary"]
        self.mode = mode
        self.grid_x, self.grid_v = self._get_mesh()
        self.pts_int = self._interior(mode=self.mode)
        self.pts_left, self.pts_right = self._boundary()
        self.grids = self._grids()

    def _grids(self):
        pts_int_ = cartesian_product(
            [
                self.grid_x[1:-1, :],
                self.grid_v[1:-1, :],
            ]
        )
        pts_int = jnp.split(pts_int_.reshape(-1, 2), 2, axis=1)
        return pts_int

    def _get_mesh(self):
        grid_x = sample_range(
            rng_seed=0,
            num_samples=self._interior_sizes[0],
            mode="uniform",
            bound=self.domain["x"],
        )[:, None]
        grid_v = sample_range(
            rng_seed=1,
            num_samples=self._interior_sizes[-1],
            mode="uniform",
            bound=self.domain["v"],
        )[:, None]
        return grid_x, grid_v

    def _interior(self, mode):
        rx = sample_range(
            rng_seed=0,
            num_samples=self._interior_sizes[0],
            mode=self.mode,
            bound=self.domain["x"],
        )[:, None]
        rv = sample_range(
            rng_seed=1,
            num_samples=self._interior_sizes[-1],
            mode=self.mode,
            bound=self.domain["v"],
        )[:, None]
        pts_int_ = cartesian_product([rx[1:-1, :], rv[1:-1, :]])
        pts_int = jnp.split(pts_int_.reshape(-1, 2), 2, axis=1)
        return pts_int

    def _boundary(self):
        vmin, vmax = self.domain["v"]
        xmin, xmax = self.domain["x"]

        mesh_v1 = jnp.linspace(vmin, vmax, self._boundary_sizes)[:, None]
        mesh_v2 = jnp.linspace(-vmax, -vmin, self._boundary_sizes)[:, None]

        pts_left = (jnp.ones_like(mesh_v1) * xmin, mesh_v1)
        pts_right = (jnp.ones_like(mesh_v2) * xmax, mesh_v2)
        return pts_left, pts_right

    # def _boundary(self):
    #     mesh_v = jnp.linspace(*self.domain["v"], self._boundary_sizes * 2)[
    #         :, None
    #     ]
    #     mesh_1 = jnp.ones((self._boundary_sizes, 1))
    #     pts_left = (
    #         mesh_1 * self.domain["x"][0],
    #         mesh_v[self._boundary_sizes:],
    #     )
    #     pts_right = (
    #         mesh_1 * self.domain["x"][1],
    #         mesh_v[: self._boundary_sizes],
    #     )
    #     return pts_left, pts_right


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

        # 组装点
        x_left = self.domain["x"][0] * jnp.ones(
            (
                self._boundary_sizes[1] * self._boundary_sizes[-1],
                1,
            )
        )
        y_left, theta_left = jnp.split(
            cartesian_product([mesh_y, mesh_theta]).reshape((-1, 2)),
            2,
            axis=1,
        )
        pts_left = x_left, y_left, theta_left

        x_right = self.domain["x"][-1] * jnp.ones(
            (
                self._boundary_sizes[1] * self._boundary_sizes[-1],
                1,
            )
        )
        y_right, theta_right = jnp.split(
            cartesian_product([mesh_y, mesh_theta]).reshape((-1, 2)),
            2,
            axis=1,
        )
        pts_right = x_right, y_right, theta_right

        y_lower = self.domain["y"][0] * jnp.ones(
            (
                self._boundary_sizes[0] * self._boundary_sizes[-1],
                1,
            )
        )
        x_lower, theta_lower = jnp.split(
            cartesian_product([mesh_x, mesh_theta]).reshape((-1, 2)),
            2,
            axis=1,
        )
        pts_lower = x_lower, y_lower, theta_lower

        y_upper = self.domain["y"][-1] * jnp.ones(
            (
                self._boundary_sizes[0] * self._boundary_sizes[-1],
                1,
            )
        )
        x_upper, theta_upper = jnp.split(
            cartesian_product([mesh_x, mesh_theta]).reshape((-1, 2)),
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
        mesh_x = jnp.linspace(*self.domain["x"], self._boundary_sizes[0] + 2)[
            1:-1, None
        ]
        mesh_y = jnp.linspace(*self.domain["y"], self._boundary_sizes[1] + 2)[
            1:-1, None
        ]
        mesh_theta = jnp.linspace(
            *self.domain["theta"], self._boundary_sizes[-1]
        )[:, None]
        # 组装点
        x_left = self.domain["x"][0] * jnp.ones(
            (
                self._boundary_sizes[1] * self._boundary_sizes[-1],
                1,
            )
        )
        y_left, theta_left = jnp.split(
            cartesian_product([mesh_y, mesh_theta]).reshape((-1, 2)),
            2,
            axis=1,
        )
        pts_left = x_left, y_left, theta_left

        x_right = self.domain["x"][-1] * jnp.ones(
            (
                self._boundary_sizes[1] * self._boundary_sizes[-1],
                1,
            )
        )
        y_right, theta_right = jnp.split(
            cartesian_product([mesh_y, mesh_theta]).reshape((-1, 2)),
            2,
            axis=1,
        )
        pts_right = x_right, y_right, theta_right

        y_lower = self.domain["y"][0] * jnp.ones(
            (
                self._boundary_sizes[0] * self._boundary_sizes[-1],
                1,
            )
        )
        x_lower, theta_lower = jnp.split(
            cartesian_product([mesh_x, mesh_theta]).reshape((-1, 2)),
            2,
            axis=1,
        )
        pts_lower = x_lower, y_lower, theta_lower

        y_upper = self.domain["y"][-1] * jnp.ones(
            (
                self._boundary_sizes[0] * self._boundary_sizes[-1],
                1,
            )
        )
        x_upper, theta_upper = jnp.split(
            cartesian_product([mesh_x, mesh_theta]).reshape((-1, 2)),
            2,
            axis=1,
        )
        pts_upper = x_upper, y_upper, theta_upper
        return pts_left, pts_right, pts_lower, pts_upper

    def _hole_boundary(self):
        mesh_x = jnp.linspace(*self.hole["x"], self._boundary_sizes[0] + 2)[
            1:-1, None
        ]
        mesh_y = jnp.linspace(*self.hole["y"], self._boundary_sizes[1] + 2)[
            1:-1, None
        ]
        mesh_theta = jnp.linspace(
            *self.domain["theta"], self._boundary_sizes[-1]
        )[:, None]

        x_left = self.hole["x"][0] * jnp.ones(
            (
                self._boundary_sizes[1] * self._boundary_sizes[-1],
                1,
            )
        )
        y_left, theta_left = jnp.split(
            cartesian_product([mesh_y, mesh_theta]).reshape((-1, 2)),
            2,
            axis=1,
        )
        pts_left = x_left, y_left, theta_left

        x_right = self.hole["x"][-1] * jnp.ones(
            (
                self._boundary_sizes[1] * self._boundary_sizes[-1],
                1,
            )
        )
        y_right, theta_right = jnp.split(
            cartesian_product([mesh_y, mesh_theta]).reshape((-1, 2)),
            2,
            axis=1,
        )
        pts_right = x_right, y_right, theta_right

        y_lower = self.hole["y"][0] * jnp.ones(
            (
                self._boundary_sizes[0] * self._boundary_sizes[-1],
                1,
            )
        )
        x_lower, theta_lower = jnp.split(
            cartesian_product([mesh_x, mesh_theta]).reshape((-1, 2)),
            2,
            axis=1,
        )
        pts_lower = x_lower, y_lower, theta_lower

        y_upper = self.hole["y"][-1] * jnp.ones(
            (
                self._boundary_sizes[0] * self._boundary_sizes[-1],
                1,
            )
        )
        x_upper, theta_upper = jnp.split(
            cartesian_product([mesh_x, mesh_theta]).reshape((-1, 2)),
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
        mesh_theta = jnp.linspace(
            *self.domain["theta"], self._boundary_sizes[-1] + 2
        )[1:-1, None]

        # ------------------
        # 左半圆边界
        # ------------------
        r = self.ymax - 1
        phi = jnp.linspace(jnp.pi / 2, 3 * jnp.pi / 2, Nb_y)[:, None]
        x_left = self.R * jnp.cos(phi)
        y_left = self.R * jnp.sin(phi)
        x_left_, t_left = jnp.split(
            cartesian_product([x_left, mesh_theta]).reshape(-1, 2), 2, axis=1
        )
        pts_left = x_left_, y_left.repeat(mesh_theta.shape[0], axis=0), t_left

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
            cartesian_product([x_r, mesh_theta]).reshape(-1, 2), 2, axis=1
        )

        # 两条竖线
        y_v1 = jnp.linspace(self.ymin, -r, Nb_y // 4)[:, None]
        y_v2 = jnp.linspace(r, self.ymax, Nb_y // 4)[:, None]
        x_v1 = self.xmax * jnp.ones_like(y_v1)
        x_v2 = self.xmax * jnp.ones_like(y_v2)
        x_vert = jnp.vstack([x_v1, x_v2])
        y_vert = jnp.vstack([y_v1, y_v2])
        x_vert_, t_vert = jnp.split(
            cartesian_product([x_vert, mesh_theta]).reshape(-1, 2), 2, axis=1
        )

        # 汇总右边界
        xr = jnp.vstack([x_r_, x_vert_])
        yr = jnp.vstack(
            [
                y_r.repeat(mesh_theta.shape[0], axis=0),
                y_vert.repeat(mesh_theta.shape[0], axis=0),
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
            cartesian_product([xt, mesh_theta]).reshape(-1, 2),
            2,
            axis=1,
        )
        pts_upper = xt_, yt.repeat(mesh_theta.shape[0], axis=0), tt

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
            cartesian_product([xb, mesh_theta]).reshape(-1, 2),
            2,
            axis=1,
        )
        pts_lower = xb_, yb.repeat(mesh_theta.shape[0], axis=0), tb

        return pts_left, pts_right, pts_lower, pts_upper


class SampleHoleCircle(metaclass=ABCMeta):
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
