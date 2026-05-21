import jax.numpy as jnp
from jax import random
from collections.abc import Callable
from typing import Dict, Tuple
from flax import linen as nn
from jax.nn.initializers import uniform
from geometry.meshxd_ds import (
    UniformMeshX,
    UniformMeshXY,
    UniformMeshXV,
    UniformMeshXYV,
)
from modules.pou_func import psi_a as psi
from functools import partial

seedX, seedXV = 1, 42


class RandomFeatureSpaceX(nn.Module):
    domain: Dict[str, Tuple[jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: int
    scale: float
    activation: Callable

    def setup(self):
        super().setup()
        self._dim = 1
        self._psi = psi
        self._mesh = UniformMeshX(domain=self.domain, strides=self.strides)
        self._center = self._mesh.center_of_cell
        self._radius = self._mesh.radius_of_cell
        self._Mp = self._mesh.number_of_cells

        feats_list, params_list = [], []
        key = random.key(seedX)
        rng_keys = random.split(key, self._Mp)
        for i in range(self._Mp):
            feats = RandomFeatureFunctionsX(
                domain=self.domain,
                strides=self.strides,
                query_index=i,
                Jn=self.Jn,
                scale=self.scale,
                activation=self.activation,
            )
            # rng_key = random.key(
            #     i + self._Mp * (self._dim - 1)
            # )  # NOTE: This is not a good practice for choosing the random key
            dummy_x = jnp.empty((self._dim,))
            params = feats.init(rng_keys[i], dummy_x)
            feats_list.append(feats)
            params_list.append(params)

        self._feats1d_list = feats_list
        self._params_list = params_list

    def __call__(self, invar_x: jnp.ndarray, has_aux=False) -> jnp.ndarray:
        results = []
        psi_norm = 0.0
        for i, param, model in zip(
            range(self._Mp), self._params_list, self._feats1d_list
        ):
            psi = self._psi((invar_x - self._center(i)) / self._radius)
            psi_norm += psi  # summation over psi_i
            outvar = model.apply(param, invar_x)
            outvar = psi * outvar
            results.append(outvar)
        outvar = (
            jnp.array(results).squeeze() / psi_norm
        )  # pou with normaization # (Mp, Jn)
        if has_aux:
            return outvar, self._get_operators()
        else:
            return outvar

    def _get_operators(self):
        feat_funcs = [
            partial(self._feats1d_list[i].apply, self._params_list[i])
            for i in range(self._Mp)
        ]
        return feat_funcs


class RandomFeatureSpaceXY(nn.Module):
    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: int
    scale: float
    activation: Callable

    def setup(self):
        super().setup()
        self._dim = 2
        self._psi = psi
        self._mesh = UniformMeshXY(domain=self.domain, strides=self.strides)
        self._center = self._mesh.center_of_cell
        self._radius = self._mesh.radius_of_cell
        self._Mp = self._mesh.number_of_cells

        feats_list, params_list = [], []
        key = random.key(seedX)
        rng_keys = random.split(key, self._Mp)
        for i in range(self._Mp):
            feats = RandomFeatureFunctionsXY(
                domain=self.domain,
                strides=self.strides,
                query_index=i,
                Jn=self.Jn,
                scale=self.scale,
                activation=self.activation,
            )
            dummy_x, dummy_y = jnp.split(
                jnp.empty((self._dim,)), self._dim, axis=0)
            # rng_key = random.key(i + self._Mp * (self._dim - 1))
            params = feats.init(rng_keys[i], dummy_x, dummy_y)
            feats_list.append(feats)
            params_list.append(params)

        self._feats2d_list = feats_list
        self._params_list = params_list

    def __call__(
        self, invar_x: jnp.ndarray, invar_y: jnp.ndarray, has_aux=False
    ) -> jnp.ndarray:
        results = []
        psi_norm = 0.0
        for i, param, model in zip(
            range(self._Mp), self._params_list, self._feats2d_list
        ):
            psi = self._psi(
                (invar_x - self._center(i)[0]) / self._radius[0]
            ) * self._psi((invar_y - self._center(i)[1]) / self._radius[1])
            psi_norm += psi  # summation over psi_i
            outvar = model.apply(param, invar_x, invar_y)
            outvar = psi * outvar
            results.append(outvar)
        outvar = (
            jnp.array(results).squeeze() / psi_norm
        )  # pou with normaization # (Mp, Jn)
        if has_aux:
            return outvar, self._get_operators()
        else:
            return outvar

    def _get_operators(self):
        feat_funcs = [
            partial(self._feats2d_list[i].apply, self._params_list[i])
            for i in range(self._Mp)
        ]
        return feat_funcs


class RandomFeatureSpaceXYV(nn.Module):
    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: int
    scale: float
    activation: Callable

    def setup(self):
        super().setup()
        self._dim = 3
        self._psi = psi
        self._mesh = UniformMeshXYV(domain=self.domain, strides=self.strides)
        self._center = self._mesh.center_of_cell
        self._radius = self._mesh.radius_of_cell
        self._Mp = self._mesh.number_of_cells

        feats_list, params_list = [], []
        key = random.key(seedX)
        rng_keys = random.split(key, self._Mp)
        for i in range(self._Mp):
            feats = RandomFeatureFunctionsXYV(
                domain=self.domain,
                strides=self.strides,
                query_index=i,
                Jn=self.Jn,
                scale=self.scale,
                activation=self.activation,
            )
            dummy_x, dummy_y, dummy_v = jnp.split(
                jnp.empty((self._dim,)), self._dim, axis=0
            )
            # rng_key = random.key(i + self._Mp * (self._dim - 1))
            params = feats.init(rng_keys[i], dummy_x, dummy_y, dummy_v)
            feats_list.append(feats)
            params_list.append(params)

        self._feats2d_list = feats_list
        self._params_list = params_list

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_y: jnp.ndarray,
        invar_theta: jnp.ndarray,
        has_aux=False,
    ) -> jnp.ndarray:
        results = []
        psi_norm = 0.0
        for i, param, model in zip(
            range(self._Mp), self._params_list, self._feats2d_list
        ):
            center_x, center_y, center_theta = self._center(i)
            # center_theta = center_theta - \
            #     jnp.floor(invar_theta / jnp.pi) * jnp.pi

            psi = (
                self._psi((invar_x - center_x) / self._radius[0])
                * self._psi((invar_y - center_y) / self._radius[1])
                * self._psi((invar_theta - center_theta) / self._radius[2])
            )
            psi_norm += psi  # summation over psi_i
            outvar = model.apply(
                param,
                invar_x,
                invar_y,
                invar_theta,
                center_theta=center_theta,
            )

            outvar = psi * outvar
            results.append(outvar)

        outvar = jnp.array(results).squeeze()
        outvar = jnp.where(psi_norm > 0, outvar / psi_norm, 0.0)

        if has_aux:
            return outvar, self._get_operators()
        else:
            return outvar

    def _get_operators(self):
        feat_funcs = [
            partial(self._feats2d_list[i].apply, self._params_list[i])
            for i in range(self._Mp)
        ]
        return feat_funcs


class RandomFeatureSpaceXV(nn.Module):
    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    Jn: int
    scale: float
    activation: Callable

    def setup(self):
        super().setup()
        self._dim = 2
        self._psi = psi
        self._mesh = UniformMeshXV(domain=self.domain, strides=self.strides)
        self._center = self._mesh.center_of_cell
        self._radius = self._mesh.radius_of_cell
        self._Mp = self._mesh.number_of_cells

        feats_list, params_list = [], []
        key = random.key(seedXV)
        rng_keys = random.split(key, self._Mp)
        for i in range(self._Mp):
            feats = RandomFeatureFunctionsXV(
                domain=self.domain,
                strides=self.strides,
                query_index=i,
                Jn=self.Jn,
                scale=self.scale,
                activation=self.activation,
            )
            # rng_key = random.key(
            #     i + self._Mp * (self._dim - 1)
            # )  # NOTE: This is not a good practice for choosing the random key
            dummy_x, dummy_v = jnp.split(
                jnp.empty((self._dim,)), self._dim, axis=0)
            params = feats.init(rng_keys[i], dummy_x, dummy_v)
            feats_list.append(feats)
            params_list.append(params)

        self._feats2d_list = feats_list
        self._params_list = params_list

    def __call__(
        self, invar_x: jnp.ndarray, invar_v: jnp.ndarray, has_aux=False
    ) -> jnp.ndarray:
        results = []
        psi_norm = 0.0

        for i, param, model in zip(
            range(self._Mp), self._params_list, self._feats2d_list
        ):
            center_x, center_v = self._center(i)

            # 对 v < 0 做对称处理
            v_rel = invar_v - jnp.where(invar_v >= 0, center_v, -center_v)

            psi = self._psi((invar_x - center_x) / self._radius[0]) * self._psi(
                v_rel / self._radius[1]
            )
            psi_norm += psi

            outvar = model.apply(param, invar_x, invar_v)  # (Jn,)
            outvar = psi * outvar
            results.append(outvar)

        outvar = jnp.array(results).squeeze()
        outvar = jnp.where(psi_norm > 0, outvar / psi_norm, 0.0)

        if has_aux:
            return outvar, self._get_operators()
        else:
            return outvar

    def _get_operators(self):
        feat_funcs = [
            partial(self._feats2d_list[i].apply, self._params_list[i])
            for i in range(self._Mp)
        ]
        return feat_funcs


class RandomFeatureFunctionsX(nn.Module):
    domain: Dict[str, Tuple[jnp.ndarray]]
    strides: Tuple[jnp.ndarray]
    query_index: int
    Jn: int
    scale: float
    activation: Callable

    def setup(self):
        self._mesh = UniformMeshX(domain=self.domain, strides=self.strides)
        self._radius = self._mesh.radius_of_cell
        self._center = self._mesh.center_of_cell(self.query_index)
        self._transform_fn = partial(
            transform, invar_center=self._center, radius=self._radius
        )

        self._feat_layer = nn.Dense(
            features=self.Jn,
            kernel_init=uniform(self.scale),
            bias_init=uniform(self.scale),
        )
        self._acti_fn = self.activation

    def __call__(self, invar_x: jnp.ndarray):
        invar = self._transform_fn(invar_x)
        invar = self._feat_layer(invar)
        invar = self._acti_fn(invar)
        return invar


class RandomFeatureFunctionsXY(nn.Module):
    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Tuple[jnp.ndarray, jnp.ndarray]
    query_index: int
    Jn: int
    scale: float
    activation: Callable

    def setup(self):
        self._mesh = UniformMeshXY(domain=self.domain, strides=self.strides)
        self._radius = self._mesh.radius_of_cell
        self._center = self._mesh.center_of_cell(self.query_index)
        self._transform_fn = partial(
            transform, invar_center=self._center, radius=self._radius
        )

        self._feat_layer = nn.Dense(
            features=self.Jn,
            kernel_init=uniform(self.scale),
            bias_init=uniform(self.scale),
        )
        self._acti_fn = self.activation

    def __call__(self, invar_x: jnp.ndarray, invar_y: jnp.ndarray):
        invar = jnp.concatenate([invar_x, invar_y])
        invar = self._transform_fn(invar)
        invar = self._feat_layer(invar)
        invar = self._acti_fn(invar)
        return invar


class RandomFeatureFunctionsXV(nn.Module):
    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Tuple[jnp.ndarray, jnp.ndarray]
    query_index: int
    Jn: int
    scale: float
    activation: Callable

    def setup(self):
        self._mesh = UniformMeshXV(domain=self.domain, strides=self.strides)
        self._radius = self._mesh.radius_of_cell
        self._center = self._mesh.center_of_cell(self.query_index)
        self._transform_fn = partial(
            lambda invar, center, radius: (invar - center) / radius,
            center=self._center,
            radius=self._radius,
        )

        self._feat_layer = nn.Dense(
            features=self.Jn,
            kernel_init=uniform(self.scale),
            bias_init=uniform(self.scale),
        )  # NOTE: bias != 0
        self._acti_fn = self.activation

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_v: jnp.ndarray,
        center_v: float | None = None,
    ):
        v_center = center_v if center_v is not None else self._center[1]
        center = jnp.array([self._center[0], v_center])
        invar = jnp.concatenate([invar_x, invar_v], axis=-1)
        invar = self._transform_fn(invar, center=center, radius=self._radius)
        invar = self._feat_layer(invar)
        invar = self._acti_fn(invar)
        return invar


class RandomFeatureFunctionsXYV(nn.Module):
    domain: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    strides: Tuple[jnp.ndarray, jnp.ndarray]
    query_index: int
    Jn: int
    scale: float
    activation: Callable

    def setup(self):
        self._mesh = UniformMeshXYV(domain=self.domain, strides=self.strides)
        self._radius = self._mesh.radius_of_cell  # (rx, ry, rtheta)
        self._center = self._mesh.center_of_cell(
            self.query_index)  # (x, y, theta)

        # 将输入坐标归一化 transform 函数
        self._transform_fn = partial(
            lambda invar, center, radius: (invar - center) / radius,
            center=self._center,
            radius=self._radius,
        )

        # 特征映射层
        self._feat_layer = nn.Dense(
            features=self.Jn,
            kernel_init=uniform(self.scale),
            bias_init=uniform(self.scale),
        )
        self._acti_fn = self.activation

    def __call__(
        self,
        invar_x: jnp.ndarray,
        invar_y: jnp.ndarray,
        invar_theta: jnp.ndarray,
        center_theta: float | None = None,
    ):
        theta_center = center_theta if center_theta is not None else self._center[2]
        theta_center = jnp.squeeze(theta_center)
        center = jnp.array([self._center[0], self._center[1], theta_center])

        invar = jnp.concatenate([invar_x, invar_y, invar_theta], axis=-1)
        invar = self._transform_fn(invar, center=center, radius=self._radius)
        out = self._feat_layer(invar)
        return self._acti_fn(out)


def transform(
    invar: jnp.ndarray, invar_center: jnp.ndarray, radius: float
) -> jnp.ndarray:
    return (invar - invar_center) / radius
