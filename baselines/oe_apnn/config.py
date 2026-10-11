"""P1--P5 configurations for the OE-APNN/OE-APRFM PDE comparison."""

from __future__ import annotations

import math

import ml_collections
import torch


def _base(problem: str, dimension: int, epsilon: float):
    config = ml_collections.ConfigDict()
    config.problem = dict(dimension=dimension, name=problem)
    config.rte = dict(kn=float(epsilon), freq=0, num_vquads=16)
    config.model = dict(
        hidden_sizes=[64] * 4,
        device_ids=[0],
        dataset=dict(interior_samples=4096, boundary_samples=1024),
        regularizers=dict(macro=1.0, even=1.0, odd=1.0, boundary=1.0),
        iteration_steps=20000,
        validation_interval=200,
        Adam=dict(lr=1.0e-3),
    )
    config.protocol = dict(
        name="oe-apnn-p1-p5-v2-hard-parity",
        parity_projection="antipodal_even_r_odd_j",
        dtype="float64",
        seeds=(7, 11, 17),
        epsilon_values=(1.0, 1.0e-3),
        checkpoint_selection="minimum_fixed_validation_physics_loss",
        evaluation_is_independent=True,
    )
    return config


def _one(value):
    return torch.ones_like(value)


def _zero(value):
    return torch.zeros_like(value)


def _p1(epsilon):
    config = _base("p1", 1, epsilon)
    config.domain = dict(x=(0.0, 1.0), v=(0.0, 1.0))
    config.rte.scattering, config.rte.absorption = _one, _zero
    config.rte.source = lambda x, v: -v / epsilon
    config.rte.boundary_left = lambda v: torch.ones_like(v)
    config.rte.boundary_right = lambda v: torch.zeros_like(v)
    config.rte.exact = lambda x, v: 1.0 - x + 0.0 * v
    return config


def _p2(epsilon):
    config = _base("p2", 1, epsilon)
    config.domain = dict(x=(0.0, 1.0), v=(0.0, 1.0))
    config.rte.scattering = lambda x: 0.01 + 0.5 * (
        torch.tanh(6.5 - 11.0 * x) + torch.tanh(11.0 * x - 4.5)
    )
    config.rte.absorption = _zero
    config.rte.source = lambda x, v: torch.zeros_like(v)
    config.rte.boundary_left = lambda v: 0.5 * torch.ones_like(v)
    config.rte.boundary_right = lambda v: torch.zeros_like(v)
    return config


def _two_dimensional(problem, epsilon):
    config = _base(problem, 2, epsilon)
    config.domain = dict(
        x=(-1.0, 1.0), y=(-1.0, 1.0), theta=(0.0, math.pi / 2.0)
    )
    if problem == "p3":
        exact = lambda x, y: torch.exp(-x - y)
        config.rte.scattering, config.rte.absorption = lambda x, y: _one(
            x
        ), lambda x, y: _zero(x)
        config.rte.source = (
            lambda x, y, theta: -(torch.cos(theta) + torch.sin(theta))
            * exact(x, y)
            / epsilon
        )
    elif problem == "p4":
        exact = lambda x, y: 1.0 / (1.0 + x.square() + y.square())
        config.rte.scattering, config.rte.absorption = lambda x, y: _one(
            x
        ), lambda x, y: _zero(x)
        config.rte.source = (
            lambda x, y, theta: -2.0
            * (x * torch.cos(theta) + y * torch.sin(theta))
            / (epsilon * (1.0 + x.square() + y.square()).square())
        )
    else:
        exact = lambda x, y: torch.exp(-x - y)
        config.rte.scattering, config.rte.absorption = lambda x, y: _one(
            x
        ), lambda x, y: _zero(x)
        config.rte.source = (
            lambda x, y, theta: -(torch.cos(theta) + torch.sin(theta))
            * exact(x, y)
            / epsilon
        )
    config.rte.exact = exact
    config.rte.boundary_value = exact
    return config


def get_config(problem: str = "p3", epsilon: float = 1.0e-3):
    if problem == "p1":
        return _p1(epsilon)
    if problem == "p2":
        return _p2(epsilon)
    if problem in ("p3", "p4", "p5"):
        return _two_dimensional(problem, epsilon)
    raise ValueError(f"unknown problem: {problem}")
