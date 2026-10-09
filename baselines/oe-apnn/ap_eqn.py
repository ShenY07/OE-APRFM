"""The same steady r/j odd-even AP system used by OE-APRFM."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch


def _grad(value: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        value,
        coordinate,
        torch.ones_like(value),
        create_graph=True,
        retain_graph=True,
    )[0]


class _OESolver:
    def __init__(self, config: object):
        device_id = int(config.model.device_ids[0])
        self.device = torch.device(
            f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        )
        self.epsilon = float(config.rte.kn)
        self.scattering = config.rte.scattering
        self.absorption = config.rte.absorption
        self.source = config.rte.source
        nodes, weights = np.polynomial.legendre.leggauss(
            int(config.rte.num_vquads)
        )
        self.nodes = torch.as_tensor(
            nodes, dtype=torch.get_default_dtype(), device=self.device
        )
        self.weights = torch.as_tensor(
            weights, dtype=torch.get_default_dtype(), device=self.device
        )

    @staticmethod
    def _variables(sample, names):
        return [
            sample[name].detach().clone().requires_grad_(True)
            for name in names
        ]


class OERadiativeTransferSolver1D(_OESolver):
    """1D OE-APRFM equations with only even r and scaled odd j."""

    def __init__(self, config: object):
        super().__init__(config)
        self.boundary_left = config.rte.boundary_left
        self.boundary_right = config.rte.boundary_right
        self.vquads = 0.5 * (self.nodes + 1.0)
        self.wquads = 0.5 * self.weights

    @staticmethod
    def model_r(net, x, v):
        return net([x, v])

    @staticmethod
    def model_j(net, x, v):
        return net([x, v])

    def _quadrature_inputs(self, x):
        shape = (*x.shape[:-1], self.vquads.numel(), 1)
        xq = x.unsqueeze(-2).expand(shape)
        vq = self.vquads.reshape(*([1] * (x.ndim - 1)), -1, 1).expand(shape)
        weight = self.wquads.reshape(*([1] * (x.ndim - 1)), -1, 1)
        return xq, vq, weight

    def averages(self, nets, x):
        xq, vq, weight = self._quadrature_inputs(x)
        r = self.model_r(nets["r"], xq, vq)
        j = self.model_j(nets["j"], xq, vq)
        vdj = vq * _grad(j, xq)
        q_plus, q_minus = self.source(xq, vq), self.source(xq, -vq)
        q_even = 0.5 * (q_plus + q_minus)
        integrate = lambda value: torch.sum(weight * value, dim=-2)
        return integrate(r), integrate(vdj), integrate(q_even)

    def residual(self, nets: Mapping[str, torch.nn.Module], sample):
        x, v = self._variables(sample, ("x", "v"))
        r = self.model_r(nets["r"], x, v)
        j = self.model_j(nets["j"], x, v)
        vdj = v * _grad(j, x)
        vdr = v * _grad(r, x)
        average_r, average_vdj, average_q_even = self.averages(nets, x)
        q_plus, q_minus = self.source(x, v), self.source(x, -v)
        q_even = 0.5 * (q_plus + q_minus)
        q_odd = 0.5 * (q_plus - q_minus)
        sigma_s, sigma_a = self.scattering(x), self.absorption(x)
        collision = sigma_s + self.epsilon**2 * sigma_a
        return {
            "macro": average_vdj + sigma_a * average_r - average_q_even,
            "even": self.epsilon**2 * (vdj - average_vdj)
            + collision * (r - average_r)
            - self.epsilon**2 * (q_even - average_q_even),
            "odd": collision * j + vdr - self.epsilon * q_odd,
        }

    def phase_residual(self, nets, sample):
        """Evaluate 256 spatial points x 16 fixed representative velocities."""
        xq, vq, weight = self._quadrature_inputs(sample["x"])
        xq = xq.clone().requires_grad_(True)
        r, j = self.model_r(nets["r"], xq, vq), self.model_j(nets["j"], xq, vq)
        vdj, vdr = vq * _grad(j, xq), vq * _grad(r, xq)
        q_plus, q_minus = self.source(xq, vq), self.source(xq, -vq)
        q_even, q_odd = 0.5 * (q_plus + q_minus), 0.5 * (q_plus - q_minus)
        integrate = lambda value: torch.sum(
            weight * value, dim=-2, keepdim=True
        )
        average_r, average_vdj, average_q_even = (
            integrate(r),
            integrate(vdj),
            integrate(q_even),
        )
        sigma_s, sigma_a = self.scattering(xq), self.absorption(xq)
        collision = sigma_s + self.epsilon**2 * sigma_a
        return {
            "macro": average_vdj
            + self.absorption(sample["x"][:, None, :]) * average_r
            - average_q_even,
            "even": self.epsilon**2 * (vdj - average_vdj)
            + collision * (r - average_r)
            - self.epsilon**2 * (q_even - average_q_even),
            "odd": collision * j + vdr - self.epsilon * q_odd,
        }

    def boundary_residual(self, nets, sample):
        x, v = sample["x"], sample["v"]
        approximation = self.reconstruct(nets, x, v)
        target = torch.where(
            sample["side"] < 0.5,
            self.boundary_left(torch.abs(v)),
            self.boundary_right(torch.abs(v)),
        )
        return {"inflow": approximation - target}

    def reconstruct(self, nets, x, v):
        return self.model_r(nets["r"], x, v) + self.epsilon * self.model_j(
            nets["j"], x, v
        )


class OERadiativeTransferSolver2D(_OESolver):
    """2D OE-APRFM equations with antipodal r/j symmetry."""

    def __init__(self, config: object):
        super().__init__(config)
        self.boundary_value = config.rte.boundary_value
        self.theta_quads = 0.5 * math.pi * (self.nodes + 1.0)
        self.theta_weights = 0.25 * self.weights

    @staticmethod
    def _raw(net, x, y, theta):
        return net([x, y, torch.cos(theta), torch.sin(theta)])

    def model_r(self, net, x, y, theta):
        return self._raw(net, x, y, theta)

    def model_j(self, net, x, y, theta):
        return self._raw(net, x, y, theta)

    def _quadrature_inputs(self, x, y):
        shape = (*x.shape[:-1], self.theta_quads.numel(), 1)
        xq, yq = x.unsqueeze(-2).expand(shape), y.unsqueeze(-2).expand(shape)
        base = self.theta_quads.reshape(*([1] * (x.ndim - 1)), -1, 1).expand(
            shape
        )
        theta = torch.cat((base, math.pi - base), dim=-2)
        weight = self.theta_weights.reshape(*([1] * (x.ndim - 1)), -1, 1)
        weight = torch.cat((weight, weight), dim=-2)
        xq, yq = torch.cat((xq, xq), dim=-2), torch.cat((yq, yq), dim=-2)
        return xq, yq, theta, weight

    def averages(self, nets, x, y):
        xq, yq, theta, weight = self._quadrature_inputs(x, y)
        r = self.model_r(nets["r"], xq, yq, theta)
        j = self.model_j(nets["j"], xq, yq, theta)
        vdj = torch.cos(theta) * _grad(j, xq) + torch.sin(theta) * _grad(j, yq)
        q_plus = self.source(xq, yq, theta)
        q_minus = self.source(xq, yq, theta + math.pi)
        q_even = 0.5 * (q_plus + q_minus)
        integrate = lambda value: torch.sum(weight * value, dim=-2)
        return integrate(r), integrate(vdj), integrate(q_even)

    def residual(self, nets: Mapping[str, torch.nn.Module], sample):
        x, y, theta = self._variables(sample, ("x", "y", "theta"))
        xi, eta = torch.cos(theta), torch.sin(theta)
        r = self.model_r(nets["r"], x, y, theta)
        j = self.model_j(nets["j"], x, y, theta)
        vdj = xi * _grad(j, x) + eta * _grad(j, y)
        vdr = xi * _grad(r, x) + eta * _grad(r, y)
        average_r, average_vdj, average_q_even = self.averages(nets, x, y)
        q_plus = self.source(x, y, theta)
        q_minus = self.source(x, y, theta + math.pi)
        q_even, q_odd = 0.5 * (q_plus + q_minus), 0.5 * (q_plus - q_minus)
        sigma_s, sigma_a = self.scattering(x, y), self.absorption(x, y)
        collision = sigma_s + self.epsilon**2 * sigma_a
        return {
            "macro": average_vdj + sigma_a * average_r - average_q_even,
            "even": self.epsilon**2 * (vdj - average_vdj)
            + collision * (r - average_r)
            - self.epsilon**2 * (q_even - average_q_even),
            "odd": collision * j + vdr - self.epsilon * q_odd,
        }

    def phase_residual(self, nets, sample):
        """Evaluate 128 spatial points x 32 fixed representative directions."""
        xq, yq, theta, weight = self._quadrature_inputs(
            sample["x"], sample["y"]
        )
        xq, yq = xq.clone().requires_grad_(True), yq.clone().requires_grad_(
            True
        )
        r = self.model_r(nets["r"], xq, yq, theta)
        j = self.model_j(nets["j"], xq, yq, theta)
        xi, eta = torch.cos(theta), torch.sin(theta)
        vdj = xi * _grad(j, xq) + eta * _grad(j, yq)
        vdr = xi * _grad(r, xq) + eta * _grad(r, yq)
        q_plus, q_minus = self.source(xq, yq, theta), self.source(
            xq, yq, theta + math.pi
        )
        q_even, q_odd = 0.5 * (q_plus + q_minus), 0.5 * (q_plus - q_minus)
        integrate = lambda value: torch.sum(
            weight * value, dim=-2, keepdim=True
        )
        average_r, average_vdj, average_q_even = (
            integrate(r),
            integrate(vdj),
            integrate(q_even),
        )
        sigma_s, sigma_a = self.scattering(xq, yq), self.absorption(xq, yq)
        collision = sigma_s + self.epsilon**2 * sigma_a
        return {
            "macro": average_vdj
            + self.absorption(sample["x"][:, None, :], sample["y"][:, None, :])
            * average_r
            - average_q_even,
            "even": self.epsilon**2 * (vdj - average_vdj)
            + collision * (r - average_r)
            - self.epsilon**2 * (q_even - average_q_even),
            "odd": collision * j + vdr - self.epsilon * q_odd,
        }

    def boundary_residual(self, nets, sample):
        target = self.boundary_value(sample["x"], sample["y"])
        return {
            "inflow": self.reconstruct(
                nets, sample["x"], sample["y"], sample["theta"]
            )
            - target
        }

    def reconstruct(self, nets, x, y, theta):
        return self.model_r(
            nets["r"], x, y, theta
        ) + self.epsilon * self.model_j(nets["j"], x, y, theta)


APRadiativeTransferSolver1D = OERadiativeTransferSolver1D
APRadiativeTransferSolver2D = OERadiativeTransferSolver2D


def build_solver(config: object):
    dimension = int(config.problem.dimension)
    if dimension == 1:
        return OERadiativeTransferSolver1D(config)
    if dimension == 2:
        return OERadiativeTransferSolver2D(config)
    raise ValueError("only spatial dimensions 1 and 2 are supported")
