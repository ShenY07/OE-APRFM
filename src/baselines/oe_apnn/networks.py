"""Locked compact residual networks used by the OE-APNN comparison."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class PeriodicResNet(nn.Module):
    """Compatibility name for the archived compact residual MLP.

    The network consumes ``(x,v)`` in 1D or ``(x,y,vx,vy)`` in 2D.  Spatial
    and velocity coordinates are already in [-1,1], except 1D x which is
    scaled here.  State-dict keys intentionally match the trained archive.
    """

    def __init__(
        self,
        spatial_dim: int,
        angular_dim: int,
        hidden_sizes: Iterable[int],
        frequency: int = 0,
    ):
        super().__init__()
        widths = list(hidden_sizes)
        if widths != [64] * 4:
            raise ValueError(
                "locked OE-APNN requires four width-64 transforms"
            )
        if frequency != 0:
            raise ValueError("locked OE-APNN does not use Fourier features")
        self.spatial_dim = spatial_dim
        self.angular_dim = angular_dim
        input_size = spatial_dim + angular_dim
        self.input = nn.Linear(input_size, 64)
        self.residual = nn.ModuleList(nn.Linear(64, 64) for _ in range(3))
        self.output = nn.Linear(64, 1)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != self.spatial_dim + self.angular_dim:
            raise ValueError("incorrect number of spatial/angular inputs")
        values = list(inputs)
        if self.spatial_dim == 1:
            values[0] = 2.0 * values[0] - 1.0
        value = torch.tanh(self.input(torch.cat(values, dim=-1)))
        for layer in self.residual:
            value = value + torch.tanh(layer(value))
        return self.output(value)


def build_networks(config: object) -> dict[str, nn.Module]:
    dimension = int(config.problem.dimension)
    angular_dim = 1 if dimension == 1 else 2
    arguments = (
        dimension,
        angular_dim,
        list(config.model.hidden_sizes),
        int(config.rte.freq),
    )
    return {"r": PeriodicResNet(*arguments), "j": PeriodicResNet(*arguments)}


def parameter_count(networks: dict[str, nn.Module]) -> int:
    return sum(
        parameter.numel()
        for net in networks.values()
        for parameter in net.parameters()
    )
