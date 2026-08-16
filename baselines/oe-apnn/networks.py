"""Neural representation of the two OE variables r and j."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width), nn.Tanh(), nn.Linear(width, width)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(value + self.layers(value))


class PeriodicResNet(nn.Module):
    """Residual MLP; positive frequency enables spatial Fourier features."""

    def __init__(
        self,
        spatial_dim: int,
        angular_dim: int,
        hidden_sizes: Iterable[int],
        frequency: int = 1,
    ):
        super().__init__()
        widths = list(hidden_sizes)
        if not widths or len(set(widths)) != 1:
            raise ValueError("hidden_sizes must be a nonempty constant-width sequence")
        self.spatial_dim = spatial_dim
        self.angular_dim = angular_dim
        self.frequency = frequency
        input_size = (
            2 * spatial_dim * frequency + angular_dim
            if frequency > 0
            else spatial_dim + angular_dim
        )
        self.input = nn.Linear(input_size, widths[0])
        self.blocks = nn.ModuleList(ResidualBlock(widths[0]) for _ in widths)
        self.output = nn.Linear(widths[0], 1)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != self.spatial_dim + self.angular_dim:
            raise ValueError("incorrect number of spatial/angular inputs")
        features = []
        for coordinate in inputs[: self.spatial_dim]:
            if self.frequency > 0:
                for k in range(1, self.frequency + 1):
                    phase = 2.0 * math.pi * k * coordinate
                    features.extend((torch.cos(phase), torch.sin(phase)))
            else:
                features.append(coordinate)
        features.extend(inputs[self.spatial_dim :])
        value = torch.tanh(self.input(torch.cat(features, dim=-1)))
        for block in self.blocks:
            value = block(value)
        return self.output(value)


def build_networks(config: object) -> dict[str, nn.Module]:
    dim = int(config.problem.dimension)
    angular_dim = 1 if dim == 1 else 2
    arguments = (
        dim,
        angular_dim,
        list(config.model.hidden_sizes),
        int(config.rte.freq),
    )
    return {"r": PeriodicResNet(*arguments), "j": PeriodicResNet(*arguments)}
