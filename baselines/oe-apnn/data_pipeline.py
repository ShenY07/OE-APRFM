"""Interior and inflow-boundary collocation sampling for OE-APNN."""

from __future__ import annotations

import math

import torch
from torch.utils.data import IterableDataset


class CollocationDataset(IterableDataset):
    def __init__(self, config: object, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

    def _uniform(self, interval, count):
        low, high = map(float, interval)
        return low + (high - low) * torch.rand(count, 1, device=self.device)

    def _boundary_1d(self, count):
        half = count // 2
        left_count = half
        right_count = count - half
        return {
            "x": torch.cat(
                (
                    torch.full((left_count, 1), float(self.config.domain.x[0]), device=self.device),
                    torch.full((right_count, 1), float(self.config.domain.x[1]), device=self.device),
                )
            ),
            "v": torch.cat(
                (self._uniform((0.0, 1.0), left_count), -self._uniform((0.0, 1.0), right_count))
            ),
            "side": torch.cat(
                (torch.zeros(left_count, 1, device=self.device), torch.ones(right_count, 1, device=self.device))
            ),
        }

    def _boundary_2d(self, count):
        counts = [count // 4] * 4
        for index in range(count % 4):
            counts[index] += 1
        xmin, xmax = map(float, self.config.domain.x)
        ymin, ymax = map(float, self.config.domain.y)
        # left/right/bottom/top directions are sampled only on their inflow half-circle.
        theta_intervals = (
            (-math.pi / 2.0, math.pi / 2.0),
            (math.pi / 2.0, 3.0 * math.pi / 2.0),
            (0.0, math.pi),
            (math.pi, 2.0 * math.pi),
        )
        xs, ys, angles = [], [], []
        for side, n in enumerate(counts):
            if side < 2:
                x = torch.full((n, 1), xmin if side == 0 else xmax, device=self.device)
                y = self._uniform((ymin, ymax), n)
            else:
                x = self._uniform((xmin, xmax), n)
                y = torch.full((n, 1), ymin if side == 2 else ymax, device=self.device)
            xs.append(x)
            ys.append(y)
            angles.append(self._uniform(theta_intervals[side], n))
        if self.config.problem.name == "p4":
            n = max(4, count // 2)
            polar = self._uniform((0.0, 2.0 * math.pi), n)
            normal = polar + math.pi
            offset = self._uniform((-0.49 * math.pi, 0.49 * math.pi), n)
            xs.append(0.5 * torch.cos(polar))
            ys.append(0.5 * torch.sin(polar))
            angles.append(normal + offset)
        elif self.config.problem.name == "p5":
            # Four inflow edges of the central square hole [-1/3,1/3]^2.
            n = max(4, count // 2)
            inner_counts = [n // 4] * 4
            for index in range(n % 4):
                inner_counts[index] += 1
            specs = ((0, 1/3, 0.0), (0, -1/3, math.pi),
                     (1, 1/3, math.pi/2), (1, -1/3, -math.pi/2))
            for index, (coord, value, inflow_center) in enumerate(specs):
                m = inner_counts[index]
                tangent = self._uniform((-1/3, 1/3), m)
                if coord == 0:
                    xs.append(torch.full((m, 1), value, device=self.device)); ys.append(tangent)
                else:
                    xs.append(tangent); ys.append(torch.full((m, 1), value, device=self.device))
                angles.append(inflow_center + self._uniform((-0.49 * math.pi, 0.49 * math.pi), m))
        return {"x": torch.cat(xs), "y": torch.cat(ys), "theta": torch.cat(angles)}

    def __iter__(self):
        config = self.config
        n_i = int(config.model.dataset.interior_samples)
        n_b = int(config.model.dataset.boundary_samples)
        while True:
            interior = {"x": self._uniform(config.domain.x, n_i)}
            if int(config.problem.dimension) == 1:
                interior["v"] = self._uniform(config.domain.v, n_i)
                boundary = self._boundary_1d(n_b)
            else:
                interior["y"] = self._uniform(config.domain.y, n_i)
                if config.problem.name == "p4":
                    inside = interior["x"].square() + interior["y"].square() < 0.25
                    while torch.any(inside):
                        count = int(inside.sum())
                        interior["x"][inside] = self._uniform(config.domain.x, count).reshape(-1)
                        interior["y"][inside] = self._uniform(config.domain.y, count).reshape(-1)
                        inside = interior["x"].square() + interior["y"].square() < 0.25
                elif config.problem.name == "p5":
                    inside = (interior["x"].abs() <= 1/3) & (interior["y"].abs() <= 1/3)
                    while torch.any(inside):
                        count = int(inside.sum())
                        interior["x"][inside] = self._uniform(config.domain.x, count).reshape(-1)
                        interior["y"][inside] = self._uniform(config.domain.y, count).reshape(-1)
                        inside = (interior["x"].abs() <= 1/3) & (interior["y"].abs() <= 1/3)
                # Use both independent directions theta and pi-theta while the
                # configured base angular interval remains [0, pi/2].
                base = self._uniform(config.domain.theta, (n_i + 1) // 2)
                interior["theta"] = torch.cat((base, math.pi - base))[:n_i]
                boundary = self._boundary_2d(n_b)
            yield {"interior": interior, "boundary": boundary}
