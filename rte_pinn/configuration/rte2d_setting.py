import ml_collections

import numpy as np
import torch


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.domain = dict(x=(-1, 1), y=(-1, 1), theta=(0, torch.pi / 2))
    config.xmin, config.xmax = config.domain["x"]
    config.ymin, config.ymax = config.domain["y"]
    config.thetamin, config.thetamax = config.domain["theta"]

    config.rte = dict(
        kn=1.0e0,
        freq=1,
        bdy_left=lambda y: torch.zeros_like(y),
        bdy_right=lambda y: torch.zeros_like(y),
        bdy_bottom=lambda x: torch.zeros_like(x),
        bdy_top=lambda x: torch.zeros_like(x),
        # bdy_left=lambda y: torch.exp(1 - y),
        # bdy_right=lambda y: torch.exp(-1 - y),
        # bdy_bottom=lambda x: torch.exp(1 - x),
        # bdy_top=lambda x: torch.exp(-1 - x),
        num_vquads=16,
        # source=lambda x, y, theta: (
        #     -(torch.cos(theta) + torch.sin(theta))
        #     * torch.exp(-(x+y))
        #     / config.rte["kn"]
        # ),
        source=lambda x, y, theta: torch.ones_like(theta) / 2,
        sigma_a=lambda x: 0.0,
        sigma_s=lambda x: 1.0,
    )
    config.model = dict(
        fn_r1=dict(
            input_size=4,
            hidden_sizes=[32] * 6,
            output_size=1,
        ),  # O(1)
        fn_j1=dict(
            input_size=4,
            hidden_sizes=[32] * 6,
            output_size=1,
        ),  # O(kn)
        fn_r2=dict(
            input_size=4,
            hidden_sizes=[32] * 6,
            output_size=1,
        ),  # O(1)
        fn_j2=dict(
            input_size=4,
            hidden_sizes=[32] * 6,
            output_size=1,
        ),  # O(kn)
        device_ids=[2],
        dataset=dict(
            interior_samples=4096,
            boundary_left_samples=256,
            boundary_right_samples=256,
            boundary_bottom_samples=256,
            boundary_top_samples=256,
        ),
        regularizers=[1.0, 1.0, 1.0, 1.0, 1.0, 10.0],
        iteration_steps=50000,
        Adam={"lr": 1e-3, "step_size": 2500, "gamma": 0.96},
    )

    return config
