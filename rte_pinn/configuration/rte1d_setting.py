import ml_collections

import numpy as np
import torch


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.domain = dict(x=(0, 1), v=(0, 1))  # kn = 1e-3
    config.xmin, config.xmax = config.domain["x"]
    config.vmin, config.vmax = config.domain["v"]

    config.rte = dict(
        kn=1.0e-1,
        bdy_left=lambda v: 1.0,
        bdy_right=lambda v: 0.0,
        num_vquads=16,
        # source=lambda x, v: -v/config.rte.kn,
        source=lambda x, v: torch.zeros_like(v),
        sigma_a=lambda x: 0.0,
        # sigma_s=lambda x: 1e-2 + 0.5 *
        # (torch.tanh(6.5 - 11 * x) + torch.tanh(11 * x - 4.5))
        sigma_s=lambda x: 1.0,
    )
    config.model = dict(
        fn_r=dict(
            input_size=2,
            hidden_sizes=[32] * 6,
            output_size=1,
        ),  # O(1)
        fn_j=dict(
            input_size=2,
            hidden_sizes=[32] * 6,
            output_size=1,
        ),  # O(kn)
        device_ids=[0],
        dataset=dict(
            interior_samples=4096,
            boundary_left_samples=256,
            boundary_right_samples=256,
        ),
        regularizers=[1.0, 1.0, 1.0, 10.0],
        iteration_steps=30000,
        Adam={"lr": 1e-3, "step_size": 2500, "gamma": 0.96},
    )

    return config
