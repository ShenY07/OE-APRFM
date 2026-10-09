import ml_collections

import numpy as np
import torch


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.domain = dict(
        x=(0, 1), y=(0, 1), theta=(0, torch.pi / 2), t=(0, 0.1)
    )  # kn = 1e-3
    config.xmin, config.xmax = config.domain["x"]
    config.ymin, config.ymax = config.domain["y"]
    config.thetamin, config.thetamax = config.domain["theta"]
    config.tmin, config.tmax = config.domain["t"]

    config.rte = dict(
        kn=1.0e-3,
        freq=1,
        f_init=lambda x, y, theta: (
            1.0
            + (torch.cos(2.0 * torch.pi * x) + torch.cos(2.0 * torch.pi * y))
            / 2.0
        )
        / np.sqrt(2.0 * torch.pi)
        * np.exp(-0.5),
        rho_init=lambda x, y: (
            1.0
            + (torch.cos(2.0 * torch.pi * x) + torch.cos(2.0 * torch.pi * y))
            / 2.0
        )
        / np.sqrt(2.0 * torch.pi)
        * np.exp(-0.5),
        psi_init=lambda x, y, theta: 0.0,
        j1_init=lambda x, y, theta: 0.0,
        j2_init=lambda x, y, theta: 0.0,
        w_init=lambda x, y, theta: 0.0,
        num_vquads=16,
    )
    config.model = dict(
        fn_rho=dict(
            input_size=1 + 4 * config.rte.freq,
            hidden_sizes=[64] * 4,
            output_size=1,
        ),  # O(1)
        fn_psi=dict(
            input_size=3 + 4 * config.rte.freq,
            hidden_sizes=[64] * 4,
            output_size=1,
        ),  # O(1)
        fn_j1=dict(
            input_size=3 + 4 * config.rte.freq,
            hidden_sizes=[64] * 4,
            output_size=1,
        ),  # O(kn)
        fn_j2=dict(
            input_size=3 + 4 * config.rte.freq,
            hidden_sizes=[64] * 4,
            output_size=1,
        ),  # O(kn)
        fn_w=dict(
            input_size=3 + 4 * config.rte.freq,
            hidden_sizes=[64] * 4,
            output_size=1,
        ),  # O(kn^2)
        device_ids=[0],
        dataset=dict(
            interior_samples=4096,
            initial_samples=1024,
        ),
        regularizers=[1.0, 1.0, 1.0, 1.0, 1.0, 10.0],
        iteration_steps=200000,
        Adam={"lr": 1e-3, "step_size": 2500, "gamma": 0.96},
    )

    return config
