"""Load a saved OE-APNN pair for parity-preserving CPU inference."""

import json
from pathlib import Path

import numpy as np
import torch

from baselines.oe_apnn.ap_eqn import build_solver
from baselines.oe_apnn.config import get_config
from baselines.oe_apnn.networks import build_networks


def load_model(directory):
    directory = Path(directory)
    record = json.loads((directory / "config.json").read_text())
    config = get_config(record["problem"].lower(), record["epsilon"])
    config.model.device = "cpu"
    nets = build_networks(config)
    for name, net in nets.items():
        net.double()
        state = torch.load(
            directory / f"{name}.pt", map_location="cpu", weights_only=True
        )
        net.load_state_dict(state)
        net.eval()
    solver = build_solver(config)
    solver.nodes, solver.weights = (
        torch.from_numpy(values)
        for values in np.polynomial.legendre.leggauss(config.rte.num_vquads)
    )
    return solver, nets
