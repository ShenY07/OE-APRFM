"""Contract tests for the locked OE-APNN baseline."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

BASELINE = Path(__file__).resolve().parents[1] / "baselines/oe-apnn"
sys.path.insert(0, str(BASELINE))

from ap_eqn import build_solver  # noqa: E402
from config import get_config  # noqa: E402
from data_pipeline import CollocationDataset  # noqa: E402
from networks import build_networks, parameter_count  # noqa: E402


@pytest.mark.parametrize("problem", ["p1", "p2", "p3", "p4", "p5"])
def test_hard_parity_derivatives_reconstruction_and_residuals(problem):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(7)
    config = get_config(problem, 1.e-3)
    config.model.dataset.interior_samples = 64
    config.model.dataset.boundary_samples = 16
    solver = build_solver(config)
    nets = {k: n.to(solver.device) for k, n in build_networks(config).items()}
    x = torch.linspace(.1, .9, 8, device=solver.device).reshape(-1, 1).requires_grad_()
    angle = torch.linspace(-2., 2., 8, device=solver.device).reshape(-1, 1)
    if config.problem.dimension == 1:
        args, opposite = (x, angle), (x, -angle)
    else:
        y = x.square()
        args, opposite = (x, y, angle), (x, y, angle + np.pi)
    r, rm = solver.model_r(nets['r'], *args), solver.model_r(nets['r'], *opposite)
    j, jm = solver.model_j(nets['j'], *args), solver.model_j(nets['j'], *opposite)
    torch.testing.assert_close(r, rm, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(j, -jm, atol=1e-12, rtol=1e-12)
    for a, b in ((r, rm), (j, -jm)):
        da = torch.autograd.grad(a.sum(), x, retain_graph=True)[0]
        db = torch.autograd.grad(b.sum(), x, retain_graph=True)[0]
        torch.testing.assert_close(da, db, atol=1e-12, rtol=1e-12)
    f, fm = solver.reconstruct(nets, *args), solver.reconstruct(nets, *opposite)
    torch.testing.assert_close((f + fm) / 2, r, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close((f - fm) / (2 * solver.epsilon), j, atol=1e-10, rtol=1e-10)
    batch = next(iter(CollocationDataset(config, solver.device)))
    losses = list(solver.phase_residual(nets, batch['interior']).values())
    losses += list(solver.boundary_residual(nets, batch['boundary']).values())
    loss = sum(value.square().mean() for value in losses)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for net in nets.values() for p in net.parameters())


def test_locked_architecture_and_protocol():
    for problem, expected in (
        ("p1", 25474),
        ("p2", 25474),
        ("p3", 25730),
        ("p4", 25730),
        ("p5", 25730),
    ):
        config = get_config(problem, 1.0e-3)
        assert parameter_count(build_networks(config)) == expected
        assert tuple(config.protocol.seeds) == (7, 11, 17)
        assert config.protocol.dtype == "float64"
        assert dict(config.model.regularizers) == {
            "boundary": 1.0,
            "even": 1.0,
            "macro": 1.0,
            "odd": 1.0,
        }


def test_p1_manufactured_solution_satisfies_all_residuals():
    torch.set_default_dtype(torch.float64)
    config = get_config("p1", 1.0e-3)
    solver = build_solver(config)

    class ExactR(torch.nn.Module):
        def forward(self, values):
            return 1.0 - values[0]

    class ExactJ(torch.nn.Module):
        def forward(self, values):
            return 0.0 * values[0]

    sample = {
        "x": torch.linspace(0.1, 0.9, 5, device=solver.device).reshape(-1, 1),
        "v": torch.linspace(0.1, 0.9, 5, device=solver.device).reshape(-1, 1),
    }
    residuals = solver.residual({"r": ExactR(), "j": ExactJ()}, sample)
    for residual in residuals.values():
        torch.testing.assert_close(
            residual, torch.zeros_like(residual), atol=1e-12, rtol=1e-12
        )


def test_stratified_boundaries_have_exact_requested_size_and_inflow():
    torch.set_default_dtype(torch.float64)
    for problem in ("p3", "p4", "p5"):
        config = get_config(problem, 1.0)
        batch = next(iter(CollocationDataset(config, torch.device("cpu"))))[
            "boundary"
        ]
        assert batch["x"].shape[0] == 1024
        velocity = torch.cat(
            (torch.cos(batch["theta"]), torch.sin(batch["theta"])), dim=1
        )
        x, y = batch["x"].squeeze(), batch["y"].squeeze()
        outer = torch.isclose(x.abs(), torch.tensor(1.0)) | torch.isclose(
            y.abs(), torch.tensor(1.0)
        )
        assert int(outer.sum()) > 0
        if problem == "p4":
            inner = ~outer
            normal = (
                -torch.cat((batch["x"][inner], batch["y"][inner]), 1) / 0.5
            )
            assert torch.all((normal * velocity[inner]).sum(1) < 0)
        if problem == "p5":
            inner = ~outer
            assert int(inner.sum()) > 0
