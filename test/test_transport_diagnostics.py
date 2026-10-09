"""Analytical moment tests independent of saved benchmark solutions."""

import importlib.util
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss

spec = importlib.util.spec_from_file_location(
    "diagnostics",
    Path(__file__).parents[1] / "scripts/build_transport_diagnostics.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_nonzero_slab_moments():
    v, w = leggauss(16)
    x = np.linspace(0, 1, 33)
    sw = m.trap_weights(x)
    eps = 0.001
    true = 2 + eps * np.broadcast_to(v, (len(x), len(v)))
    pred = 2 + 1.2 * eps * np.broadcast_to(v, true.shape)
    r = m.compute(pred, true, w, v[:, None], sw, eps)
    np.testing.assert_allclose(r["flux_reference_l2"], 1 / 3, rtol=1e-10)
    np.testing.assert_allclose(r["g_reference_l2"], 1 / np.sqrt(3), rtol=1e-10)
    np.testing.assert_allclose(
        [r["flux_relative_l2"], r["g_relative_l2"]], 0.2, rtol=1e-10
    )


def test_zero_truth_and_two_dimensional_vector_flux():
    theta = np.arange(64) * 2 * np.pi / 64
    dirs = np.column_stack((np.cos(theta), np.sin(theta)))
    eps = 0.001
    true = np.ones((3, 3, 64))
    pred = true + eps * (dirs[:, 0] + 2 * dirs[:, 1])
    sw = np.ones((3, 3)) / 9
    r = m.compute(pred, true, np.ones(64), dirs, sw, eps, zero_truth=True)
    assert r["flux_relative_l2"] is None and r["g_relative_l2"] is None
    np.testing.assert_allclose(
        r["flux_absolute_l2"], np.sqrt(1.25), rtol=1e-10
    )
    np.testing.assert_allclose(r["g_absolute_l2"], np.sqrt(2.5), rtol=1e-10)


if __name__ == "__main__":
    test_nonzero_slab_moments()
    test_zero_truth_and_two_dimensional_vector_flux()
    print(
        "PASS: analytic slab and vector moments; zero-truth relative errors omitted"
    )
