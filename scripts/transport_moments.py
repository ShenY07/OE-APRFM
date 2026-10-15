"""Quadrature-weighted flux and microscopic-error diagnostics."""

import numpy as np


def trap_weights(x):
    x = np.asarray(x)
    assert len(x) > 1 and np.all(np.diff(x) > 0)
    w = np.empty_like(x, dtype=float)
    w[0] = (x[1] - x[0]) / 2
    w[-1] = (x[-1] - x[-2]) / 2
    w[1:-1] = (x[2:] - x[:-2]) / 2
    return w


def compute(f, truth, aw, directions, sw, epsilon, zero_truth=False):
    aw = np.asarray(aw, dtype=float)
    aw = aw / aw.sum()
    directions = np.asarray(directions).reshape(len(aw), -1)
    rho = f @ aw
    rho_true = truth @ aw
    g = (f - rho[..., None]) / epsilon
    gt = (truth - rho_true[..., None]) / epsilon
    flux = np.einsum("...q,qd,q->...d", f, directions, aw) / epsilon
    ft = np.einsum("...q,qd,q->...d", truth, directions, aw) / epsilon
    if zero_truth:
        gt = np.zeros_like(gt)
        ft = np.zeros_like(ft)
    gnorm = float(np.sqrt(np.sum(sw * np.sum(gt**2 * aw, axis=-1))))
    fnorm = float(np.sqrt(np.sum(sw * np.sum(ft**2, axis=-1))))
    ge = float(np.sqrt(np.sum(sw * np.sum((g - gt) ** 2 * aw, axis=-1))))
    fe = float(np.sqrt(np.sum(sw * np.sum((flux - ft) ** 2, axis=-1))))
    result = dict(
        flux_absolute_l2=fe,
        g_absolute_l2=ge,
        flux_reference_l2=fnorm,
        g_reference_l2=gnorm,
        flux_relative_l2=fe / fnorm if fnorm > 1e-14 else None,
        g_relative_l2=ge / gnorm if gnorm > 1e-14 else None,
    )
    opposite = np.argmin(
        np.sum(
            (directions[:, None, :] + directions[None, :, :]) ** 2, axis=-1
        ),
        axis=1,
    )
    assert np.allclose(directions[opposite], -directions, atol=1e-12)
    assert np.allclose(aw[opposite], aw, atol=1e-12)
    dg = g - gt
    even = (dg + dg[..., opposite]) / 2
    odd = (dg - dg[..., opposite]) / 2
    result["g_even_error_l2"] = float(
        np.sqrt(np.sum(sw * np.sum(even**2 * aw, axis=-1)))
    )
    result["g_odd_error_l2"] = float(
        np.sqrt(np.sum(sw * np.sum(odd**2 * aw, axis=-1)))
    )
    assert np.isclose(
        ge**2,
        result["g_even_error_l2"] ** 2 + result["g_odd_error_l2"] ** 2,
        rtol=1e-9,
        atol=1e-20,
    )
    return result
