"""Provide test streaming least squares functionality for the test layer.

Distinction: This file is the canonical implementation in this module family; numerical logic is preserved.
"""

import numpy as np
from scipy.linalg import lstsq

from solver.streaming_least_squares import StreamingQRLeastSquares


def _normalized_dense_solve(A, b, *, damping=0.0):
    A = np.asarray(A, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).reshape(-1, 1).copy()
    row_scale = np.max(np.abs(A), axis=1, keepdims=True)
    row_scale[row_scale <= 1e-30] = 1.0
    A /= row_scale
    b /= row_scale
    column_scale = np.linalg.norm(A, axis=0)
    column_scale[column_scale <= 1e-14] = 1.0
    A /= column_scale[None, :]
    if damping:
        A = np.vstack((A, np.sqrt(damping) * np.eye(A.shape[1])))
        b = np.vstack((b, np.zeros((column_scale.size, 1))))
    x, *_ = lstsq(A, b, lapack_driver="gelsd")
    return x / column_scale[:, None]


def test_streaming_qr_matches_dense_lstsq():
    rng = np.random.default_rng(7)
    A = rng.normal(size=(97, 12))
    b = rng.normal(size=(97, 1))
    expected = _normalized_dense_solve(A, b)

    stream = StreamingQRLeastSquares(A.shape[1])
    for start in range(0, A.shape[0], 13):
        stream.add(A[start : start + 13], b[start : start + 13])
    actual, diagnostics = stream.solve()

    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    assert diagnostics["rank"] == A.shape[1]
    assert diagnostics["num_rows"] == A.shape[0]


def test_streaming_qr_matches_damped_dense_lstsq():
    rng = np.random.default_rng(11)
    A = rng.normal(size=(81, 9))
    b = rng.normal(size=81)
    damping = 2.5e-4
    expected = _normalized_dense_solve(A, b, damping=damping)

    stream = StreamingQRLeastSquares(A.shape[1])
    stream.add(A[:40], b[:40])
    stream.add(A[40:], b[40:])
    actual, _ = stream.solve(damping=damping)

    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
