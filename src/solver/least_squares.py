"""Provide least squares functionality for the solver layer.

Distinction: This file is the canonical implementation in this module family; numerical logic is preserved.
"""

import numpy as np
from scipy.linalg import lstsq


def solve(
    matrix_A: np.ndarray,
    vector_b: np.ndarray,
    c: float = 1.0,
    *,
    n_boundary: int | None = None,
    block_weights: tuple[float, float, float, float] | None = None,
    damping: float = 0.0,
    rcond: float | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Solve a scaled dense least-squares problem.

    The legacy call remains unchanged.  When ``n_boundary`` and
    ``block_weights`` are supplied, the rows are interpreted as one boundary
    block followed by point-major triples for equations 0, 1, and 2.  Block
    weights are applied after row normalization, so they are not cancelled by
    the normalization.  Column equilibration and optional Tikhonov damping
    improve robustness without increasing the random-feature dimension.
    """
    A = np.asarray(matrix_A, dtype=np.float64).copy()
    b = np.asarray(vector_b, dtype=np.float64).reshape(-1, 1).copy()
    if A.ndim != 2 or A.shape[0] != b.shape[0]:
        raise ValueError("matrix_A and vector_b have incompatible shapes")
    if damping < 0:
        raise ValueError("damping must be nonnegative")

    # Use the same unit-L2 row normalization as the streaming 2-D solver so
    # formulation comparisons are not confounded by different row metrics.
    row_scale = np.linalg.norm(A, axis=1, keepdims=True)
    row_scale = np.where(row_scale > 1e-30, row_scale, 1.0)
    A /= row_scale
    b /= row_scale

    if block_weights is not None:
        if n_boundary is None or not 0 < n_boundary < A.shape[0]:
            raise ValueError(
                "a valid n_boundary is required for block weighting"
            )
        n_equation_rows = A.shape[0] - n_boundary
        if n_equation_rows % 3:
            raise ValueError(
                "interior rows must be point-major equation triples"
            )
        weights = np.asarray(block_weights, dtype=np.float64)
        if weights.shape != (4,) or np.any(weights <= 0):
            raise ValueError("block_weights must contain four positive values")

        n_interior = n_equation_rows // 3
        boundary_factor = np.sqrt(weights[0] / n_boundary)
        equation_factors = np.sqrt(weights[1:] / n_interior)
        A[:n_boundary] *= boundary_factor
        b[:n_boundary] *= boundary_factor
        A_equations = A[n_boundary:].reshape(n_interior, 3, A.shape[1])
        b_equations = b[n_boundary:].reshape(n_interior, 3, 1)
        A_equations *= equation_factors[None, :, None]
        b_equations *= equation_factors[None, :, None]
        A[n_boundary:] = A_equations.reshape(-1, A.shape[1])
        b[n_boundary:] = b_equations.reshape(-1, 1)
    else:
        A *= c
        b *= c

    column_scale = np.linalg.norm(A, axis=0)
    column_scale = np.where(column_scale > 1e-14, column_scale, 1.0)
    A_scaled = A / column_scale[None, :]

    if damping:
        n_unknowns = A.shape[1]
        A_solve = np.vstack((A_scaled, np.sqrt(damping) * np.eye(n_unknowns)))
        b_solve = np.vstack((b, np.zeros((n_unknowns, 1))))
    else:
        A_solve, b_solve = A_scaled, b

    scaled_x, _, rank, singular_values = lstsq(
        A_solve, b_solve, cond=rcond, lapack_driver="gelsd"
    )
    x = scaled_x / column_scale[:, None]
    if not return_diagnostics:
        return x

    threshold = (
        rcond
        if rcond is not None
        else np.finfo(float).eps * max(A_solve.shape)
    ) * singular_values[0]
    effective = singular_values[singular_values > threshold]
    smallest_effective = (
        float(effective[-1]) if effective.size else float("nan")
    )
    diagnostics = {
        "rank": int(rank),
        "condition_number": float(singular_values[0] / smallest_effective),
        "largest_singular_value": float(singular_values[0]),
        "smallest_singular_value": float(singular_values[-1]),
        "smallest_effective_singular_value": smallest_effective,
        "singular_value_threshold": float(threshold),
        "coefficient_norm": float(np.linalg.norm(x)),
    }
    residual_l2 = float(np.linalg.norm(A @ x - b))
    diagnostics.update(
        {
            "normalized_residual_l2": residual_l2,
            "normalized_residual_rms": residual_l2 / np.sqrt(A.shape[0]),
            "relative_normalized_residual": (
                residual_l2 / float(np.linalg.norm(b))
                if np.linalg.norm(b) > 0
                else float("nan")
            ),
        }
    )
    return x, diagnostics


# def solve(
#     matrix_A: np.ndarray,
#     vector_b: np.ndarray,
#     c: float = 1.0,
#     damping: float = 0.0,
#     method: str = "auto",
# ) -> np.ndarray:
#     """
#     Solve min ||A x - b||^2 + damping * ||x||^2 with row scaling.

#     - If sparsity > 0.9 and method is auto or lsqr, use
#       scipy.sparse.linalg.lsqr with LSQR's built-in damping.
#     - Otherwise use dense lstsq; if damping > 0, solve the augmented system
#       [A; sqrt(damping) I] x ≈ [b; 0].
#     """
#     # Normalize by row-wise max values
#     max_val = np.max(np.abs(matrix_A), axis=-1, keepdims=True)
#     matrix_A_norm = c * np.array(matrix_A / max_val)
#     vector_b_norm = c * np.array(vector_b / max_val)

#     # Check sparsity: count non-zero elements
#     nonzero_count = np.count_nonzero(matrix_A_norm)
#     total_elements = matrix_A_norm.shape[0] * matrix_A_norm.shape[1]
#     sparsity = 1 - nonzero_count / total_elements

#     # Use sparse solver if matrix is >90% sparse
#     if sparsity > 0.9 and method in ("auto", "lsqr"):
#         A_sparse = csr_matrix(matrix_A_norm)
#         b_flat = vector_b_norm.ravel()
#         result = lsqr(A_sparse, b_flat, damp=damping, atol=1e-12, btol=1e-12)
#         x = result[0]
#         return x.reshape(-1, 1)

#     # Dense path: optional Tikhonov via augmentation
#     if damping > 0.0:
#         n = matrix_A_norm.shape[1]
#         aug_A = np.vstack([matrix_A_norm, np.sqrt(damping) * np.eye(n)])
#         aug_b = np.vstack([vector_b_norm, np.zeros((n, 1))])
#         x, *_ = lstsq(aug_A, aug_b)
#         return x

#     x, *_ = lstsq(matrix_A_norm, vector_b_norm)
#     return x
