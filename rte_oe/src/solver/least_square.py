import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import lsqr
from scipy.linalg import lstsq
from scipy.linalg import svd


def solve(matrix_A: np.ndarray, vector_b: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Solve the least square problem with row-wise normalization."""
    max_val = np.max(np.abs(matrix_A), axis=-1, keepdims=True)

    # Avoid division by zero: only normalize non-zero rows
    # For zero rows, keep them as is (they contribute nothing to the solution)
    epsilon = 1e-30
    max_val = np.where(max_val > epsilon, max_val, 1.0)

    matrix_A = c * np.array(matrix_A / max_val)
    vector_b = c * np.array(vector_b / max_val)
    x, *_ = lstsq(matrix_A, vector_b)
    return x


# def solve(
#     matrix_A: np.ndarray,
#     vector_b: np.ndarray,
#     c: float = 1.0,
#     damping: float = 0.0,
#     method: str = "auto",
# ) -> np.ndarray:
#     """
#     Solve min ||A x - b||^2 + damping * ||x||^2 with simple row scaling.

#     - If sparsity > 0.9 and method is auto or lsqr, use scipy.sparse.linalg.lsqr
#         with LSQR's built-in damping ("damp" parameter).
#     - Otherwise use dense lstsq; if damping > 0, solve the augmented system
#         [A; sqrt(damping) I] x ≈ [b; 0].
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
