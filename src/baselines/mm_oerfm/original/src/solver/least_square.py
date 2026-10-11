import numpy as np

# from numpy.linalg import cond
# from scipy.sparse.linalg import lsqr
from scipy.linalg import lstsq


def solve(
    matrix_A: np.ndarray, vector_b: np.ndarray, c: float = 1.0
) -> np.ndarray:
    """Solve the least square problem."""
    max_val = np.max(np.abs(matrix_A), axis=-1, keepdims=True)
    matrix_A = c * np.array(matrix_A / max_val)
    vector_b = c * np.array(vector_b / max_val)
    x, *_ = lstsq(matrix_A, vector_b)
    return x
