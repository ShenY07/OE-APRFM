"""Memory-bounded dense least squares via incremental QR compression."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.linalg import lstsq

Array = np.ndarray


def iter_evaluated_blocks(
    num_columns: int,
    matrix_fn: Callable[..., Array],
    rhs_fn: Callable[..., Array],
    points: Sequence[Array],
    *,
    eval_batch_size: int = 128,
    qr_point_batch_size: int = 1024,
):
    """Yield synchronized NumPy blocks without retaining all feature rows."""

    if not points:
        raise ValueError("points must not be empty")
    num_points = int(points[0].shape[0])
    if any(point.shape[0] != num_points for point in points):
        raise ValueError(
            "all point arrays must have the same leading dimension"
        )
    if eval_batch_size <= 0 or qr_point_batch_size <= 0:
        raise ValueError("batch sizes must be positive")

    for start in range(0, num_points, qr_point_batch_size):
        stop = min(start + qr_point_batch_size, num_points)
        matrix_parts: list[Array] = []
        rhs_parts: list[Array] = []
        for eval_start in range(start, stop, eval_batch_size):
            eval_stop = min(eval_start + eval_batch_size, stop)
            point_block = tuple(
                point[eval_start:eval_stop] for point in points
            )
            matrix_parts.append(
                np.asarray(matrix_fn(*point_block)).reshape(-1, num_columns)
            )
            rhs_parts.append(np.asarray(rhs_fn(*point_block)).reshape(-1))
        yield (
            np.concatenate(matrix_parts, axis=0),
            np.concatenate(rhs_parts, axis=0),
            start,
            stop,
        )


class StreamingQRLeastSquares:
    """Accumulate every equation while retaining only an augmented QR factor.

    Rows are normalized exactly as in :mod:`solver.least_squares`.  Adding a
    block performs an orthogonal compression of ``[A | b]``; it does not drop,
    sample, or independently solve any equations.
    """

    def __init__(self, num_columns: int, *, rcond: float | None = None):
        if num_columns <= 0:
            raise ValueError("num_columns must be positive")
        self.num_columns = int(num_columns)
        self.rcond = rcond
        self._augmented_r: Array | None = None
        self.num_rows = 0
        self.max_block_bytes = 0
        self._rhs_norm_squared = 0.0

    @property
    def augmented_r(self) -> Array:
        if self._augmented_r is None:
            raise RuntimeError("no equation blocks have been added")
        return self._augmented_r

    @property
    def dense_matrix_bytes(self) -> int:
        return self.num_rows * self.num_columns * np.dtype(np.float64).itemsize

    def add(
        self,
        matrix_block: Array,
        rhs_block: Array,
        *,
        row_factors: float | Array = 1.0,
    ) -> None:
        """Normalize and add one block of global least-squares rows.

        ``row_factors`` are applied after row normalization.  They can be used
        for block weighting by passing the square root of the desired weight.
        """

        A = np.array(matrix_block, dtype=np.float64, copy=True)
        b = np.array(rhs_block, dtype=np.float64, copy=True).reshape(-1)
        if A.ndim != 2 or A.shape[1] != self.num_columns:
            raise ValueError(
                f"matrix block must have shape (rows, {self.num_columns}); got {A.shape}"
            )
        if b.size == 1 and A.shape[0] != 1:
            b = np.full(A.shape[0], b.item(), dtype=np.float64)
        if A.shape[0] != b.size:
            raise ValueError(
                f"incompatible block shapes: {A.shape}, {b.shape}"
            )
        if not np.all(np.isfinite(A)):
            raise ValueError(
                f"matrix block contains non-finite values before row {self.num_rows}"
            )
        if not np.all(np.isfinite(b)):
            raise ValueError(
                f"RHS block contains non-finite values before row {self.num_rows}"
            )

        # The frozen experiment protocol uses unit-L2 row scaling.  Infinity
        # norm scaling changes the relative least-squares metric according to
        # the random shape of each feature row and is especially harmful when
        # different parity equations occupy separate blocks.
        row_scale = np.linalg.norm(A, axis=1)
        row_scale[row_scale <= 1e-30] = 1.0
        A /= row_scale[:, None]
        b /= row_scale

        factors = np.asarray(row_factors, dtype=np.float64)
        if factors.ndim == 0:
            if factors <= 0:
                raise ValueError("row_factors must be positive")
            A *= factors
            b *= factors
        else:
            factors = factors.reshape(-1)
            if factors.size != A.shape[0] or np.any(factors <= 0):
                raise ValueError(
                    "row_factors must contain one positive value per row"
                )
            A *= factors[:, None]
            b *= factors

        augmented = np.empty(
            (A.shape[0], self.num_columns + 1), dtype=np.float64
        )
        augmented[:, :-1] = A
        augmented[:, -1] = b
        self.max_block_bytes = max(self.max_block_bytes, A.nbytes + b.nbytes)
        self.num_rows += A.shape[0]
        self._rhs_norm_squared += float(b @ b)

        if self._augmented_r is not None:
            augmented = np.vstack((self._augmented_r, augmented))
        self._augmented_r = np.linalg.qr(augmented, mode="r")
        if not np.all(np.isfinite(self._augmented_r)):
            raise FloatingPointError(
                f"QR factor became non-finite after accumulating {self.num_rows} rows"
            )

    def add_function(
        self,
        matrix_fn: Callable[..., Array],
        rhs_fn: Callable[..., Array],
        points: Sequence[Array],
        *,
        eval_batch_size: int = 128,
        qr_point_batch_size: int = 1024,
        row_factor_fn: Callable[[int, int], float | Array] | None = None,
    ) -> None:
        """Evaluate pointwise matrix/RHS functions in small, synchronized batches."""

        for A, b, start, stop in iter_evaluated_blocks(
            self.num_columns,
            matrix_fn,
            rhs_fn,
            points,
            eval_batch_size=eval_batch_size,
            qr_point_batch_size=qr_point_batch_size,
        ):
            row_factors = (
                1.0 if row_factor_fn is None else row_factor_fn(start, stop)
            )
            self.add(A, b, row_factors=row_factors)

    def solve(
        self, *, damping: float = 0.0
    ) -> tuple[Array, dict[str, float | int]]:
        """Solve the compressed global system with column equilibration."""

        if damping < 0:
            raise ValueError("damping must be nonnegative")
        R_aug = self.augmented_r
        if R_aug.shape[0] < self.num_columns:
            raise ValueError(
                "the accumulated system has fewer rows than columns"
            )

        R_A = R_aug[: self.num_columns, : self.num_columns]
        transformed_b = R_aug[: self.num_columns, -1:]
        column_scale = np.linalg.norm(R_A, axis=0)
        column_scale[column_scale <= 1e-14] = 1.0
        R_scaled = R_A / column_scale[None, :]

        if damping:
            solve_A = np.vstack(
                (R_scaled, np.sqrt(damping) * np.eye(self.num_columns))
            )
            solve_b = np.vstack(
                (transformed_b, np.zeros((self.num_columns, 1)))
            )
        else:
            solve_A, solve_b = R_scaled, transformed_b

        scaled_x, _, rank, singular_values = lstsq(
            solve_A,
            solve_b,
            cond=self.rcond,
            lapack_driver="gelsd",
        )
        x = scaled_x / column_scale[:, None]
        threshold = (
            self.rcond
            if self.rcond is not None
            else np.finfo(float).eps * max(solve_A.shape)
        ) * singular_values[0]
        effective = singular_values[singular_values > threshold]
        smallest_effective = (
            float(effective[-1]) if effective.size else float("nan")
        )
        condition_number = float(singular_values[0] / smallest_effective)

        # Evaluate the data residual of the returned (possibly truncated or
        # damped) solution, not only the orthogonal QR tail. The latter omits
        # unresolved singular directions. Regularization is not part of this
        # data-residual diagnostic.
        residual_l2 = float(
            np.linalg.norm(R_aug[:, : self.num_columns] @ x - R_aug[:, -1:])
        )
        relative_residual = float("nan")
        if self._rhs_norm_squared > 0 and np.isfinite(residual_l2):
            relative_residual = residual_l2 / np.sqrt(self._rhs_norm_squared)
        diagnostics: dict[str, float | int] = {
            "rank": int(rank),
            "condition_number": condition_number,
            "largest_singular_value": float(singular_values[0]),
            "smallest_effective_singular_value": smallest_effective,
            "singular_value_threshold": float(threshold),
            "normalized_residual_l2": residual_l2,
            "normalized_residual_rms": residual_l2 / np.sqrt(self.num_rows),
            "relative_normalized_residual": relative_residual,
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "dense_matrix_bytes": self.dense_matrix_bytes,
            "max_block_bytes": self.max_block_bytes,
            "augmented_r_bytes": R_aug.nbytes,
        }
        return x, diagnostics
