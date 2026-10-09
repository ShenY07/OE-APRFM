"""Batchable parity reference solvers adapted from the two parity SI notebooks."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.sparse.linalg import spsolve
from scipy.special import roots_legendre


def _field_1d(function, values, shape):
    return np.broadcast_to(np.asarray(function(values), dtype=float), shape)


def solve_parity_si_dsa_1d(
    config,
    *,
    grid=(512, 256),
    tol=1.0e-11,
    max_iter=16384,
    dsa_relaxation=1.0e-2,
    report_every=100,
):
    """Diamond-difference parity SI with the notebook's DSA correction."""
    started = perf_counter()
    nx, full_nv = map(int, grid)
    npos = full_nv // 2
    xmin, xmax = map(float, config.mesh.domain.x)
    dx = (xmax - xmin) / nx
    x_faces = np.linspace(xmin, xmax, nx + 1)
    x = 0.5 * (x_faces[:-1] + x_faces[1:])
    # Standard S_N angular discretization on [-1, 1].
    full_v, full_w = roots_legendre(full_nv)
    v, w = full_v[npos:], full_w[npos:]
    eps = float(config.model.knudsen_number)
    sigma_s = _field_1d(config.model.coeff.scattering, x, (nx,))
    sigma_a = _field_1d(config.model.coeff.absorption, x, (nx,))
    sigma_t = sigma_s + eps**2 * sigma_a
    diffusion = 1.0 / (3.0 * sigma_s)
    sigma_a_face = _field_1d(config.model.coeff.absorption, x_faces, (nx + 1,))
    xx, vv = np.meshgrid(x, v, indexing="ij")
    source_p = np.broadcast_to(
        np.asarray(config.model.source(xx, vv), float), (nx, npos)
    )
    source_m = np.broadcast_to(
        np.asarray(config.model.source(xx, -vv), float), (nx, npos)
    )
    f_plus = np.ones((nx + 1, npos))
    f_minus = np.ones((nx + 1, npos))
    f_plus[0] = np.broadcast_to(
        np.asarray(config.model.bdy_cond.f_l(v), float), (npos,)
    )
    f_minus[-1] = np.broadcast_to(
        np.asarray(config.model.bdy_cond.f_r(-v), float), (npos,)
    )

    def density_faces(fp, fm):
        return 0.5 * (fp + fm) @ w

    def collision(fp, fm):
        even_cell = 0.25 * (fp[1:] + fp[:-1] + fm[1:] + fm[:-1])
        density = even_cell @ w
        common = sigma_s[:, None] * density[:, None]
        return common + eps**2 * source_p, common + eps**2 * source_m

    def dsa(before, after):
        correction = np.zeros(nx + 1)
        bands = np.zeros((3, nx - 1))
        bands[1] = diffusion[:-1] + diffusion[1:] + sigma_a_face[1:-1] * dx**2
        if nx > 2:
            bands[0, 1:] = -diffusion[1:-1]
            bands[2, :-1] = -diffusion[1:-1]
        rhs = (after[1:-1] - before[1:-1]) * (dx / eps) ** 2
        correction[1:-1] = solve_banded((1, 1), bands, rhs)
        return correction

    converged = False
    for iteration in range(1, max_iter + 1):
        old_p, old_m = f_plus.copy(), f_minus.copy()
        before = density_faces(old_p, old_m)
        _, q_minus = collision(f_plus, f_minus)
        for index in range(nx - 1, -1, -1):
            denom = 0.5 * sigma_t[index] + eps * v / dx
            ratio = (0.5 * sigma_t[index] - eps * v / dx) / denom
            f_minus[index] = (
                q_minus[index] / denom - ratio * f_minus[index + 1]
            )
        q_plus, _ = collision(f_plus, f_minus)
        for index in range(nx):
            denom = 0.5 * sigma_t[index] + eps * v / dx
            ratio = (0.5 * sigma_t[index] - eps * v / dx) / denom
            f_plus[index + 1] = q_plus[index] / denom - ratio * f_plus[index]
        correction = dsa_relaxation * dsa(
            before, density_faces(f_plus, f_minus)
        )
        f_plus += correction[:, None]
        f_minus += correction[:, None]
        difference = np.hypot(
            np.linalg.norm(f_plus - old_p), np.linalg.norm(f_minus - old_m)
        )
        if iteration == 1 or iteration % report_every == 0:
            print(
                f"SI+DSA iteration {iteration}: difference={difference:.6e}",
                flush=True,
            )
        if difference < tol:
            converged = True
            break
    positive = 0.5 * (f_plus[1:] + f_plus[:-1])
    negative = 0.5 * (f_minus[1:] + f_minus[:-1])
    even = 0.5 * (positive + negative)
    velocity = np.concatenate((-v[::-1], v))
    weights = np.concatenate((w[::-1], w))
    distribution = np.concatenate((negative[:, ::-1], positive), axis=1)
    return {
        "x": x,
        "velocity": velocity,
        "weights": weights,
        "f": distribution,
        "r": even,
        "j": (positive - negative) / (2.0 * eps),
        "rho": even @ w,
        "iterations": iteration,
        "converged": converged,
        "final_difference": difference,
        "runtime_seconds": perf_counter() - started,
    }


def solve_parity_gmres_1d(
    config,
    *,
    grid=(256, 128),
    tol=1.0e-9,
    max_iter=200,
    restart=500,
    report_every=50,
):
    """Solve the same diamond-difference fixed point with restarted GMRES.

    This is the robust fallback for diffusive regimes where the notebook DSA
    correction is not consistent enough with the transport sweep to converge.
    Boundary face values are eliminated from the Krylov unknowns.
    """
    started = perf_counter()
    nx, full_nv = map(int, grid)
    npos = full_nv // 2
    xmin, xmax = map(float, config.mesh.domain.x)
    dx = (xmax - xmin) / nx
    x_faces = np.linspace(xmin, xmax, nx + 1)
    x = 0.5 * (x_faces[:-1] + x_faces[1:])
    # Standard S_N angular discretization on [-1, 1].
    full_v, full_w = roots_legendre(full_nv)
    v, w = full_v[npos:], full_w[npos:]
    eps = float(config.model.knudsen_number)
    sigma_s = _field_1d(config.model.coeff.scattering, x, (nx,))
    sigma_a = _field_1d(config.model.coeff.absorption, x, (nx,))
    sigma_t = sigma_s + eps**2 * sigma_a
    xx, vv = np.meshgrid(x, v, indexing="ij")
    source_p = np.broadcast_to(
        np.asarray(config.model.source(xx, vv), float), (nx, npos)
    )
    source_m = np.broadcast_to(
        np.asarray(config.model.source(xx, -vv), float), (nx, npos)
    )
    left = np.broadcast_to(
        np.asarray(config.model.bdy_cond.f_l(v), float), (npos,)
    )
    right = np.broadcast_to(
        np.asarray(config.model.bdy_cond.f_r(-v), float), (npos,)
    )

    def unpack(vector):
        fp = np.empty((nx + 1, npos))
        fm = np.empty_like(fp)
        fp[0] = left
        fp[1:] = vector[: nx * npos].reshape(nx, npos)
        fm[-1] = right
        fm[:-1] = vector[nx * npos :].reshape(nx, npos)
        return fp, fm

    def pack(fp, fm):
        return np.concatenate((fp[1:].ravel(), fm[:-1].ravel()))

    def sweep_vector(vector):
        fp, fm = unpack(vector)
        even = 0.25 * (fp[1:] + fp[:-1] + fm[1:] + fm[:-1])
        density = even @ w
        common = sigma_s[:, None] * density[:, None]
        q_p, q_m = common + eps**2 * source_p, common + eps**2 * source_m
        new_m = fm.copy()
        for index in range(nx - 1, -1, -1):
            denom = 0.5 * sigma_t[index] + eps * v / dx
            ratio = (0.5 * sigma_t[index] - eps * v / dx) / denom
            new_m[index] = q_m[index] / denom - ratio * new_m[index + 1]
        # Recompute the positive collision source with the updated negative flux.
        even = 0.25 * (fp[1:] + fp[:-1] + new_m[1:] + new_m[:-1])
        density = even @ w
        q_p = sigma_s[:, None] * density[:, None] + eps**2 * source_p
        new_p = fp.copy()
        for index in range(nx):
            denom = 0.5 * sigma_t[index] + eps * v / dx
            ratio = (0.5 * sigma_t[index] - eps * v / dx) / denom
            new_p[index + 1] = q_p[index] / denom - ratio * new_p[index]
        return pack(new_p, new_m)

    zero = np.zeros(2 * nx * npos)
    affine = sweep_vector(zero)
    operator = LinearOperator(
        (zero.size, zero.size),
        matvec=lambda z: z - (sweep_vector(z) - affine),
        dtype=float,
    )
    history = []

    def callback(residual):
        history.append(float(residual))
        if len(history) == 1 or len(history) % report_every == 0:
            print(
                f"1D GMRES iteration {len(history)}: residual={history[-1]:.6e}",
                flush=True,
            )

    solution, info = gmres(
        operator,
        affine,
        x0=zero,
        rtol=tol,
        atol=0.0,
        restart=restart,
        maxiter=max_iter,
        callback=callback,
        callback_type="pr_norm",
    )
    fp, fm = unpack(solution)
    positive = 0.5 * (fp[1:] + fp[:-1])
    negative = 0.5 * (fm[1:] + fm[:-1])
    even = 0.5 * (positive + negative)
    velocity = np.concatenate((-v[::-1], v))
    weights = np.concatenate((w[::-1], w))
    distribution = np.concatenate((negative[:, ::-1], positive), axis=1)
    residual = np.linalg.norm(operator @ solution - affine) / max(
        np.linalg.norm(affine), 1e-300
    )
    return {
        "x": x,
        "velocity": velocity,
        "weights": weights,
        "f": distribution,
        "r": even,
        "j": (positive - negative) / (2.0 * eps),
        "rho": even @ w,
        "iterations": len(history),
        "converged": info == 0,
        "relative_residual": float(residual),
        "runtime_seconds": perf_counter() - started,
    }


def solve_parity_gmres_2d(
    config,
    *,
    grid=(32, 32, 8),
    tol=1.0e-9,
    max_iter=100,
    restart=500,
    report_every=50,
):
    """Four-quadrant matrix-free parity solve from parity_si_2d.ipynb."""
    started = perf_counter()
    nx, ny, na = map(int, grid)
    eps = float(config.model.knudsen_number)
    xmin, xmax = map(float, config.mesh.domain.x)
    ymin, ymax = map(float, config.mesh.domain.y)
    dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
    x = xmin + (np.arange(nx) + 0.5) * dx
    y = ymin + (np.arange(ny) + 0.5) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    z, w = roots_legendre(na)
    theta = 0.25 * np.pi * (z + 1.0)
    w = 0.5 * w
    angles = np.concatenate(
        (theta, np.pi - theta, np.pi + theta, 2.0 * np.pi - theta)
    )
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    nang = angles.size
    sigma_s = np.broadcast_to(
        np.asarray(config.model.coeff.scattering(xx, yy), float), (nx, ny)
    )
    sigma_a = np.broadcast_to(
        np.asarray(config.model.coeff.absorption(xx, yy), float), (nx, ny)
    )
    sigma_t = sigma_s + eps**2 * sigma_a
    source = np.empty((nx, ny, nang))
    for index, angle in enumerate(angles):
        source[:, :, index] = np.broadcast_to(
            np.asarray(config.model.source(xx, yy, angle), float), (nx, ny)
        )
    boundary = config.model.bdy_cond
    vx, vy = directions[:, 0], directions[:, 1]
    stabilization = eps * (np.abs(vx) / dx + np.abs(vy) / dy)

    def density(field):
        return 0.25 * np.tensordot(
            field.reshape(nx, ny, 4, na).sum(axis=2), w, axes=([-1], [0])
        )

    def derivative_x(field):
        result = np.empty_like(field)
        result[1:-1] = (field[2:] - field[:-2]) / (2.0 * dx)
        for angle_index, speed in enumerate(vx):
            if speed > 0:
                left = (
                    2.0 * np.asarray(boundary.f_l(y), float)
                    - field[0, :, angle_index]
                )
                right = (
                    3.0 * field[-1, :, angle_index]
                    - 3.0 * field[-2, :, angle_index]
                    + field[-3, :, angle_index]
                )
            else:
                left = (
                    3.0 * field[0, :, angle_index]
                    - 3.0 * field[1, :, angle_index]
                    + field[2, :, angle_index]
                )
                right = (
                    2.0 * np.asarray(boundary.f_r(y), float)
                    - field[-1, :, angle_index]
                )
            result[0, :, angle_index] = (field[1, :, angle_index] - left) / (
                2.0 * dx
            )
            result[-1, :, angle_index] = (
                right - field[-2, :, angle_index]
            ) / (2.0 * dx)
        return result

    def derivative_y(field):
        result = np.empty_like(field)
        result[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2.0 * dy)
        for angle_index, speed in enumerate(vy):
            if speed > 0:
                lower = (
                    2.0 * np.asarray(boundary.f_b(x), float)
                    - field[:, 0, angle_index]
                )
                upper = (
                    3.0 * field[:, -1, angle_index]
                    - 3.0 * field[:, -2, angle_index]
                    + field[:, -3, angle_index]
                )
            else:
                lower = (
                    3.0 * field[:, 0, angle_index]
                    - 3.0 * field[:, 1, angle_index]
                    + field[:, 2, angle_index]
                )
                upper = (
                    2.0 * np.asarray(boundary.f_t(x), float)
                    - field[:, -1, angle_index]
                )
            result[:, 0, angle_index] = (field[:, 1, angle_index] - lower) / (
                2.0 * dy
            )
            result[:, -1, angle_index] = (
                upper - field[:, -2, angle_index]
            ) / (2.0 * dy)
        return result

    def sweep(field):
        transport = eps * (
            vx[None, None, :] * derivative_x(field)
            + vy[None, None, :] * derivative_y(field)
        )
        rhs = (
            sigma_s[:, :, None] * density(field)[:, :, None] + eps**2 * source
        )
        return (rhs + stabilization[None, None, :] * field - transport) / (
            sigma_t[:, :, None] + stabilization[None, None, :]
        )

    zero = np.zeros((nx, ny, nang))
    affine = sweep(zero)

    def matvec(vector):
        field = vector.reshape(nx, ny, nang)
        return (field - (sweep(field) - affine)).ravel()

    operator = LinearOperator(
        (zero.size, zero.size), matvec=matvec, dtype=float
    )
    history = []

    def callback(residual):
        history.append(float(residual))
        if len(history) == 1 or len(history) % report_every == 0:
            print(
                f"GMRES iteration {len(history)}: residual={history[-1]:.6e}",
                flush=True,
            )

    initial = np.ones_like(zero)
    if eps < 1.0e-2:
        # Diffusion-limit initial guess.  It removes the slowly damped isotropic
        # mode responsible for Krylov stagnation in strongly diffusive P5 runs.
        q_average = source.mean(axis=2)
        diffusion = 1.0 / (2.0 * sigma_s)
        nxi, nyi = nx - 2, ny - 2
        matrix = lil_matrix((nxi * nyi, nxi * nyi), dtype=float)
        rhs = q_average[1:-1, 1:-1].reshape(-1)

        def idx(i, j):
            return i * nyi + j

        for i in range(nxi):
            for j in range(nyi):
                ii, jj, row = i + 1, j + 1, idx(i, j)
                de = (
                    2
                    * diffusion[ii, jj]
                    * diffusion[ii + 1, jj]
                    / (diffusion[ii, jj] + diffusion[ii + 1, jj])
                )
                dw = (
                    2
                    * diffusion[ii, jj]
                    * diffusion[ii - 1, jj]
                    / (diffusion[ii, jj] + diffusion[ii - 1, jj])
                )
                dn = (
                    2
                    * diffusion[ii, jj]
                    * diffusion[ii, jj + 1]
                    / (diffusion[ii, jj] + diffusion[ii, jj + 1])
                )
                ds = (
                    2
                    * diffusion[ii, jj]
                    * diffusion[ii, jj - 1]
                    / (diffusion[ii, jj] + diffusion[ii, jj - 1])
                )
                matrix[row, row] = (
                    (de + dw) / dx**2 + (dn + ds) / dy**2 + sigma_a[ii, jj]
                )
                if i + 1 < nxi:
                    matrix[row, idx(i + 1, j)] = -de / dx**2
                if i > 0:
                    matrix[row, idx(i - 1, j)] = -dw / dx**2
                if j + 1 < nyi:
                    matrix[row, idx(i, j + 1)] = -dn / dy**2
                if j > 0:
                    matrix[row, idx(i, j - 1)] = -ds / dy**2
        rho0 = np.zeros((nx, ny))
        rho0[1:-1, 1:-1] = spsolve(matrix.tocsr(), rhs).reshape(nxi, nyi)
        initial = np.broadcast_to(rho0[:, :, None], zero.shape).copy()
    effective_restart = min(restart, max(50, 16_000_000 // zero.size))
    solution, info = gmres(
        operator,
        affine.ravel(),
        x0=initial.ravel(),
        rtol=tol,
        atol=0.0,
        restart=effective_restart,
        maxiter=max_iter,
        callback=callback,
        callback_type="pr_norm",
    )
    distribution = solution.reshape(nx, ny, nang)
    quadrants = distribution.reshape(nx, ny, 4, na)
    r_first = 0.5 * (quadrants[:, :, 0] + quadrants[:, :, 2])
    j_first = (quadrants[:, :, 0] - quadrants[:, :, 2]) / (2.0 * eps)
    r_second = 0.5 * (quadrants[:, :, 3] + quadrants[:, :, 1])
    j_second = (quadrants[:, :, 3] - quadrants[:, :, 1]) / (2.0 * eps)
    residual = np.linalg.norm(matvec(solution) - affine.ravel())
    rhs_norm = np.linalg.norm(affine)
    return {
        "x": x,
        "y": y,
        "theta": angles,
        "weights_quadrant": w,
        "f": distribution,
        "r1": r_first,
        "j1": j_first,
        "r2": r_second,
        "j2": j_second,
        "rho": density(distribution),
        "iterations": len(history),
        "converged": info == 0,
        "relative_residual": residual / rhs_norm if rhs_norm else residual,
        "runtime_seconds": perf_counter() - started,
    }
