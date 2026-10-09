"""Frozen-budget source-free slab; symmetric RF initialization, direct j current."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import json
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import jax
import jax.numpy as jnp
from diagnose_oe_minimum_gain import rf_evaluator, gauss, ROOT
import modules.function_space as fs
from solver.least_squares import solve


def symmetric(scale):
    def init(key, shape, dtype=jnp.float64):
        return jax.random.uniform(
            key, shape, dtype, minval=-scale, maxval=scale
        )

    return init


def relative(a, b, w):
    return float(np.sqrt(np.sum(w * (a - b) ** 2) / np.sum(w * b**2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quadrature", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[1.0, 0.1, 0.01, 0.001, 0.0001, 1e-6],
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/e3_slab"
    )
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    fs.uniform = symmetric
    for seed in args.seeds:
        start = perf_counter()
        ev = rf_evaluator(64, seed)
        x = np.linspace(0, 1, 32)[1:-1]
        v = (np.arange(32) + 0.5) / 32
        xx, vv = np.meshgrid(x, v, indexing="ij")
        r, j, rx, jx = ev(xx, vv)
        nodes, weights = gauss(args.quadrature)
        xq, vq = np.meshgrid(x, nodes, indexing="ij")
        rq, _, _, jxq = ev(xq, vq)
        rho = np.einsum("xvk,v->xk", rq, weights)
        mean = np.einsum("xvk,v->xk", jxq, weights * nodes)
        vb = np.linspace(0, 1, 33)[1:]
        rl, jl, _, _ = ev(np.zeros_like(vb), vb)
        rr, jr, _, _ = ev(np.ones_like(vb), vb)
        feature_time = perf_counter() - start
        # Dense independent physical quadrature, separately checked below.
        evaluation = {}
        for n in (64, 128):
            xe, wx = gauss(n)
            ve, wv = gauss(n)
            xe2, ve2 = np.meshgrid(xe, ve, indexing="ij")
            re, je, rxe, _ = ev(xe2, ve2)
            zi, wi = gauss(n)
            zi = 0.1 + 0.8 * zi
            wi = 0.8 * wi
            xi, vi = np.meshgrid(zi, ve, indexing="ij")
            _, ji, rxi, _ = ev(xi, vi)
            evaluation[n] = (xe, wx, wv, re, je, rxe, wi, ji, rxi)
        for eps in args.epsilons:
            path = out / f"seed{seed}_eps{eps:.0e}.json"
            if path.exists():
                continue
            start = perf_counter()
            a1 = np.broadcast_to(mean[:, None, :], r.shape)
            a2 = (
                eps**2 * (vv[..., None] * jx - mean[:, None, :])
                + r
                - rho[:, None, :]
            )
            a3 = vv[..., None] * rx + j
            a = np.concatenate(
                [
                    rl + eps * jl,
                    rr - eps * jr,
                    np.stack([a1, a2, a3], axis=2).reshape(-1, 128),
                ]
            )
            b = np.concatenate([np.ones(32), np.zeros(a.shape[0] - 32)])
            assert a.shape == (2944, 128)
            assembly_time = perf_counter() - start
            start = perf_counter()
            c, d = solve(a, b, rcond=1e-12, return_diagnostics=True)
            c = c.ravel()
            solve_time = perf_counter() - start
            metrics = {}
            for n, (
                xe,
                wx,
                wv,
                re,
                je,
                rxe,
                wi,
                ji,
                rxi,
            ) in evaluation.items():
                density = (re @ c) @ wv
                current = (je @ c) @ (gauss(n)[0] * wv)
                defect = (ji @ c) @ (gauss(n)[0] * wv) + (rxi @ c) @ wv / 3
                metrics[n] = dict(
                    E_diff=relative(density, 1 - xe, wx),
                    E_q_diff=relative(current, np.full_like(xe, 1 / 3), wx),
                    fick_defect=float(
                        np.sqrt(np.sum(wi * defect**2) / np.sum(wi / 9))
                    ),
                )
            record = dict(
                seed=seed,
                epsilon=eps,
                initialization="U(-1,1)",
                num_columns=128,
                num_rows=2944,
                angular_collocation=32,
                angular_quadrature=args.quadrature,
                rcond=1e-12,
                rank=d["rank"],
                coefficient_norm=d["coefficient_norm"],
                feature_seconds=feature_time,
                assembly_seconds=assembly_time,
                solve_seconds=solve_time,
                total_seconds=feature_time + assembly_time + solve_time,
                evaluation=metrics,
                reference_status="finite-epsilon transport comparison pending",
            )
            path.write_text(json.dumps(record, indent=2) + "\n")
            np.savez_compressed(path.with_suffix(".npz"), coefficients=c)
            print(seed, eps, metrics[128], flush=True)


if __name__ == "__main__":
    main()
