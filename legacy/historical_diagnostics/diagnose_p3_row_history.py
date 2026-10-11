"""Test boundary assembly hypotheses against retained P3 fields, without edits."""

import hashlib
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import lstsq
import run_p3_oe_aprfm as runner

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/review_audit_20261010/p3_row_history"
OLD = (
    ROOT
    / ".release-backup/20261009/original-worktree/results/p34_representation_comparison"
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    captured = []
    original_add = runner.StreamingQRLeastSquares.add

    def capture(self, A, b, **kwargs):
        captured.append(
            (np.asarray(A).copy(), np.asarray(b).reshape(-1).copy())
        )
        return original_add(self, A, b, **kwargs)

    runner.StreamingQRLeastSquares.add = capture
    runner.run(
        1.0,
        11,
        OUT / "current",
        "p3",
        partitions=(1, 1, 1),
        features=128,
        collocation=(16, 16, 16),
    )
    runner.StreamingQRLeastSquares.add = original_add
    config = runner.get_config(1.0)
    sample = runner.Sample2D(
        dict(config.mesh.domain),
        {"interior": (16, 16, 16), "boundary": (16, 16, 16)},
        config.model.sample_mode,
    )
    common = dict(
        domain=dict(config.mesh.domain),
        strides={"x": 2.0, "y": 2.0, "theta": np.pi / 2},
        Jn={"j": 128, "r": 128},
        scale=1.0,
        init_rng=jax.random.key(11),
        kn=1.0,
        activation=jnp.tanh,
    )
    bdy = runner.V2BoundaryConstraint2D(**common)
    pair = jax.jit(jax.vmap(lambda x, y, t: bdy.apply({}, x, y, t)))
    trace = jax.jit(
        jax.vmap(lambda x, y, t: bdy.apply({}, x, y, t, method=bdy.trace))
    )
    coords = [
        sample.pts_left,
        sample.pts_right,
        sample.pts_lower,
        sample.pts_upper,
    ]
    data = np.concatenate([b for A, b in captured[:8]])
    matrices = {}
    for mode in ("physical", "folded"):
        pieces = []
        for x, y, t in coords:
            if mode == "folded":
                t = jnp.arctan2(jnp.abs(jnp.sin(t)), jnp.abs(jnp.cos(t)))
            pieces.append(np.asarray(pair(x, y, t)))
        matrices[mode] = np.concatenate(pieces)
    with np.load(
        OLD / "p3_oe_aprfm_eps_1e+00_seed_11_four_component.npz"
    ) as d:
        rng = np.random.default_rng(823)
        ix = rng.integers(0, len(d["x"]), 3000)
        iy = rng.integers(0, len(d["y"]), 3000)
        it = rng.integers(0, len(d["theta"]), 3000)
        # Avoid quadrant seams in the first diagnostic; inspect those separately.
        it = np.where(it % 16 == 0, (it + 1) % 64, it)
        x, y, t = d["x"][ix], d["y"][iy], d["theta"][it]
        target = d["f"][ix, iy, it]
    E = np.asarray(
        trace(
            jnp.asarray(x[:, None]),
            jnp.asarray(y[:, None]),
            jnp.asarray(t[:, None]),
        )
    )
    fit, _, rank, _ = lstsq(E, target, cond=1e-12)
    output = [
        dict(
            mode="fit_archived_field_to_current_basis",
            rank=int(rank),
            relative_field_difference=float(
                np.linalg.norm(E @ fit - target) / np.linalg.norm(target)
            ),
        )
    ]
    print(output[-1], flush=True)
    np.savez_compressed(
        OUT / "archived_field_fit.npz",
        coefficients=fit,
        E=E,
        target=target,
        x=x,
        y=y,
        theta=t,
    )
    scenarios = [
        ("current", np.concatenate([A for A, b in captured[:8]]), data)
    ]
    for mode, A in matrices.items():
        mixed = np.stack((A[:, 0] + A[:, 1], A[:, 0] - A[:, 1]), axis=1)
        scenarios += [
            (mode + "_pair_repeat", A.reshape(-1, 512), np.repeat(data, 2)),
            (
                mode + "_mixed_correct",
                mixed.reshape(-1, 512),
                np.column_stack((2 * data, np.zeros_like(data))).ravel(),
            ),
            (
                mode + "_mixed_repeat",
                mixed.reshape(-1, 512),
                np.repeat(data, 2),
            ),
        ]
    for name, A, b in scenarios:
        for norm in ("l2", "linf"):
            solver = runner.StreamingQRLeastSquares(512, rcond=1e-12)

            def add(A, b):
                factors = (
                    1.0
                    if norm == "l2"
                    else np.linalg.norm(A, axis=1)
                    / np.maximum(np.max(np.abs(A), axis=1), 1e-30)
                )
                solver.add(A, b, row_factors=factors)

            for start in range(0, len(b), 512):
                add(A[start : start + 512], b[start : start + 512])
            for Ai, bi in captured[8:]:
                add(Ai, bi)
            c, diag = solver.solve()
            c = c.ravel()
            delta = float(
                np.linalg.norm(E @ c - target) / np.linalg.norm(target)
            )
            row = dict(
                mode=name,
                norm=norm,
                Nrow=solver.num_rows,
                relative_field_difference=delta,
                coefficient_fit_relative_difference=float(
                    np.linalg.norm(c - fit) / np.linalg.norm(fit)
                ),
                rank=int(diag["rank"]),
            )
            output.append(row)
            np.save(OUT / f"coefficients_{name}_{norm}.npy", c)
            (OUT / "hypotheses.json").write_text(json.dumps(output, indent=2))
            print(row, flush=True)


if __name__ == "__main__":
    import os

    if os.environ.get("OE_RUN_HISTORICAL_DIAGNOSTICS") != "1":
        raise SystemExit(
            "Archived diagnostic only. See archive/README.md; not a production experiment."
        )
    main()
