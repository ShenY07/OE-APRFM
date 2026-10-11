"""Validate the recovered 15,856-row boundary signature on all archived P3 runs."""

import json
import hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import run_p3_oe_aprfm as runner

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/review_audit_20261010/p3_row_history/validation"
ARCHIVE = (
    ROOT
    / ".release-backup/20261009/original-worktree/results/p34_representation_comparison"
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original_boundary = runner.V2BoundaryConstraint2D
    original_add = runner.StreamingQRLeastSquares.add
    original_solve = runner.StreamingQRLeastSquares.solve
    state = {}

    class PairedBoundary(original_boundary):
        def trace(self, x, y, t):
            return self.__call__(x, y, t)

    def add(self, A, b, **kwargs):
        if A.ndim == 3:
            assert A.shape[1] == 2
            A = A.reshape(-1, A.shape[-1])
            b = np.repeat(np.asarray(b).reshape(-1), 2)
        return original_add(self, A, b, **kwargs)

    def solve(self, **kwargs):
        result = original_solve(self, **kwargs)
        state["c"] = result[0].ravel()
        return result

    runner.V2BoundaryConstraint2D = PairedBoundary
    runner.StreamingQRLeastSquares.add = add
    runner.StreamingQRLeastSquares.solve = solve
    records = []
    for eps in (1.0, 0.001):
        for seed in (11, 23, 37):
            print("START", eps, seed, flush=True)
            r = runner.run(
                eps,
                seed,
                OUT / "raw",
                "p3",
                partitions=(1, 1, 1),
                features=128,
                collocation=(16, 16, 16),
            )
            assert r["num_rows"] == 15856
            oldpath = (
                ARCHIVE
                / f"p3_oe_aprfm_eps_{eps:.0e}_seed_{seed}_four_component.json"
            )
            old = json.loads(oldpath.read_text())
            with np.load(oldpath.with_suffix(".npz")) as d:
                x, y, t, oldf, exact = (
                    d[k] for k in ["x", "y", "theta", "f", "reference_f"]
                )
            common = dict(
                domain={
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "theta": (0.0, np.pi / 2),
                },
                strides={"x": 2.0, "y": 2.0, "theta": np.pi / 2},
                Jn={"j": 128, "r": 128},
                scale=1.0,
                init_rng=jax.random.key(seed),
                kn=eps,
                activation=jnp.tanh,
            )
            boundary = original_boundary(**common)
            trace = jax.jit(
                jax.vmap(
                    lambda a, b, t: boundary.apply(
                        {}, a, b, t, method=boundary.trace
                    )
                )
            )
            newf = np.empty_like(oldf)
            for i, xx in enumerate(x):
                yy, tt = np.meshgrid(y, t, indexing="ij")
                rows = np.asarray(
                    trace(
                        jnp.full((yy.size, 1), xx),
                        jnp.asarray(yy.reshape(-1, 1)),
                        jnp.asarray(tt.reshape(-1, 1)),
                    )
                )
                newf[i] = (rows @ state["c"]).reshape(len(y), len(t))
            nonseam = (np.arange(len(t)) % 16) != 0
            Ef = float(np.linalg.norm(newf - exact) / np.linalg.norm(exact))
            rho = newf.mean(axis=2)
            exactrho = exact.mean(axis=2)
            Erho = float(
                np.linalg.norm(rho - exactrho) / np.linalg.norm(exactrho)
            )
            result = dict(
                epsilon=eps,
                seed=seed,
                Nrow=r["num_rows"],
                macro_rows=2352,
                even_rows=4704,
                odd_rows=4704,
                boundary_rows=4096,
                archived_Ef=old["relative_l2_f"],
                reconstructed_Ef_old_grid=Ef,
                reconstructed_Ef_midpoint_grid=r["relative_l2_f"],
                archived_Erho=old["relative_l2_rho"],
                reconstructed_Erho_old_grid=Erho,
                field_relative_difference=float(
                    np.linalg.norm(newf - oldf) / np.linalg.norm(oldf)
                ),
                field_nonseam_relative_difference=float(
                    np.linalg.norm((newf - oldf)[:, :, nonseam])
                    / np.linalg.norm(oldf[:, :, nonseam])
                ),
                field_max_abs_difference=float(np.max(np.abs(newf - oldf))),
                archive_source=str(oldpath.relative_to(ROOT)),
                archive_sha256=hashlib.sha256(
                    oldpath.read_bytes()
                ).hexdigest(),
                warning="Recovered numerical signature using current interior and paired boundary at physical angles; not a recovered original command/commit.",
            )
            records.append(result)
            np.savez_compressed(
                OUT / f"field_eps{eps:.0e}_seed{seed}.npz",
                x=x,
                y=y,
                theta=t,
                f=newf,
                coefficients=state["c"],
            )
            (OUT / "comparison.json").write_text(json.dumps(records, indent=2))
            print(result, flush=True)
            jax.clear_caches()


if __name__ == "__main__":
    import os

    if os.environ.get("OE_RUN_HISTORICAL_DIAGNOSTICS") != "1":
        raise SystemExit(
            "Archived diagnostic only. See archive/README.md; not a production experiment."
        )
    main()
