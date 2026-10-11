"""Reproduce archived parity budgets; audit folded evaluation of identical fits.

Run with CPU float64 and one BLAS thread, PYTHONPATH=src:scripts.
No production operator or archived output is modified.
"""

import hashlib
import argparse
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import legacy_run_p1_oe_aprfm as runner

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/review_audit_20261010/parity"
ARCHIVE = (
    ROOT
    / ".release-backup/20261009/original-worktree/results/requirement_2026_08_20/parity_budget_raw"
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        jax=jax.__version__,
        numpy=np.__version__,
        sources={
            str(p.relative_to(ROOT)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in [
                Path(__file__).resolve(),
                Path(__file__).resolve().with_name("legacy_run_p1_oe_aprfm.py"),
                Path(__file__).resolve().with_name("legacy_solution_construction.py"),
                *sorted((ROOT / "src").rglob("*.py")),
            ]
        },
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    original_solve = runner.solve
    captured = {}

    def capture(A, b, **kw):
        result = original_solve(A, b, **kw)
        captured.update(
            c=result[0].copy(),
            A=A,
            b=b,
            matrix_sha256=hashlib.sha256(A.tobytes()).hexdigest(),
        )
        return result

    runner.solve = capture
    records = []
    paths = [
        p
        for p in ARCHIVE.glob("*.json")
        if json.loads(p.read_text()).get("variant") in ("full", "full_angular")
    ]
    paths = sorted(
        paths,
        key=lambda p: (
            json.loads(p.read_text())["features_per_patch"],
            str(p),
        ),
    )
    for path in paths:
        old = json.loads(path.read_text())
        J, seed, variant = (
            old["features_per_patch"],
            old["seed"],
            old["variant"],
        )
        tag = f"J{J}_{variant}"
        print(f"START {tag} seed={seed}", flush=True)
        r = runner.run(
            old["epsilon"],
            seed,
            OUT / "raw",
            partitions=tuple(old["partitions"]),
            features=J,
            scale=old["scale"],
            rcond=old["rcond"],
            block_weights=old["block_weights"],
            collocation=tuple(old["collocation"]["interior"]),
            tag=tag,
            variant=variant,
            legacy_signed_velocity=True,
        )
        assert r["collocation"] == old["collocation"] or (
            list(r["collocation"]["interior"])
            == old["collocation"]["interior"]
            and r["collocation"]["boundary"] == old["collocation"]["boundary"]
        )
        assert r["num_rows"] == old["num_rows"]
        c = captured["c"].reshape(-1)
        np.save(
            OUT
            / "raw"
            / f"coefficients_eps{old['epsilon']:.0e}_J{J}_{variant}_s{seed}.npy",
            c,
        )
        stem = f"p1_oe_aprfm_eps_{old['epsilon']:.0e}_seed_{seed}_{tag}"
        with np.load(OUT / "raw" / f"{stem}.npz") as d:
            x, v, exact, numerical = d["x"], d["v"], d["exact"], d["f"]
        r.update(
            archive_source=str(path.relative_to(ROOT)),
            archive_Ef=old["relative_l2_f"],
            archive_Erho=old["relative_l2_rho"],
            matrix_sha256=captured["matrix_sha256"],
        )
        if variant == "full_angular":
            field = runner.function_space.RandomFeatureSpaceXV(
                domain={"x": (0.0, 1.0), "v": (0.0, 1.0)},
                strides=dict(r["strides"]),
                Jn=J,
                scale=old["scale"],
                activation=jnp.tanh,
            )
            params = field.init(
                jax.random.key(seed), jnp.zeros(1), jnp.ones(1)
            )
            split = c.size // 2

            def folded(xx, vv):
                features = field.apply(params, xx, jnp.abs(vv)).reshape(-1)
                return features @ c[split:] + old["epsilon"] * jnp.sign(
                    vv
                ).squeeze() * (features @ c[:split])

            xx, vv = np.meshgrid(x, v, indexing="ij")
            folded_f = np.asarray(
                jax.vmap(folded)(
                    jnp.asarray(xx.reshape(-1, 1)),
                    jnp.asarray(vv.reshape(-1, 1)),
                )
            ).reshape(exact.shape)
            rho = np.trapezoid(folded_f, v, axis=1) / 2
            erho = np.trapezoid(exact, v, axis=1) / 2
            r.update(
                folded_Ef=float(
                    np.linalg.norm(folded_f - exact) / np.linalg.norm(exact)
                ),
                folded_Erho=float(
                    np.linalg.norm(rho - erho) / np.linalg.norm(erho)
                ),
                positive_half_max_change=float(
                    np.max(np.abs(folded_f[:, v > 0] - numerical[:, v > 0]))
                ),
            )
            # Compare the fitted right-boundary rows with the physical trace.
            config = runner.get_config(old["epsilon"])
            sample = runner.Sample1D(
                dict(config.mesh.domain),
                r["collocation"],
                config.model.sample_mode,
            )
            bx, bv = sample.pts_right
            nb = len(bv)
            trained = captured["A"][nb : 2 * nb] @ c
            folded_trace = np.asarray(jax.vmap(folded)(bx, -bv)).reshape(-1)
            constructor = runner.OddEvenConstructor1D(
                domain=dict(config.mesh.domain),
                strides=dict(r["strides"]),
                Jn={"j": J, "r": J},
                scale=old["scale"],
                init_rng=jax.random.key(seed),
                kn=old["epsilon"],
                activation=jnp.tanh,
                coefficients=jnp.asarray(c),
            )
            cp = constructor.init(
                jax.random.key(seed), jnp.zeros(1), jnp.ones(1)
            )
            legacy_trace = np.asarray(
                jax.vmap(lambda xx, vv: constructor.apply(cp, xx, vv))(bx, -bv)
            ).reshape(-1)
            r.update(
                legacy_right_trace_mismatch=float(
                    np.max(np.abs(legacy_trace - trained))
                ),
                folded_right_trace_mismatch=float(
                    np.max(np.abs(folded_trace - trained))
                ),
            )
            np.savez_compressed(
                OUT / "raw" / f"{stem}_folded.npz",
                x=x,
                v=v,
                f=folded_f,
                rho=rho,
            )
        records.append(r)
        (OUT / "audit.json").write_text(json.dumps(records, indent=2))
        print(
            json.dumps(
                {
                    k: r[k]
                    for k in [
                        "features_per_patch",
                        "seed",
                        "variant",
                        "relative_l2_f",
                        "archive_Ef",
                        "folded_Ef",
                        "legacy_right_trace_mismatch",
                        "folded_right_trace_mismatch",
                    ]
                    if k in r
                }
            ),
            flush=True,
        )
        jax.clear_caches()


if __name__ == "__main__":
    import os

    if os.environ.get("OE_RUN_HISTORICAL_DIAGNOSTICS") != "1":
        raise SystemExit(
            "Archived diagnostic only. See archive/README.md; not a production experiment."
        )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study", choices=("budget", "mechanism"), default="budget"
    )
    args = parser.parse_args()
    if args.study == "mechanism":
        OUT = ROOT / "results/review_audit_20261010/parity_mechanism"
        ARCHIVE = (
            ROOT
            / ".release-backup/20261009/original-worktree/results/ablation"
        )
    main()
