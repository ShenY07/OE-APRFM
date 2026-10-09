"""Postprocess archived fields; never infer nonzero relative errors at zero truth.

Angular averages use normalized quadrature. Spatial norms use trapezoidal
weights for endpoint grids, cell-volume weights for deterministic cell grids.
P1/P3 are isotropic; P6 has a nonzero scaled flux and microscopic component.
"""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/requirement_2026_08_20"
OUT = BASE / "transport_diagnostics"


def trap_weights(x):
    x = np.asarray(x)
    assert len(x) > 1 and np.all(np.diff(x) > 0)
    w = np.empty_like(x, dtype=float)
    w[0] = (x[1] - x[0]) / 2
    w[-1] = (x[-1] - x[-2]) / 2
    w[1:-1] = (x[2:] - x[:-2]) / 2
    return w


def compute(f, truth, aw, directions, sw, epsilon, zero_truth=False):
    aw = np.asarray(aw, dtype=float)
    aw = aw / aw.sum()
    directions = np.asarray(directions).reshape(len(aw), -1)
    rho = f @ aw
    rho_true = truth @ aw
    g = (f - rho[..., None]) / epsilon
    gt = (truth - rho_true[..., None]) / epsilon
    flux = np.einsum("...q,qd,q->...d", f, directions, aw) / epsilon
    ft = np.einsum("...q,qd,q->...d", truth, directions, aw) / epsilon
    if zero_truth:
        gt = np.zeros_like(gt)
        ft = np.zeros_like(ft)
    gnorm = float(np.sqrt(np.sum(sw * np.sum(gt**2 * aw, axis=-1))))
    fnorm = float(np.sqrt(np.sum(sw * np.sum(ft**2, axis=-1))))
    ge = float(np.sqrt(np.sum(sw * np.sum((g - gt) ** 2 * aw, axis=-1))))
    fe = float(np.sqrt(np.sum(sw * np.sum((flux - ft) ** 2, axis=-1))))
    result = dict(
        flux_absolute_l2=fe,
        g_absolute_l2=ge,
        flux_reference_l2=fnorm,
        g_reference_l2=gnorm,
        flux_relative_l2=fe / fnorm if fnorm > 1e-14 else None,
        g_relative_l2=ge / gnorm if gnorm > 1e-14 else None,
    )
    opposite = np.argmin(
        np.sum(
            (directions[:, None, :] + directions[None, :, :]) ** 2, axis=-1
        ),
        axis=1,
    )
    assert np.allclose(directions[opposite], -directions, atol=1e-12)
    assert np.allclose(aw[opposite], aw, atol=1e-12)
    dg = g - gt
    even = (dg + dg[..., opposite]) / 2
    odd = (dg - dg[..., opposite]) / 2
    result["g_even_error_l2"] = float(
        np.sqrt(np.sum(sw * np.sum(even**2 * aw, axis=-1)))
    )
    result["g_odd_error_l2"] = float(
        np.sqrt(np.sum(sw * np.sum(odd**2 * aw, axis=-1)))
    )
    assert np.isclose(
        ge**2,
        result["g_even_error_l2"] ** 2 + result["g_odd_error_l2"] ** 2,
        rtol=1e-9,
        atol=1e-20,
    )
    return result


def sources():
    with (
        BASE / "tables_frozen/table_S2b_efficiency_seedwise.csv"
    ).open() as f:
        expected = {
            (r["problem"], r["method"], r["budget"], str(r["seed"]))
            for r in csv.DictReader(f)
        }
    seen = set()
    paths = list((BASE / "efficiency_raw").glob("*/*.json")) + list(
        (BASE / "efficiency_raw_four_component_normalized").glob("*/*.json")
    )
    paths += list(
        (BASE / "oe_dsa_si_sweep").glob("p3_oe_dsa_si_N*_A16_eps_*.json")
    )
    paths += list(
        (BASE / "oe_sn_krylov_verified_gauss").glob(
            "p1_oe_sn_krylov_N*_eps_*.json"
        )
    )
    paths += list((BASE / "p6_raw").glob("*.json"))
    for path in sorted(paths):
        m = json.loads(path.read_text())
        problem = m.get("problem", "").lower()
        if not problem:
            continue
        deterministic = "oe_dsa_si_sweep" in str(
            path
        ) or "oe_sn_krylov_verified_gauss" in str(path)
        if deterministic:
            if not m.get("converged"):
                continue
            method = r"OE-$S_N$-Krylov"
            seed = "deterministic"
            grid = m["grid"]
            budget = (
                "x".join(map(str, grid[:2] + [4 * grid[2]]))
                if problem == "p3"
                else "x".join(map(str, grid))
            )
        elif path.name == "metrics.json":
            method = "OE-APNN"
            seed = str(m["seed"])
            budget = f"{m['steps']} steps"
        elif "mm_" in path.parent.name:
            method = "MM-APRFM"
            seed = str(m["seed"])
            j = m.get(
                "features_per_field_patch", m.get("rho_features_per_patch")
            )
            budget = f"J={j}"
        elif m.get("method") == "oe_aprfm":
            if (
                problem == "p3"
                and "efficiency_raw_four_component_normalized" not in str(path)
            ):
                continue
            method = "OE-APRFM"
            seed = str(m["seed"])
            budget = f"J={m['features_per_patch']}"
        else:
            continue
        key = (problem, method, budget, seed)
        if problem != "p6" and key not in expected:
            continue
        if problem != "p6":
            assert key not in seen, key
            seen.add(key)
        npz = (
            path.parent / "solution.npz"
            if path.name == "metrics.json"
            else path.with_suffix(".npz")
        )
        yield path, npz, m, dict(
            problem=problem,
            method=method,
            budget=budget,
            seed=seed,
            epsilon=m["epsilon"],
            grid_type="cell centers" if deterministic else "endpoint grid",
        )
    assert seen == expected, ("Missing S3 metadata", expected - seen)


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    missing = []
    for path, npz, m, record in sources():
        if not npz.exists():
            missing.append(str(npz.relative_to(ROOT)))
            continue
        with np.load(npz) as z:
            f = z["f"]
            truth = z["reference_f"] if "reference_f" in z else z["exact"]
            assert f.shape == truth.shape and np.isfinite(f).all()
            if "theta" in z:
                theta = z["theta"]
                directions = np.column_stack((np.cos(theta), np.sin(theta)))
                if "weights_quadrant" in z:
                    aw = np.tile(z["weights_quadrant"], 4)
                else:
                    assert np.allclose(np.diff(theta), 2 * np.pi / len(theta))
                    aw = np.ones(len(theta))
            else:
                v = z["v"] if "v" in z else z["velocity"]
                directions = v[:, None]
                aw = z["weights"] if "weights" in z else trap_weights(v)
            sw = None
            for coord in ("x", "y"):
                if coord not in z:
                    continue
                x = z[coord]
                weights = (
                    np.full(len(x), x[1] - x[0])
                    if record["grid_type"] == "cell centers"
                    else trap_weights(x)
                )
                sw = weights if sw is None else sw[:, None] * weights[None, :]
            if "mask" in z:
                assert z["mask"].shape == sw.shape
                sw = sw * z["mask"]
            zero = record["problem"] in ("p1", "p3")
            if zero:
                assert np.max(np.abs(truth - truth[..., :1])) < 1e-12
            record.update(
                compute(f, truth, aw, directions, sw, record["epsilon"], zero)
            )
            record.update(
                reference_kind="identically zero" if zero else "nonzero",
                source=str(npz.relative_to(ROOT)),
                sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
                angular_nodes=len(aw),
            )
            rows.append(record)

    def write(name, items):
        with (OUT / name).open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(items[0]))
            w.writeheader()
            w.writerows(items)

    write("seedwise.csv", rows)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["problem"], r["epsilon"], r["method"], r["budget"])].append(
            r
        )
    summary = []
    metrics = [
        "flux_absolute_l2",
        "g_absolute_l2",
        "flux_relative_l2",
        "g_relative_l2",
        "g_even_error_l2",
        "g_odd_error_l2",
    ]
    for key, group in sorted(groups.items()):
        r = dict(zip(("problem", "epsilon", "method", "budget"), key))
        r["seeds"] = len(group)
        for metric in metrics:
            values = [x[metric] for x in group if x[metric] is not None]
            for stat, fn in [
                ("median", np.median),
                ("min", min),
                ("max", max),
            ]:
                r[metric + "_" + stat] = float(fn(values)) if values else None
        summary.append(r)
    write("summary.csv", summary)
    (OUT / "missing_archives.json").write_text(json.dumps(missing, indent=2))
    lines = [
        "# Archived transport diagnostics",
        "",
        "These are postprocessed saved fields, not new solves. P1/P3 in S3 have identically zero exact flux and g=(f-rho)/epsilon; only absolute L2 errors are meaningful. Their relative errors are blank, not zero.",
        "",
        "Flux is <v f>/epsilon (a two-component vector in 2D). All angular weights are normalized. rho is recomputed from f with the same rule. Spatial L2 norms use physical integration weights, not raw unweighted array norms. P6 uses nonzero manufactured truth and permits relative errors.",
        "",
        "Learned methods use their archived evaluation grids (P1 257x128; P3 65x65x64); deterministic methods retain native cell/angle grids with saved Gauss weights. No interpolation is introduced. Tiny absolute errors near roundoff should not be ranked; a publication comparison requires common evaluation and quadrature-refinement checks.",
        "",
        "All S3 metadata are matched by problem/method/budget/seed; legacy two-component OE P3 runs are excluded. Source paths and SHA256 are recorded in seedwise.csv. No P6 external-method result is inferred.",
        "",
        "Even/odd g-error components are also reported. Orthogonality is checked seedwise: ||error_g||^2=||error_g_even||^2+||error_g_odd||^2. Stable odd-flux accuracy does not imply stable accuracy of the entire scaled microscopic field.",
        "",
        "| Problem | epsilon | Method | Budget | abs flux L2 | abs g L2 | relative flux | relative g | even g error | odd g error |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary:
        vals = [r[k + "_median"] for k in metrics]
        formatted = ["N/A" if v is None else f"{v:.4e}" for v in vals]
        lines.append(
            "| "
            + " | ".join(
                [r["problem"], str(r["epsilon"]), r["method"], r["budget"]]
                + formatted
            )
            + " |"
        )
    lines += [
        "",
        "Missing field archives: " + str(len(missing)),
        "",
        "Reproduce: `python3 scripts/build_transport_diagnostics.py`.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        f"Processed {len(rows)} fields into {len(summary)} groups; missing archives: {len(missing)}"
    )


if __name__ == "__main__":
    main()
