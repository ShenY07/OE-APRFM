"""Independent deterministic transport and test quadrature checks for E3."""

import json
import numpy as np
from run_e3_slab import ROOT, fs, symmetric, rf_evaluator, relative
from configuration.p1_manufactured_1d import get_config
from numerical.parity_reference import solve_parity_gmres_1d

out = ROOT / "results/e3_slab"
fs.uniform = symmetric
for eps in (1.0, 0.1):
    config = get_config(eps)
    config.model.source = lambda x, v: np.zeros_like(v)
    refs = []
    for nx, nv in ((256, 64), (512, 128)):
        path = out / f"reference_eps{eps:.0e}_{nx}_{nv}.npz"
        if path.exists():
            refs.append(dict(np.load(path)))
            continue
        ref = solve_parity_gmres_1d(config, grid=(nx, nv), tol=1e-11)
        np.savez_compressed(path, **ref)
        refs.append(ref)
    coarse, fine = refs
    if not all(bool(ref["converged"]) for ref in refs):
        raise RuntimeError(
            "Unconverged E3 reference; errors must not be accepted"
        )
    # Compare on the coarse grid without extrapolating spatial endpoints.
    fspace = np.stack(
        [
            np.interp(coarse["x"], fine["x"], fine["f"][:, k])
            for k in range(len(fine["velocity"]))
        ],
        axis=1,
    )
    fcommon = np.stack(
        [np.interp(coarse["velocity"], fine["velocity"], a) for a in fspace]
    )
    refdiff = relative(
        fcommon,
        coarse["f"],
        np.broadcast_to(coarse["weights"], coarse["f"].shape),
    )
    rhodiff = relative(
        np.interp(coarse["x"], fine["x"], fine["rho"]),
        coarse["rho"],
        np.ones_like(coarse["x"]),
    )
    for seed in (11, 23, 37):
        path = out / f"seed{seed}_eps{eps:.0e}.json"
        row = json.loads(path.read_text())
        c = np.load(path.with_suffix(".npz"))["coefficients"]
        ev = rf_evaluator(64, seed)
        pred = []
        for start in range(0, len(fine["x"]), 32):
            x, v = np.meshgrid(
                fine["x"][start : start + 32], fine["velocity"], indexing="ij"
            )
            r, j, _, _ = ev(x, v)
            pred.append(r @ c + eps * (j @ c))
        pred = np.concatenate(pred)
        rho = pred @ fine["weights"] / 2
        ef = relative(
            pred, fine["f"], np.broadcast_to(fine["weights"], pred.shape)
        )
        er = relative(rho, fine["rho"], np.ones_like(rho))
        row["transport_reference"] = dict(
            E_f=ef,
            E_rho=er,
            refinement_f=refdiff,
            refinement_rho=rhodiff,
            refinement_below_ten_percent=bool(
                refdiff < 0.1 * ef and rhodiff < 0.1 * er
            ),
        )
        row["reference_status"] = "evaluated; see refinement_below_ten_percent"
        path.write_text(json.dumps(row, indent=2) + "\n")
        print(eps, seed, row["transport_reference"], flush=True)
