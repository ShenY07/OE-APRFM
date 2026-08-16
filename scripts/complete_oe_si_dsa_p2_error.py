#!/usr/bin/env python3
"""Complete P2 OE-SI-DSA phase-space errors on its saved evaluation grid."""

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
for epsilon in (1.0, 1e-3):
    tag = f"{epsilon:.0e}"
    base = ROOT / "results/baselines/oe_si_dsa" / f"p2_oe_si_dsa_eps_{tag}"
    reference = ROOT / "results/references" / f"p2_parity_ref_eps_{tag}_level_B.npz"
    with np.load(base.with_suffix(".npz")) as result, np.load(reference) as ref:
        interpolator = RegularGridInterpolator((ref["x"], ref["velocity"]), ref["f"],
                                               bounds_error=True)
        xx, vv = np.meshgrid(result["x"], result["velocity"], indexing="ij")
        target = interpolator(np.column_stack((xx.ravel(), vv.ravel()))).reshape(xx.shape)
        error = float(np.linalg.norm(result["f"] - target) / np.linalg.norm(target))
    metrics_path = base.with_suffix(".json")
    metrics = json.loads(metrics_path.read_text())
    metrics["relative_l2_f"] = error
    metrics["phase_reference"] = str(reference.relative_to(ROOT))
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(tag, f"E_f={error:.6e}")
