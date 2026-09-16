#!/usr/bin/env python3
"""Build the two title-free direct-RFM limitation panels for P1."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oeraprfm-direct-limitation-mpl")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/requirement_2026_08_20/direct_rfm_limitation_seeded_raw"
OUT = ROOT / "results/requirement_2026_08_20/figures_separate"
DATA = ROOT / "results/requirement_2026_08_20/tables_frozen"


def residual_response(epsilons: np.ndarray) -> np.ndarray:
    """Numerically evaluate ||A_eps phi||/||phi|| for phi=sin(pi*x)."""
    zx, wx = leggauss(128)
    x, wx = 0.5 * (zx + 1.0), 0.5 * wx
    v, wv = leggauss(128)
    phi = np.sin(np.pi * x)[:, None] + np.zeros((1, v.size))
    derivative = np.pi * np.cos(np.pi * x)[:, None]
    weights = wx[:, None] * wv[None, :]
    denominator = np.sqrt(np.sum(weights * phi**2))
    return np.asarray([
        np.sqrt(np.sum(weights * (epsilon * v[None, :] * derivative) ** 2))
        / denominator
        for epsilon in epsilons
    ])


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{stem}.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    records = [json.loads(path.read_text()) for path in sorted(RAW.glob("*.json"))]
    if len(records) != 39:
        raise RuntimeError(f"expected 39 direct-RFM records, found {len(records)}")
    groups: dict[float, list[dict]] = defaultdict(list)
    for record in records:
        if record["num_columns"] != 128 or record["num_rows"] != 1088:
            raise ValueError("direct-RFM budget changed inside the epsilon sweep")
        groups[float(record["epsilon"])].append(record)
    epsilons = np.asarray(sorted(groups))
    if any(sorted(r["seed"] for r in group) != [11, 23, 37] for group in groups.values()):
        raise ValueError("each epsilon must contain seeds 11, 23, and 37")

    gamma = residual_response(epsilons)
    reference = gamma[-1] * epsilons / epsilons[-1]
    slope = float(np.polyfit(np.log(epsilons), np.log(gamma), 1)[0])
    medians = np.asarray([np.median([r["relative_l2_f"] for r in groups[e]]) for e in epsilons])
    minima = np.asarray([np.min([r["relative_l2_f"] for r in groups[e]]) for e in epsilons])
    maxima = np.asarray([np.max([r["relative_l2_f"] for r in groups[e]]) for e in epsilons])

    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "direct_rfm_macroscopic_residual_response.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epsilon", "gamma_direct", "reference_O_epsilon"))
        writer.writerows(zip(epsilons, gamma, reference))
    with (DATA / "direct_rfm_fixed_discretization_error.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epsilon", "median_E_f", "min_E_f", "max_E_f", "seeds"))
        writer.writerows((e, m, lo, hi, 3) for e, m, lo, hi in zip(epsilons, medians, minima, maxima))
    metadata = {
        "problem": "p1",
        "perturbation": "sin(pi*x)",
        "observed_loglog_slope_gamma": slope,
        "theoretical_prefactor_pi_over_sqrt3": float(np.pi / np.sqrt(3.0)),
        "direct_rfm": {
            "seeds": [11, 23, 37], "partitions": [1, 1],
            "num_columns": 128, "num_rows": 1088,
            "interior_collocation": [30, 32], "boundary_collocation_per_side": 64,
            "collision_quadrature_order": 8, "feature_scale": 1.0,
            "rcond": 1.0e-12, "row_scaling": "unit-L2",
        },
    }
    (DATA / "direct_rfm_limitation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.4, 2.75), constrained_layout=True)
    ax.loglog(epsilons, gamma, color="#0072B2", marker="o", label=r"computed $\gamma_{\rm dir}$")
    ax.loglog(epsilons, reference, color="#333333", linestyle="--", label=r"$\mathcal{O}(\varepsilon)$")
    ax.set_xlabel(r"Knudsen number $\varepsilon$")
    ax.set_ylabel("Normalized residual response")
    ax.grid(True, which="major", color="#B8B8B8", alpha=0.35, linewidth=0.55)
    ax.grid(False, which="minor")
    ax.legend(frameon=False)
    save(fig, "fig00_direct_rfm_macroscopic_residual_response")

    fig, ax = plt.subplots(figsize=(3.4, 2.75), constrained_layout=True)
    ax.fill_between(epsilons, minima, maxima, color="#009E73", alpha=0.15, linewidth=0)
    ax.loglog(epsilons, medians, color="#009E73", marker="D", label="Direct RFM")
    ax.set_xlabel(r"Knudsen number $\varepsilon$")
    ax.set_ylabel(r"Relative $L^2$ error $E_f$")
    ax.grid(True, which="major", color="#B8B8B8", alpha=0.35, linewidth=0.55)
    ax.grid(False, which="minor")
    ax.legend(frameon=False)
    save(fig, "fig00_direct_rfm_fixed_discretization_error")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
