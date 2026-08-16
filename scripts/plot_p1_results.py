"""Create the P1 manuscript figure from the aggregated CSV table."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-p1-summary")

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    table = Path("results/tables/p1_summary.csv")
    rows = list(csv.DictReader(table.open()))
    rows.sort(key=lambda row: float(row["epsilon"]), reverse=True)
    epsilon = np.asarray([float(row["epsilon"]) for row in rows])
    specs = (
        ("relative_l2_f", r"relative $L^2$ error $E_f$"),
        ("condition_number", r"scaled condition number $\kappa_2$"),
        ("total_seconds", "wall-clock time (s)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), constrained_layout=True)
    for axis, (metric, label) in zip(axes, specs):
        median = np.asarray([float(row[f"{metric}_median"]) for row in rows])
        lower = np.asarray([float(row[f"{metric}_min"]) for row in rows])
        upper = np.asarray([float(row[f"{metric}_max"]) for row in rows])
        axis.plot(epsilon, median, "o-", color="#1f5a94", label="median")
        axis.fill_between(epsilon, lower, upper, color="#1f5a94", alpha=0.2, label="[min, max]")
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_xlabel(r"Knudsen number $\varepsilon$")
        axis.set_ylabel(label)
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].legend(frameon=False)
    output = Path("results/figures/p1_accuracy_conditioning_time.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    print(output)


if __name__ == "__main__":
    main()
