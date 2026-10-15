#!/usr/bin/env python3
"""Summarize and plot complete three-seed hard-parity training groups."""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def save(fig, path):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        target = path.with_suffix("." + ext)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/runs/neural_study",
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    seedwise, summaries = [], []
    for group in manifest["groups"]:
        records = []
        for seed in manifest["seeds"]:
            directory = out / "runs" / group["id"] / f"seed_{seed}"
            path = directory / "metrics.json"
            if not path.exists():
                continue
            m = json.loads(path.read_text())
            assert m["protocol"] == manifest["protocol"]
            assert m["seed"] == seed and m["steps"] == group["steps"]
            assert (
                m["interior_samples"] == 4096 and m["boundary_samples"] == 1024
            )
            assert (
                m["evaluation_status"] == "complete"
                and m["status"] == "success"
            )
            assert all(
                np.isfinite(m[k]) for k in ("E_f", "E_rho", "train_time_s")
            )
            record = dict(
                suite=group["suite"],
                problem=group["problem"],
                epsilon=group["epsilon"],
                steps=group["steps"],
                seed=seed,
                device=m["device"],
                E_f=m["E_f"],
                E_rho=m["E_rho"],
                train_time_s=m["train_time_s"],
                best_step=m["best_step"],
                source=str(path.relative_to(out)),
            )
            seedwise.append(record)
            records.append(record)
        if len(records) != 3:
            continue
        s = {k: group[k] for k in ("suite", "problem", "epsilon", "steps")}
        for metric in ("E_f", "E_rho", "train_time_s"):
            values = [r[metric] for r in records]
            for name, fn in (
                ("median", np.median),
                ("mean", np.mean),
                ("std", lambda a: np.std(a, ddof=1)),
                ("min", min),
                ("max", max),
            ):
                s[f"{metric}_{name}"] = float(fn(values))
        summaries.append(s)
        stem = group["id"].replace("/", "_")
        fig, ax = plt.subplots(figsize=(6, 4))
        for seed in manifest["seeds"]:
            with np.load(
                out / "runs" / group["id"] / f"seed_{seed}/history.npz"
            ) as z:
                ax.semilogy(z["step"], z["total_loss"], label=f"seed {seed}")
        ax.set(
            xlabel="Adam steps",
            ylabel="Fixed validation physics loss",
            title=f"{group['problem'].upper()}, epsilon={group['epsilon']:g}",
        )
        ax.legend()
        save(fig, figures / (stem + "_loss"))
        # Seed 11 shown consistently; never choose the best seed for field figures.
        with np.load(out / "runs" / group["id"] / "seed_11/solution.npz") as z:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
            if "y" in z:
                mask = z["mask"]
                for ax, key, title in zip(
                    axes,
                    ("reference_rho", "rho", "error_rho"),
                    (
                        "Reference density",
                        "OE-APNN density",
                        "Absolute density error",
                    ),
                ):
                    field = np.where(mask, z[key], np.nan)
                    im = ax.pcolormesh(z["x"], z["y"], field.T, shading="auto")
                    fig.colorbar(im, ax=ax)
                    ax.set(xlabel="x", ylabel="y", title=title, aspect="equal")
            else:
                axes[0].plot(z["x"], z["reference_rho"], label="Reference")
                axes[0].plot(z["x"], z["rho"], "--", label="OE-APNN")
                axes[0].legend()
                axes[0].set(xlabel="x", ylabel="Density")
                axes[1].plot(z["x"], z["error_rho"])
                axes[1].set(xlabel="x", ylabel="Absolute density error")
                im = axes[2].pcolormesh(
                    z["x"], z["velocity"], z["error_f"].T, shading="auto"
                )
                fig.colorbar(im, ax=axes[2])
                axes[2].set(xlabel="x", ylabel="v", title="Absolute f error")
            save(fig, figures / (stem + "_fields"))

    for name, rows in (("seedwise.csv", seedwise), ("summary.csv", summaries)):
        if rows:
            with (out / name).open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    lines = [
        "# OE-APNN hard-parity retraining",
        "",
        f"Completed runs: {len(seedwise)}/48; complete three-seed groups: {len(summaries)}/16.",
        "",
        "Seeds 7, 11, 17; float64; Adam lr=0.001; 4096 interior + 1024 boundary phase samples; 16 base quadrature nodes; unit loss weights. Best fixed-validation checkpoint, checked every 200 steps and at the final step. P5 is the square-hole manufactured problem, not heterogeneous transport.",
        "",
        "CUDA GPUs 1 and 2 each run three seeds concurrently. Timings include training and validation, exclude final field evaluation, and are not same-hardware speedups against historical CPU methods. P2 errors use the regenerated level-B numerical reference, not an exact solution. Incomplete groups are excluded from summary statistics.",
        "",
        "| Suite | Problem | epsilon | Steps | E_f median | E_rho median | Training seconds median |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['suite']} | {s['problem'].upper()} | {s['epsilon']:g} | {s['steps']} | {s['E_f_median']:.4e} | {s['E_rho_median']:.4e} | {s['train_time_s_median']:.2f} |"
        )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    for problem in ("p1", "p3"):
        rows = sorted(
            [
                s
                for s in summaries
                if s["suite"] == "cost" and s["problem"] == problem
            ],
            key=lambda r: r["steps"],
        )
        if rows:
            # Keep the manuscript panels consistent with the publication export.
            with plt.rc_context(
                {
                    "font.family": "DejaVu Sans",
                    "font.size": 10,
                    "axes.labelsize": 10,
                    "legend.fontsize": 10,
                    "pdf.fonttype": 42,
                    "ps.fonttype": 42,
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                }
            ):
                fig, ax = plt.subplots(figsize=(3.6, 3.6))
                x = [s["steps"] for s in rows]
                for metric, label, color, style in (
                    ("E_f", r"$E_f$", "#0072B2", "o-"),
                    ("E_rho", r"$E_\rho$", "#D55E00", "s--"),
                ):
                    ax.semilogy(
                        x,
                        [s[metric + "_median"] for s in rows],
                        style,
                        color=color,
                        label=label,
                        linewidth=1.5,
                        markersize=4,
                        markerfacecolor="white",
                    )
                    ax.fill_between(
                        x,
                        [s[metric + "_min"] for s in rows],
                        [s[metric + "_max"] for s in rows],
                        color=color,
                        alpha=0.14,
                        linewidth=0,
                    )
                ax.set(
                    xlabel="Adam steps",
                    ylabel="Relative error",
                    xticks=[500, 2000, 5000],
                )
                ax.grid(which="major", alpha=0.16, linewidth=0.5)
                ax.legend(
                    frameon=False,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.01),
                    ncol=2,
                )
                save(fig, figures / (problem + "_cost_accuracy"))
    print(
        f"Refreshed {len(seedwise)}/48 runs, {len(summaries)}/16 complete groups",
        flush=True,
    )


if __name__ == "__main__":
    main()
