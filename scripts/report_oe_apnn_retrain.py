#!/usr/bin/env python3
"""Refresh tables and figures from complete three-seed hard-parity runs only."""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def save(fig, path):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)


def latex(value):
    mantissa, exponent = f"{value:.3e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def update_manuscript(out, summaries):
    original = out / "original/table.md"
    if not original.exists():
        return
    # Other concurrent work may refresh non-APNN rows in this manuscript.
    # Always patch the current file; the original is a recovery snapshot only.
    text = (ROOT / "results/table.md").read_text()
    by_key = {
        (s["suite"], s["problem"], s["epsilon"], s["steps"]): s
        for s in summaries
    }
    for problem, label in (
        ("p1", "tab:supp-efficiency"),
        ("p3", "tab:supp-efficiency-2d"),
    ):
        pattern = r"\\begin\{table\}.*?\\end\{table\}"

        def replace(match):
            block = match.group()
            if "\\label{" + label + "}" not in block:
                return block
            for steps in (500, 2000, 5000):
                row_pattern = rf"^(?:OE-APNN)?\s*& {steps} steps & .*?$"
                s = by_key.get(("cost", problem, 0.001, steps))
                prefix = "OE-APNN" if steps == 500 else " "
                n = 25474 if problem == "p1" else 25730
                values = (
                    f"{latex(s['E_f_median'])} & {latex(s['E_rho_median'])} & {s['train_time_s_median']:.2f}"
                    if s
                    else r"\text{pending} & \text{pending} & \text{pending}"
                )
                new = (
                    f"{prefix} & {steps} steps & {n} & 5120 & {values}" + r"\\"
                )
                block = re.sub(row_pattern, lambda _: new, block, flags=re.M)
            note = (
                r" Hard-parity OE-APNN uses seeds 7, 11, 17, 4096 interior and 1024 boundary phase samples per step "
                r"($N_{\rm phase}=5120$, replacing the previous 640-sample budget); $N_{\rm res}$ in its rows denotes this phase-sample budget, not scalar equation count. "
                r"Times are concurrent CUDA training wall times on A800 GPUs with three seeds per GPU; other-method times are historical and do not establish a same-hardware speedup."
            )
            if "Hard-parity OE-APNN uses seeds" in block:
                return block
            pos = block.index(r"\label")
            end = block.rfind("}", 0, pos)
            return block[:end] + note + block[end:]

        text = re.sub(pattern, replace, text, flags=re.S)
    # Two summary rows reuse the 5000-step cost runs.
    pattern = r"\\begin\{table\}.*?\\end\{table\}"

    def summary_block(match):
        block = match.group()
        if r"\label{tab:efficiency-summary}" not in block:
            return block
        for problem, count in (("p1", 25474), ("p3", 25730)):
            s = by_key.get(("cost", problem, 0.001, 5000))
            values = (
                f"{latex(s['E_f_median'])} & {latex(s['E_rho_median'])} & {s['train_time_s_median']:.2f}"
                if s
                else r"\text{pending} & \text{pending} & \text{pending}"
            )
            pat = rf"(   & OE-APNN & {count}\s*\n)   & .*?\\\\"
            block = re.sub(
                pat, lambda m: m[1] + "   & " + values + r"\\", block
            )
        pos = block.index(r"\label")
        end = block.rfind("}", 0, pos)
        note = r" OE-APNN uses the new hard-parity 5000-step, 5120-phase-sample CUDA protocol (seeds 7, 11, 17); concurrent GPU times are not directly comparable to historical timings of other methods."
        if "OE-APNN uses the new hard-parity" in block:
            return block
        return block[:end] + note + block[end:]

    text = re.sub(pattern, summary_block, text, flags=re.S)
    target = ROOT / "results/table.md"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/oe_apnn_parity_20261010",
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
    update_manuscript(out, summaries)
    original_figure = out / "original/figure.md"
    if original_figure.exists():
        additions = []
        for problem in ("p1", "p3"):
            path = figures / (problem + "_cost_accuracy.pdf")
            if path.exists():
                rel = path.relative_to(ROOT / "results").as_posix()
                additions.append(
                    r"\begin{figure}[!htbp]"
                    + "\n"
                    + r"\centering"
                    + "\n"
                    + r"\includegraphics[width=0.8\linewidth]{"
                    + rel
                    + "}\n"
                    + r"\caption{Hard-parity OE-APNN "
                    + problem.upper()
                    + r" retraining at $\varepsilon=10^{-3}$ with 4096 interior and 1024 boundary phase samples. Medians and minimum--maximum bands use seeds 7, 11, 17. Only completed three-seed budgets are shown. Kinetic and density errors are plotted on a shared axis.}"
                    + "\n"
                    + r"\label{fig:apnn-hard-parity-"
                    + problem
                    + "}\n"
                    + r"\end{figure}"
                )
        target = ROOT / "results/figure.md"
        current = target.read_text()
        current = re.sub(
            r"\\begin\{figure\}.*?\\end\{figure\}",
            lambda m: "" if r"\label{fig:apnn-hard-parity-" in m[0] else m[0],
            current,
            flags=re.S,
        )
        target.write_text(
            current.rstrip() + "\n\n" + "\n\n".join(additions) + "\n"
        )
    print(
        f"Refreshed {len(seedwise)}/48 runs, {len(summaries)}/16 complete groups",
        flush=True,
    )


if __name__ == "__main__":
    main()
