#!/usr/bin/env python3
"""Aggregate P3/P4 three-seed errors and draw title-free method comparisons."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = {7, 11, 17}


def load_json(paths, method, problem, epsilon):
    found = {}
    for path in paths:
        data = json.loads(path.read_text())
        if str(data.get("problem", "")).lower() != problem.lower():
            continue
        if not np.isclose(float(data.get("epsilon", -1)), epsilon):
            continue
        seed = int(data.get("seed", -1))
        if seed in SEEDS:
            found[seed] = {
                "method": method,
                "problem": problem,
                "epsilon": epsilon,
                "seed": seed,
                "E_f": float(data["relative_l2_f"]),
                "E_rho": float(data.get("relative_l2_rho", data["relative_l2_f"])),
            }
    return found


def records():
    rows = []
    for problem in ("P1", "P2", "P3", "P4"):
        for epsilon, label in ((1.0, "eps_1e0"), (1e-3, "eps_1e-3")):
            apnn = {}
            for root in (ROOT / "results/baselines/oe_apnn", ROOT / "results/baselines/oe_apnn_imported"):
                for path in (root / problem / label).glob("seed_*/metrics.json"):
                    data = json.loads(path.read_text())
                    if data.get("evaluation_status") == "complete":
                        apnn[int(data["seed"])] = {"method": "OE-APNN", "problem": problem,
                            "epsilon": epsilon, "seed": int(data["seed"]),
                            "E_f": float(data["E_f"]), "E_rho": float(data["E_rho"])}
            oe_paths = list((ROOT / "results/baselines/oe_aprfm_3seed").glob("*.json"))
            oe_paths += list((ROOT / "results/quadrant/final").glob("*.json"))
            oe_paths += list((ROOT / "results/consistency/p2").glob("*fixed.json"))
            oe_paths += list((ROOT / "results/domain_decomposition/1d_fixed_local").glob("*p11_f64.json"))
            mm_paths = list((ROOT / "results/baselines/mm_aprfm").glob("*.json"))
            groups = [apnn, load_json(oe_paths, "OE-APRFM", problem, epsilon),
                      load_json(mm_paths, "MM-APRFM", problem, epsilon)]
            for group in groups:
                rows.extend(group.values())
    return rows


def main():
    rows = records()
    table = ROOT / "results/tables/three_seed_method_comparison.md"
    lines = ["# 三种子方法对比", "",
             "| 问题 | ε | 方法 | E_f 均值 | E_f 标准差 | E_ρ 均值 | E_ρ 标准差 | 种子数 |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    out = ROOT / "results/figures/method_comparison"
    out.mkdir(parents=True, exist_ok=True)
    methods = ("OE-APRFM", "MM-APRFM", "OE-APNN")
    colors = ("#0072B2", "#D55E00", "#009E73")
    for problem in ("P1", "P2", "P3", "P4"):
        for metric, short in (("E_f", "ef"), ("E_rho", "erho")):
            fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
            x = np.arange(2)
            for offset, (method, color) in enumerate(zip(methods, colors)):
                means, stds = [], []
                for epsilon in (1.0, 1e-3):
                    vals = np.array([r[metric] for r in rows if r["problem"] == problem
                                     and r["method"] == method and r["epsilon"] == epsilon])
                    means.append(vals.mean() if len(vals) == 3 else np.nan)
                    stds.append(vals.std(ddof=1) if len(vals) == 3 else np.nan)
                ax.errorbar(x + (offset - 1) * 0.08, means, yerr=stds, marker="o",
                            capsize=4, linewidth=1.6, color=color, label=method)
            ax.set_xticks(x, [r"$1$", r"$10^{-3}$"])
            ax.set_xlabel(r"$\varepsilon$")
            ax.set_ylabel(r"$E_f$" if metric == "E_f" else r"$E_\rho$")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=.25)
            ax.legend(frameon=False)
            fig.savefig(out / f"cmp_{problem.lower()}_{short}.png", dpi=320)
            plt.close(fig)
        for epsilon in (1.0, 1e-3):
            for method in methods:
                subset = [r for r in rows if r["problem"] == problem and r["method"] == method
                          and r["epsilon"] == epsilon]
                if len(subset) != 3:
                    continue
                ef = np.array([r["E_f"] for r in subset]); er = np.array([r["E_rho"] for r in subset])
                lines.append(f"| {problem} | {epsilon:.0e} | {method} | {ef.mean():.3e} | "
                             f"{ef.std(ddof=1):.3e} | {er.mean():.3e} | {er.std(ddof=1):.3e} | 3 |")
    table.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
