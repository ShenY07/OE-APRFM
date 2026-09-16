#!/usr/bin/env python3
"""Build the report-requested tables/figures without mixing obsolete P5 data."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oeraprfm-requirements-mpl")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "requirement_2026_08_20"
TABLES, FIGURES = OUT / "tables", OUT / "figures"


def load(pattern):
    return [json.loads(p.read_text()) for p in sorted(ROOT.glob(pattern))]


def write_csv(name, rows):
    path = TABLES / name
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def median_rows(records, keys, values):
    groups = defaultdict(list)
    for record in records:
        groups[tuple(record[key] for key in keys)].append(record)
    rows = []
    for group, members in sorted(groups.items()):
        row = dict(zip(keys, group)); row["seeds"] = len(members)
        for value in values:
            data = [float(member[value]) for member in members if member.get(value) is not None]
            row[value] = float(np.median(data)) if data else ""
        rows.append(row)
    return rows


def main():
    TABLES.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    scan = [r for r in load("results/epsilon_scan/*.json") if r["problem"] in ("p1", "p3")]
    accuracy = median_rows(scan, ("problem", "epsilon"), ("relative_l2_f", "relative_l2_rho", "empirical_stability_ratio"))
    write_csv("table_2_accuracy_knudsen.csv", accuracy)

    from feature_resolution_data import feature_summary, export as export_features
    summary = feature_summary()
    export_features(TABLES)

    collocation = load("results/collocation_sufficiency/*.json")
    write_csv("table_4_collocation_refinement.csv", median_rows(collocation, ("problem", "oversampling_ratio"), ("relative_l2_f", "relative_l2_rho")))

    mechanism = load("results/ablation/*.json")
    mechanism += [r for r in load("results/baselines/rfm/p1_*.json") if r.get("seed") == 11]
    mechanism += [r for r in load("results/baselines/mm_aprfm/p1_*.json") if r.get("seed") == 11]
    mechanism += [r for r in load("results/baselines/oe_apnn_imported/p1_*.json")]
    names = {"full": "OE-APRFM", "oe_original": "OE without projection/rescaling", "full_angular": "unconstrained angular RFM", "rfm": "Direct RFM", "mm_aprfm": "MM-APRFM", "oe_apnn": "OE-APNN"}
    mech_rows = []
    for r in mechanism:
        key = r.get("variant", r.get("method"));
        if key in names and r.get("relative_l2_f") is not None:
            mech_rows.append({"method": names[key], "epsilon": r["epsilon"], "seed": r.get("seed", ""), "relative_l2_f": r["relative_l2_f"]})
    write_csv("table_5_formulation_comparison_raw.csv", mech_rows)

    refmeta = load("results/references_p5_smooth/p5_*_level_*.json")
    write_csv("table_7_p5_reference_status.csv", refmeta)

    colors = {1.0: "#3B6FB6", 1e-3: "#E07A5F", 1e-6: "#4C956C"}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), constrained_layout=True)
    for ax, metric, ylabel in zip(axes, ("relative_l2_f", "relative_l2_rho"), (r"$E_f$", r"$E_\rho$")):
        for problem, marker in (("p1", "o"), ("p3", "s")):
            rows = [r for r in accuracy if r["problem"] == problem]
            ax.loglog([r["epsilon"] for r in rows], [r[metric] for r in rows], marker + "-", label=problem.upper())
        ax.invert_xaxis(); ax.set(xlabel=r"$\varepsilon$", ylabel=ylabel); ax.grid(True, which="both", alpha=.25)
    axes[0].legend(frameon=False); fig.savefig(FIGURES / "figure_1_error_vs_epsilon.png", dpi=320); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), constrained_layout=True)
    for col, problem in enumerate(("p1", "p3")):
        for eps in sorted({r["epsilon"] for r in summary}):
            rows = [r for r in summary if r["problem"] == problem and r["epsilon"] == eps]
            axes[0, col].loglog([r["features_per_patch"] for r in rows], [r["relative_l2_f"] for r in rows], "o-", color=colors[eps], label=fr"$\varepsilon={eps:.0e}$")
            axes[1, col].semilogx([r["features_per_patch"] for r in rows], [r["rank_fraction"] for r in rows], "o-", color=colors[eps])
        axes[0, col].set(title=problem.upper(), ylabel=r"median $E_f$"); axes[1, col].set(xlabel="J", ylabel=r"median $r_{eff}/N_{coef}$");
        for ax in axes[:, col]: ax.grid(True, which="both", alpha=.25)
    axes[0, 0].legend(frameon=False, fontsize=8); fig.savefig(FIGURES / "figure_2_feature_resolution.png", dpi=320); plt.close(fig)

    status = """# 按《OE-APRFM 数值实验图表需求报告》整理的结果\n\n生成日期：2026-08-20。该目录只收录数学定义兼容的数据；旧 manufactured P5 数据已明确排除。\n\n## 完成状态\n\n- Table 2/3/4/5：已由现有 P1/P3 原始 JSON 重建，逐 seed 数据保留在 CSV。\n- Figure 1/2：已重建；仓库中未找到附件所称的 ε≈1e-16 原始 JSON，因此没有臆造极限尺度点。\n- 新 P5：物理定义已冻结为 ε-independent Gaussian source、vacuum inflow、heterogeneous disk/channel medium。ε=1 A/B reference 已完成，但 A/B discrepancy 为 Ef=7.1119e-2、Eρ=9.5688e-3，尚未达到高精度认证。\n- P5 ε=1e-3/1e-6：当前无 diffusion preconditioner 的 parity-GMRES 停滞，未写入正式误差表。\n- Accuracy-cost frontier：现有数据不是多预算 matched-accuracy sweep，故不生成误导性的 Pareto 图。\n- 旧 P5、旧 P5 的 MM-APRFM/OE-APNN 与图像均不得用于新 P5 横向比较。\n\n## 尚需计算\n\n1. 为 P5 reference solver 加 diffusion synthetic acceleration/preconditioner 后，完成 ε=1e-3/1e-6 的 A/B refinement。\n2. 基于认证 fine reference 重跑 OE-APRFM、MM-APRFM、OE-APNN 三个 seeds。\n3. 三方法各做至少三个预算点和三次 timing repeats，才能生成 Table 6 / Figure 4 / runtime breakdown。\n"""
    (OUT / "README.md").write_text(status)


if __name__ == "__main__":
    main()
