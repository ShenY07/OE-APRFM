#!/usr/bin/env python3
"""Build baseline cost/resource table and title-free scientific cost figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "figures/method_comparison"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = {7, 11, 17}
COLORS = {"OE-APRFM": "#3B6FB6", "MM-APRFM": "#E07A5F", "OE-APNN": "#4C956C",
          "OE-SI-DSA": "#8E6C8A", "RFM": "#D6A84B", "SI-DSA": "#6C757D"}


def read(path):
    return json.loads(path.read_text())


def oe_seed11(problem, epsilon):
    tag = f"{epsilon:.0e}"
    if problem == "P1":
        path = next((RESULTS / "domain_decomposition/1d_fixed_local").glob(
            f"p1_oe_aprfm_eps_{tag}_seed_11_*p11_f64.json"))
    elif problem == "P2":
        path = RESULTS / "consistency/p2" / f"p2_oe_aprfm_eps_{tag}_seed_11_fixed.json"
    else:
        path = next((RESULTS / "quadrant/final").glob(
            f"{problem.lower()}_oe_aprfm_eps_{tag}_seed_11_*.json"))
    return read(path)


def resource_oe(data):
    if data.get("num_columns"):
        return int(data["num_columns"])
    return int(2 * np.prod(data["partitions"]) * data["features_per_patch"])


def collect():
    rows = []
    for pi in range(1, 6):
        problem = f"P{pi}"
        for epsilon, label in ((1.0, "eps_1e0"), (1e-3, "eps_1e-3")):
            oe = {11: oe_seed11(problem, epsilon)}
            for path in (RESULTS / "baselines/oe_aprfm_3seed").glob(f"p{pi}_*.json"):
                data = read(path)
                if int(data["seed"]) in SEEDS and np.isclose(data["epsilon"], epsilon):
                    oe[int(data["seed"])] = data
            vals = list(oe.values())
            rows.append(dict(problem=problem, epsilon=epsilon, method="OE-APRFM",
                             time=np.median([x["total_seconds"] for x in vals]),
                             time_min=np.min([x["total_seconds"] for x in vals]), time_max=np.max([x["total_seconds"] for x in vals]),
                             resource=resource_oe(vals[0]), resource_type="线性未知数", sample="3 seeds"))

            mm = {}
            for path in (RESULTS / "baselines/mm_aprfm").glob(f"p{pi}_*.json"):
                data = read(path)
                seed = int(data.get("seed", -1))
                if seed in SEEDS and np.isclose(data["epsilon"], epsilon):
                    mm[seed] = data
            if len(mm) == 3:
                vals = list(mm.values())
                rows.append(dict(problem=problem, epsilon=epsilon, method="MM-APRFM",
                                 time=np.median([x["total_seconds"] for x in vals]),
                                 time_min=np.min([x["total_seconds"] for x in vals]), time_max=np.max([x["total_seconds"] for x in vals]),
                                 resource=int(vals[0]["num_columns"]), resource_type="线性未知数", sample="3 seeds"))

            apnn_by_seed = {}
            for root in (RESULTS / "baselines/oe_apnn", RESULTS / "baselines/oe_apnn_imported"):
                for path in (root / problem / label).glob("seed_*/metrics.json"):
                    data = read(path)
                    if data.get("evaluation_status") == "complete": apnn_by_seed[int(data["seed"])] = data
            apnn = list(apnn_by_seed.values())
            if problem != "P5":
                rows.append(dict(problem=problem, epsilon=epsilon, method="OE-APNN",
                                 time=np.median([x["train_time_s"] for x in apnn]),
                                 time_min=np.min([x["train_time_s"] for x in apnn]), time_max=np.max([x["train_time_s"] for x in apnn]),
                                 resource=int(apnn[0]["parameter_count"]), resource_type="网络参数", sample="3 seeds"))

            suffix = "" if pi <= 2 else "_krylov"
            path = RESULTS / "baselines/oe_si_dsa" / f"p{pi}_oe_si_dsa{suffix}_eps_{epsilon:.0e}.json"
            data = read(path); grid = data["grid"]
            dof = int(grid[0] * grid[1] if pi <= 2 else grid[0] * grid[1] * 4 * grid[2])
            rows.append(dict(problem=problem, epsilon=epsilon, method="OE-SI-DSA",
                             time=data["runtime_seconds"], time_min=data["runtime_seconds"], time_max=data["runtime_seconds"], resource=dof,
                             resource_type="离散自由度", sample="确定性"))

            if problem == "P1":
                data = read(RESULTS / "baselines/rfm" / f"p1_rfm_eps_{epsilon:.0e}_seed_11.json")
                rows.append(dict(problem=problem, epsilon=epsilon, method="RFM",
                                 time=data["total_seconds"], time_min=data["total_seconds"], time_max=data["total_seconds"],
                                 resource=int(data["num_columns"]), resource_type="线性未知数", sample="seed 11"))
                data = read(RESULTS / "baselines/si_dsa" / f"p1_si_dsa_eps_{epsilon:.0e}.json")
                grid = data["grid"]
                rows.append(dict(problem=problem, epsilon=epsilon, method="SI-DSA",
                                 time=data["runtime_seconds"], time_min=data["runtime_seconds"], time_max=data["runtime_seconds"],
                                 resource=int(grid[0] * grid[1]), resource_type="离散自由度", sample="确定性"))
    return rows


def grouped(rows, problems, field, ylabel, filename):
    methods = [m for m in COLORS if any(r["method"] == m and r["problem"] in problems for r in rows)]
    categories = [(p, e) for p in problems for e in (1.0, 1e-3)]
    x = np.arange(len(categories)); width = min(0.13, 0.78 / len(methods))
    fig, ax = plt.subplots(figsize=(8.2 if len(problems) == 2 else 10.2, 5.2), constrained_layout=True)
    for index, method in enumerate(methods):
        values = []
        for problem, epsilon in categories:
            hit = next((r for r in rows if r["problem"] == problem and r["epsilon"] == epsilon and r["method"] == method), None)
            values.append(np.nan if hit is None else hit[field])
        pos = x + (index - (len(methods)-1)/2) * width
        ax.bar(pos, values, width, color=COLORS[method], edgecolor="white", linewidth=.55, label=method)
    labels = [f"{p}\n" + (r"$1$" if e == 1 else r"$10^{-3}$") for p, e in categories]
    ax.set_xticks(x, labels); ax.set_xlabel(r"Problem and $\varepsilon$"); ax.set_ylabel(ylabel)
    ax.set_yscale("log"); ax.grid(axis="y", which="both", linestyle="--", alpha=.28)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.02))
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    fig.savefig(OUT / filename, dpi=360, bbox_inches="tight", pad_inches=.08); plt.close(fig)


def main():
    rows = collect()
    grouped(rows, ("P1", "P2"), "time", "Time (s)", "cost_time_1d.png")
    grouped(rows, ("P3", "P4", "P5"), "time", "Time (s)", "cost_time_2d.png")
    grouped(rows, ("P1", "P2"), "resource", "Parameters / unknowns", "cost_size_1d.png")
    grouped(rows, ("P3", "P4", "P5"), "resource", "Parameters / unknowns", "cost_size_2d.png")
    table = RESULTS / "tables/baseline_cost_parameters.md"
    lines = ["# 基准方法时间与规模对比", "",
             "| 问题 | ε | 方法 | 统计口径 | 时间中位数或确定值/s | 时间范围/s | 数量 | 数量口径 |",
             "|---|---:|---|---|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['problem']} | {r['epsilon']:.0e} | {r['method']} | {r['sample']} | "
                     f"{r['time']:.3e} | [{r['time_min']:.3e}, {r['time_max']:.3e}] | {r['resource']} | {r['resource_type']} |")
    lines += ["", "注：OE-APNN 为 GPU 训练时间；随机特征方法为特征生成、组装、求解和评估总时间；迭代方法为求解时间。",
              "", "注：数量列按方法分别表示网络可训练参数、线性系统未知数或离散自由度，图中仅比较数量级。",
              "", "注：OE-APNN 方孔版 P5 与其他方法的异质介质 P5 不同，未放入成本横向比较。"]
    table.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
