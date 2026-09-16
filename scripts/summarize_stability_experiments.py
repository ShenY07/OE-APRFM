"""Create publication tables and separate, title-free stability figures."""

from __future__ import annotations

import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results/tables"
FIGURES = ROOT / "results/figures/stability"


def records(directory: str):
    return [json.loads(path.read_text()) for path in sorted((ROOT / directory).glob("*.json"))]


def fmt(value):
    return "—" if value is None or not np.isfinite(value) else f"{value:.4e}"


def grouped(records_, keys):
    result = {}
    for record in records_:
        result.setdefault(tuple(record[key] for key in keys), []).append(record)
    return result


def mean_std(rows, key):
    values = np.asarray([row[key] for row in rows], dtype=float)
    return values.mean(), values.std(ddof=1) if values.size > 1 else 0.0


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def save_metric(scan, key, ylabel, filename):
    fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    for problem, marker in (("p1", "o"), ("p2", "s"), ("p3", "^")):
        rows = sorted((r for r in scan if r["problem"] == problem), key=lambda r: r["epsilon"], reverse=True)
        if rows:
            ax.loglog([r["epsilon"] for r in rows], [r[key] for r in rows], marker=marker, linewidth=1.6, markersize=5, label=problem.upper())
    ax.set_xlabel(r"$\varepsilon$"); ax.set_ylabel(ylabel); ax.grid(True, which="both", alpha=.22); ax.legend(frameon=False)
    fig.savefig(FIGURES / filename, dpi=300); plt.close(fig)


def main():
    TABLES.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    scan = records("results/epsilon_scan")
    lines = ["# 固定离散参数的 ε 扫描", "", "三种子统计采用均值±样本标准差。残差口径：$R_\\varepsilon^{1/2}=\\|A_s c-b_s\\|_2/\\sqrt{N_{row}}$。", "", "| 问题 | ε | seeds | Ef mean±std | Eρ mean±std | Rε^(1/2) mean±std | Cemp mean±std | κ(A) median | rank/Ncoef |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    scan_csv = []
    for (problem, epsilon), rows in sorted(grouped(scan, ("problem", "epsilon")).items(), key=lambda x:(x[0][0],-x[0][1])):
        ef,efs=mean_std(rows,"relative_l2_f"); er,ers=mean_std(rows,"relative_l2_rho"); rr,rrs=mean_std(rows,"residual_half"); ce,ces=mean_std(rows,"empirical_stability_ratio"); cond=np.median([r["condition_number"] for r in rows]); rank=np.median([r["rank"] for r in rows]); ncoef=rows[0]["num_columns"]
        lines.append(f"| {problem.upper()} | {epsilon:.0e} | {len(rows)} | {ef:.3e}±{efs:.2e} | {er:.3e}±{ers:.2e} | {rr:.3e}±{rrs:.2e} | {ce:.3e}±{ces:.2e} | {cond:.3e} | {rank:.0f}/{ncoef} |")
        scan_csv.append(dict(problem=problem,epsilon=epsilon,seeds=len(rows),ef_mean=ef,ef_std=efs,erho_mean=er,erho_std=ers,residual_mean=rr,residual_std=rrs,cemp_mean=ce,cemp_std=ces,condition_median=cond,rank_median=rank,num_columns=ncoef))
    (TABLES / "epsilon_scan.md").write_text("\n".join(lines) + "\n")
    write_csv(TABLES / "epsilon_scan.csv", scan_csv)

    conv = records("results/feature_convergence")
    conv_groups = grouped(conv, ("problem", "epsilon", "features_per_patch"))
    lines = ["# 最小随机特征收敛实验", "", "| 问题 | ε | J | seeds | Ef mean±std | Eρ mean±std | κ(A) median [min,max] | rank median [min,max] | Nrow/Ncoef |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    conv_csv=[]
    for (p,e,j), rs in sorted(conv_groups.items(), key=lambda z:(z[0][0],-z[0][1],z[0][2])):
        ef=np.array([r["relative_l2_f"] for r in rs]); er=np.array([r["relative_l2_rho"] for r in rs]); co=np.array([r["condition_number"] for r in rs]); ra=np.array([r["rank"] for r in rs]); ov=np.array([r["oversampling_ratio"] for r in rs])
        lines.append(f"| {p.upper()} | {e:.0e} | {j} | {len(rs)} | {ef.mean():.3e}±{ef.std(ddof=1) if len(rs)>1 else 0:.2e} | {er.mean():.3e}±{er.std(ddof=1) if len(rs)>1 else 0:.2e} | {np.median(co):.3e} [{co.min():.2e},{co.max():.2e}] | {np.median(ra):.0f} [{ra.min()},{ra.max()}] | {np.median(ov):.2f} |")
        conv_csv.append(dict(problem=p,epsilon=e,features=j,seeds=len(rs),ef_mean=ef.mean(),ef_std=ef.std(ddof=1) if len(rs)>1 else 0,erho_mean=er.mean(),erho_std=er.std(ddof=1) if len(rs)>1 else 0,condition_median=np.median(co),rank_median=np.median(ra),oversampling_median=np.median(ov)))
    (TABLES / "feature_convergence.md").write_text("\n".join(lines) + "\n")
    write_csv(TABLES / "feature_convergence.csv", conv_csv)

    collocation = records("results/collocation_sufficiency")
    collocation_groups = grouped(collocation, ("problem", "epsilon", "oversampling_ratio"))
    lines = ["# 配点充分性实验", "", "| 问题 | ε | 实际 Nrow/Ncoef | seeds | Ef mean±std | Eρ mean±std | κ(A) median | rank median |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    collocation_csv=[]
    for (p,e,ratio),rs in sorted(collocation_groups.items()):
        ef,efs=mean_std(rs,"relative_l2_f"); er,ers=mean_std(rs,"relative_l2_rho"); cond=np.median([r["condition_number"] for r in rs]); rank=np.median([r["rank"] for r in rs])
        lines.append(f"| {p.upper()} | {e:.0e} | {ratio:.2f} | {len(rs)} | {ef:.3e}±{efs:.2e} | {er:.3e}±{ers:.2e} | {cond:.3e} | {rank:.0f} |")
        collocation_csv.append(dict(problem=p,epsilon=e,oversampling_ratio=ratio,seeds=len(rs),ef_mean=ef,ef_std=efs,erho_mean=er,erho_std=ers,condition_median=cond,rank_median=rank))
    (TABLES / "collocation_sufficiency.md").write_text("\n".join(lines)+"\n")
    write_csv(TABLES / "collocation_sufficiency.csv",collocation_csv)

    p5 = records("results/main_final")
    lines = ["# P5 三随机种子主结果", "", "固定配置：2×2×1 分区、每分区 96 个特征、12×12×12 配点，实际 Nrow/Ncoef=7.75。", "", "| ε | seeds | Ef mean±std | Eρ mean±std | Rε^(1/2) mean±std | Cemp mean±std | κ(A) median [min,max] | rank/Ncoef | time (s) mean±std |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    p5_csv=[]
    for (epsilon,),rs in sorted(grouped(p5,("epsilon",)).items(), reverse=True):
        ef,efs=mean_std(rs,"relative_l2_f"); er,ers=mean_std(rs,"relative_l2_rho"); rr,rrs=mean_std(rs,"residual_half"); ce,ces=mean_std(rs,"empirical_stability_ratio"); tm,tms=mean_std(rs,"total_seconds")
        co=np.asarray([r["condition_number"] for r in rs]); rank=np.median([r["rank"] for r in rs]); ncoef=rs[0]["num_columns"]
        lines.append(f"| {epsilon:.0e} | {len(rs)} | {ef:.3e}±{efs:.2e} | {er:.3e}±{ers:.2e} | {rr:.3e}±{rrs:.2e} | {ce:.3e}±{ces:.2e} | {np.median(co):.3e} [{co.min():.2e},{co.max():.2e}] | {rank:.0f}/{ncoef} | {tm:.1f}±{tms:.1f} |")
        p5_csv.append(dict(epsilon=epsilon,seeds=len(rs),ef_mean=ef,ef_std=efs,erho_mean=er,erho_std=ers,residual_mean=rr,residual_std=rrs,cemp_mean=ce,cemp_std=ces,condition_median=np.median(co),condition_min=co.min(),condition_max=co.max(),rank_median=rank,num_columns=ncoef,time_mean=tm,time_std=tms))
    (TABLES / "p5_main_3seed.md").write_text("\n".join(lines)+"\n")
    write_csv(TABLES / "p5_main_3seed.csv",p5_csv)

    ablation = records("results/ablation")
    ablation += [r for r in records("results/baselines/rfm") if r.get("problem")=="p1" and r.get("seed")==11]
    ablation += [r for r in records("results/baselines/mm_aprfm") if r.get("problem")=="p1" and r.get("seed")==11]
    labels={"full":"OE-APRFM (parity features)","full_angular":"OE-APRFM (unconstrained angular features)","oe_original":"OE original equations","rfm":"Direct RFM","mm_aprfm":"MM-APRFM"}
    lines=["# P1 结构消融（seed=11）", "", "所有方法均使用 128 个系数。条件数为求解器列均衡后的有效条件数；不同方程结构的行数和过采样比列在表中，故误差对比应结合离散成本解释。", "", "| 方法/变体 | ε | Ef | Eρ | Rε^(1/2) | κ(A) | rank/Ncoef | Nrow/Ncoef | time (s) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    ablation_csv=[]
    for r in sorted(ablation,key=lambda x:(labels.get(x.get("variant") or x["method"],x["method"]),-x["epsilon"])):
        key=r.get("variant") or r["method"]; label=labels.get(key,key); residual=r.get("residual_half")
        lines.append(f"| {label} | {r['epsilon']:.0e} | {fmt(r['relative_l2_f'])} | {fmt(r['relative_l2_rho'])} | {fmt(residual)} | {fmt(r['condition_number'])} | {r['rank']}/{r['num_columns']} | {r['oversampling_ratio']:.2f} | {r['total_seconds']:.2f} |")
        ablation_csv.append(dict(label=label,method=r["method"],variant=r.get("variant"),epsilon=r["epsilon"],seed=r["seed"],ef=r["relative_l2_f"],erho=r["relative_l2_rho"],residual=residual,condition=r["condition_number"],rank=r["rank"],num_columns=r["num_columns"],num_rows=r["num_rows"],oversampling_ratio=r["oversampling_ratio"],time_seconds=r["total_seconds"]))
    (TABLES / "p1_structure_ablation.md").write_text("\n".join(lines)+"\n")
    write_csv(TABLES / "p1_structure_ablation.csv",ablation_csv)

    save_metric(scan,"relative_l2_f",r"$E_f$","eps_ef.png")
    save_metric(scan,"relative_l2_rho",r"$E_\rho$","eps_erho.png")
    save_metric(scan,"condition_number",r"$\kappa(A)$","eps_kappa.png")
    save_metric(scan,"residual_half",r"$R_\varepsilon^{1/2}$","eps_res.png")
    save_metric(scan,"empirical_stability_ratio",r"$C_{emp}$","eps_cemp.png")


if __name__ == "__main__": main()
