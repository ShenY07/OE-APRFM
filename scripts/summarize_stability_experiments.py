"""Create publication tables and separate, title-free stability figures."""

from __future__ import annotations

import json
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
    lines = ["# 固定离散参数的 ε 扫描", "", "残差口径：$R_\\varepsilon^{1/2}=\\|A_s c-b_s\\|_2/\\sqrt{N_{row}}$，其中 $A_s,b_s$ 是实际求解所用的行归一化（及指定加权）系统。", "", "| 问题 | ε | Ef | Eρ | κ(A) | Rε^(1/2) | Cemp | rank/Ncoef | Nrow/Ncoef |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(scan, key=lambda x: (x["problem"], -x["epsilon"])):
        lines.append(f"| {r['problem'].upper()} | {r['epsilon']:.0e} | {fmt(r.get('relative_l2_f'))} | {fmt(r.get('relative_l2_rho'))} | {fmt(r.get('condition_number'))} | {fmt(r.get('residual_half'))} | {fmt(r.get('empirical_stability_ratio'))} | {r.get('rank','—')}/{r.get('num_columns','—')} | {r.get('oversampling_ratio', float('nan')):.2f} |")
    (TABLES / "epsilon_scan.md").write_text("\n".join(lines) + "\n")

    conv = records("results/feature_convergence")
    grouped = {}
    for r in conv: grouped.setdefault((r["problem"], r["epsilon"], r["features_per_patch"]), []).append(r)
    lines = ["# 最小随机特征收敛实验", "", "| 问题 | ε | J | seeds | Ef mean±std | Eρ mean±std | κ(A) median [min,max] | rank median [min,max] | Nrow/Ncoef |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (p,e,j), rs in sorted(grouped.items(), key=lambda z:(z[0][0],-z[0][1],z[0][2])):
        ef=np.array([r["relative_l2_f"] for r in rs]); er=np.array([r["relative_l2_rho"] for r in rs]); co=np.array([r["condition_number"] for r in rs]); ra=np.array([r["rank"] for r in rs]); ov=np.array([r["oversampling_ratio"] for r in rs])
        lines.append(f"| {p.upper()} | {e:.0e} | {j} | {len(rs)} | {ef.mean():.3e}±{ef.std(ddof=1) if len(rs)>1 else 0:.2e} | {er.mean():.3e}±{er.std(ddof=1) if len(rs)>1 else 0:.2e} | {np.median(co):.3e} [{co.min():.2e},{co.max():.2e}] | {np.median(ra):.0f} [{ra.min()},{ra.max()}] | {np.median(ov):.2f} |")
    (TABLES / "feature_convergence.md").write_text("\n".join(lines) + "\n")

    save_metric(scan,"relative_l2_f",r"$E_f$","eps_ef.png")
    save_metric(scan,"relative_l2_rho",r"$E_\rho$","eps_erho.png")
    save_metric(scan,"condition_number",r"$\kappa(A)$","eps_kappa.png")
    save_metric(scan,"residual_half",r"$R_\varepsilon^{1/2}$","eps_res.png")
    save_metric(scan,"empirical_stability_ratio",r"$C_{emp}$","eps_cemp.png")


if __name__ == "__main__": main()
