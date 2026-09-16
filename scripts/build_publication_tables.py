"""Build the eight canonical manuscript data tables from retained raw JSON."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oeraprfm-mpl")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures" / "publication"


def load(pattern: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(ROOT.glob(pattern))]


def sci(value) -> str:
    return "—" if value is None else f"{float(value):.8e}"


def csv_table(number: int, rows: list[dict]) -> None:
    path = OUT / f"table_{number}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def md_table(lines: list[str], rows: list[dict], columns: list[tuple[str, str]]) -> None:
    lines.append("| " + " | ".join(label for _, label in columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key, _ in columns) + " |")
    lines.append("")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OE-APRFM 投稿数据表",
        "",
        "数据口径：所有逐 seed 表均只采用 seeds 11、23、37；κeff=σmax/σmin_eff，reff 为最终 SVD 阈值下的有效秩。表中“—”表示原始结果未保存该字段，不作推断。",
        "",
    ]

    # Table 1 follows the actual retained formal result sets, not configuration defaults.
    t1 = [
        dict(problem="P1 manufactured 1D", partition="1×1", M=1, J=64, Ncoef=128, Nrow=2944, ratio="23.0000", angular_order=8, scale="1.0", rcond="1e-12"),
        dict(problem="P2 heterogeneous 1D", partition="2×4", M=8, J=64, Ncoef=1024, Nrow=2944, ratio="2.8750", angular_order=8, scale="1.0", rcond="1e-6"),
        dict(problem="P3 smooth manufactured 2D (four component)", partition="1×1×1", M=1, J=128, Ncoef=512, Nrow=15856, ratio="30.9688", angular_order=32, scale="1.0", rcond="1e-12"),
        dict(problem="P4 perforated domain (four component)", partition="1×1×1", M=1, J=128, Ncoef=512, Nrow=14736, ratio="28.7812", angular_order=32, scale="1.0", rcond="1e-12"),
        dict(problem="P5 heterogeneous medium", partition="2×2×1", M=4, J=96, Ncoef=768, Nrow=5952, ratio="7.7500", angular_order=32, scale="1.0", rcond="1e-12"),
    ]
    csv_table(1, t1)
    lines += ["## 表 1：基础离散参数", "", "J 为每个子区域、每个奇偶未知分量的随机特征数；Ncoef 为实际线性系统列数。P3/P4 使用四分量 (j1,r1,j2,r2) 和第一象限角采样，每分量 J=128，总列数仍为 512。", ""]
    md_table(lines, t1, [("problem","算例"),("partition","domain partition"),("M","M"),("J","J"),("Ncoef","Ncoef"),("Nrow","Nrow"),("ratio","Nrow/Ncoef"),("angular_order","angular order"),("scale","feature scale"),("rcond","SVD rcond")])

    scan = [r for r in load("results/epsilon_scan/*.json") if r["seed"] in (11,23,37)]
    t2=[]
    for r in scan:
        if r["problem"] in ("p1","p2"):
            t2.append(dict(problem=r["problem"].upper(),epsilon=f"{r['epsilon']:.0e}",seed=r["seed"],Ef=sci(r["relative_l2_f"]),Erho=sci(r["relative_l2_rho"]),Rhalf=sci(r["residual_half"])))
    csv_table(2,t2)
    lines += ["## 表 2：一维误差原始数据", "", "P2 仅列参考解通过验证的 ε=1、1e-3；ε=1e-6 不填入失败数据。", ""]
    md_table(lines,t2,[("problem","算例"),("epsilon","ε"),("seed","seed"),("Ef","Ef"),("Erho","Eρ"),("Rhalf","Rε^(1/2)")])

    t3=[]
    for r in scan:
        if r["problem"] in ("p1","p3"):
            t3.append(dict(problem=r["problem"].upper(),epsilon=f"{r['epsilon']:.0e}",seed=r["seed"],Ef=sci(r["relative_l2_f"]),Erho=sci(r["relative_l2_rho"]),Cemp=sci(r["empirical_stability_ratio"]),sigma_max=sci(r["largest_singular_value"]),sigma_min_eff=sci(r["smallest_effective_singular_value"]),r_eff=r["rank"],Ncoef=r["num_columns"],kappa_eff=sci(r["condition_number"])))
    csv_table(3,t3)
    lines += ["## 表 3：ε-一致性原始数据", ""]
    md_table(lines,t3,[("problem","算例"),("epsilon","ε"),("seed","seed"),("Ef","Ef"),("Erho","Eρ"),("Cemp","Cemp"),("sigma_max","σmax"),("sigma_min_eff","σmin_eff"),("r_eff","reff"),("Ncoef","Ncoef"),("kappa_eff","κeff")])

    p34_four = [r for r in load("results/p34_representation_comparison/*.json") if r["seed"] in (11,23,37) and r.get("angular_representation") == "four_component"]
    p5 = [r for r in load("results/main_final/p5_*.json") if r["seed"] in (11,23,37)]
    t4=[]
    for r in p34_four+p5:
        ncoef=r.get("num_columns",512)
        t4.append(dict(problem=r["problem"].upper(),epsilon=f"{r['epsilon']:.0e}",seed=r["seed"],Ef=sci(r["relative_l2_f"]),Erho=sci(r["relative_l2_rho"]),rank=f"{r['rank']}/{ncoef}"))
    csv_table(4,t4)
    lines += ["## 表 4：二维误差原始数据", "", "P3/P4 已更新为四分量第一象限表示，固定 16×16×16 配点、Ncoef=512 和 seeds 11/23/37。P5 原始结果已经保存 Eρ，并非缺失。", ""]
    md_table(lines,t4,[("problem","算例"),("epsilon","ε"),("seed","seed"),("Ef","Ef"),("Erho","Eρ"),("rank","reff/Ncoef")])

    conv=[r for r in load("results/feature_convergence/*.json") if r["seed"] in (11,23,37)]
    t5=[]
    for r in conv:
        t5.append(dict(problem=r["problem"].upper(),epsilon=f"{r['epsilon']:.0e}",J=r["features_per_patch"],seed=r["seed"],Ef=sci(r["relative_l2_f"]),Erho=sci(r["relative_l2_rho"]),r_eff=r["rank"],Ncoef=r["num_columns"],kappa_eff=sci(r["condition_number"])))
    csv_table(5,t5)
    lines += ["## 表 5：随机特征收敛数据", "", "完整 72 行原始数据见 `table_5.csv`；Markdown 正文不重复展开，建议直接用于 log-scale 曲线和代表值汇总。", ""]

    coll=[r for r in load("results/collocation_sufficiency/*.json") if r["seed"] in (11,23,37)]
    t6=[]
    for r in coll:
        target_ratio=min((2,4,8,12),key=lambda value:abs(value-r["oversampling_ratio"]))
        t6.append(dict(problem=r["problem"].upper(),target_ratio=target_ratio,actual_ratio=f"{r['oversampling_ratio']:.6f}",seed=r["seed"],Nrow=r["num_rows"],Ncoef=r["num_columns"],Ef=sci(r["relative_l2_f"]),Erho=sci(r["relative_l2_rho"]),r_eff=r["rank"],kappa_eff=sci(r["condition_number"])))
    csv_table(6,t6)
    lines += ["## 表 6：配点数实验", ""]
    md_table(lines,t6,[("problem","算例"),("target_ratio","目标 ratio"),("actual_ratio","实际 ratio"),("seed","seed"),("Nrow","Nrow"),("Ncoef","Ncoef"),("Ef","Ef"),("Erho","Eρ"),("r_eff","reff"),("kappa_eff","κeff")])

    abl=load("results/ablation/*.json")
    abl += [r for r in load("results/baselines/rfm/p1_*.json") if r["seed"]==11]
    abl += [r for r in load("results/baselines/mm_aprfm/p1_*.json") if r["seed"]==11]
    names={"rfm":"Direct RFM","mm_aprfm":"MM-APRFM","oe_original":"OE without projection/rescaling","full_angular":"unconstrained angular features","full":"full OE-APRFM"}
    t7=[]
    for r in abl:
        key=r.get("variant") or r["method"]
        t7.append(dict(method=names[key],epsilon=f"{r['epsilon']:.0e}",seed=r["seed"],Ef=sci(r["relative_l2_f"]),matrix_size=f"{r['num_rows']}×{r['num_columns']}",r_eff=r["rank"],kappa_eff=sci(r["condition_number"])))
    t7.sort(key=lambda x:(x["method"],-float(x["epsilon"])))
    csv_table(7,t7)
    lines += ["## 表 7：P1 结构消融", "", "这是当前最小消融范围（seed=11）。各方法 Ncoef 均为 128，但 Nrow 不完全相同，表中保留实际矩阵尺寸。", ""]
    md_table(lines,t7,[("method","方法"),("epsilon","ε"),("Ef","Ef"),("matrix_size","matrix size"),("r_eff","reff"),("kappa_eff","κeff")])

    parity=[r for r in abl if r.get("variant") in ("full","full_angular")]
    t8=[]
    for r in parity:
        representation="parity-reduced" if r["variant"]=="full" else "full/unconstrained angular"
        t8.append(dict(representation=representation,epsilon=f"{r['epsilon']:.0e}",seed=r["seed"],feature_budget=r["num_columns"],Ef=sci(r["relative_l2_f"]),sigma_max=sci(r["largest_singular_value"]),sigma_min_eff=sci(r["smallest_effective_singular_value"]),r_eff=r["rank"],kappa_eff=sci(r["condition_number"])))
    t8.sort(key=lambda x:(x["representation"],-float(x["epsilon"])))
    csv_table(8,t8)
    lines += ["## 表 8：P1 角向对称降维", "", "相同线性 feature budget（Ncoef=128）、相同 Nrow=2944、seed=11。这里的 full angular 指未施加奇偶约束的角特征表示。", ""]
    md_table(lines,t8,[("representation","角向表示"),("epsilon","ε"),("feature_budget","feature budget"),("Ef","Ef"),("sigma_max","σmax"),("sigma_min_eff","σmin_eff"),("r_eff","reff"),("kappa_eff","κeff")])

    lines += [
        "## P5 heterogeneous medium 的实际数学定义", "",
        "当前正式 P5 是二维非制造问题：`sigma_s=1+x2`，`sigma_a=0`，`Q=1+0.5 sin(pi x1) sin(pi x2)`，真空入流。", "",
        "两个 Knudsen 数共用 2×2×1 分区、每分量每空间 patch 64 features、32×32×16 均匀配点与相同 `rcond`。只有独立 OE-SN 参考解通过加密判据后的三种子中位数才能进入表格。", "",
        "## 效率数据处理决定", "", "暂不生成效率比较表。现有不同方法数据主要是单预算 raw time，尚未形成匹配精度或匹配预算的 Pareto 点；直接比较会造成过度解释。", "",
    ]

    refinement_specs = [
        (1.0, "C", "D", "results/references_notebook/p2_parity_ref_eps_1e+00_refinement_DC.json", "relative_difference_f_DC", "relative_difference_rho_DC"),
        (1.0e-3, "A", "B", "results/references_notebook/p2_parity_ref_eps_1e-03_refinement.json", "relative_difference_f_BA", "relative_difference_rho_BA"),
    ]
    t9=[]
    for epsilon, coarse, fine, comparison_path, ef_key, erho_key in refinement_specs:
        stem=f"results/references_notebook/p2_parity_ref_eps_{epsilon:.0e}_level_"
        coarse_meta=json.loads((ROOT/f"{stem}{coarse}.json").read_text())
        fine_meta=json.loads((ROOT/f"{stem}{fine}.json").read_text())
        comparison=json.loads((ROOT/comparison_path).read_text())
        t9.append(dict(
            epsilon=f"{epsilon:.0e}",
            coarse_grid="×".join(map(str,coarse_meta["grid"])),
            fine_grid="×".join(map(str,fine_meta["grid"])),
            E_f_ref=sci(comparison[ef_key]),
            E_rho_ref=sci(comparison[erho_key]),
            coarse_iterations=coarse_meta["iterations"],
            fine_iterations=fine_meta["iterations"],
            coarse_final_difference=sci(coarse_meta["final_difference"]),
            fine_final_difference=sci(fine_meta["final_difference"]),
            tolerance=sci(fine_meta["tolerance"]),
            converged=f"{coarse_meta['converged']}/{fine_meta['converged']}",
        ))
    csv_table(9,t9)
    lines += ["## 表 9：P2 数值参考解 refinement certification", "", "参考解由 `src/numerical/parity_si_1d.ipynb` 中的 parity-SI+DSA 生成。Ef_ref 与 Eρ_ref 是相邻 coarse/fine 参考解的相对差异，不是 OE-APRFM 误差。", ""]
    md_table(lines,t9,[("epsilon","ε"),("coarse_grid","coarse grid"),("fine_grid","fine grid"),("E_f_ref","Ef_ref"),("E_rho_ref","Eρ_ref"),("coarse_iterations","coarse iter."),("fine_iterations","fine iter."),("coarse_final_difference","coarse final diff."),("fine_final_difference","fine final diff."),("tolerance","tol."),("converged","converged coarse/fine")])
    lines += ["ε=1e-6 使用原 notebook 的 DSA relaxation=1e-2 时发生数值溢出，减小 relaxation 后仍未在合理迭代预算内达到 stopping/refinement criterion；因此不认证参考解，也不报告该尺度的 OE-APRFM error。", ""]

    # Compact feature-resolution summary: best J minimizes the three-seed median Ef.
    t10=[]
    for problem in ("P1","P3"):
        for epsilon in (1.0,1.0e-3,1.0e-6):
            subset=[r for r in t5 if r["problem"]==problem and float(r["epsilon"])==epsilon]
            by_j={j:[r for r in subset if int(r["J"])==j] for j in (32,64,128,256)}
            medians={j:float(np.median([float(r["Ef"]) for r in rows])) for j,rows in by_j.items()}
            best_j=min(medians,key=medians.get)
            loss=[j for j,rows in by_j.items() if np.median([int(r["r_eff"])/int(r["Ncoef"]) for r in rows])<1.0]
            t10.append(dict(problem=problem,epsilon=f"{epsilon:.0e}",best_J=best_j,median_Ef_at_best_J=sci(medians[best_j]),rank_loss_starts_at=min(loss) if loss else "none through J=256"))
    csv_table(10,t10)
    lines += ["## 表 10：随机特征收敛代表值", "", "best J 按三个种子 Ef 的中位数最小确定；rank loss starts at 按三个种子中位 reff/Ncoef<1 的最小 J 确定。完整 72 行仍保存在 `table_5.csv`。", ""]
    md_table(lines,t10,[("problem","Problem"),("epsilon","ε"),("best_J","best J"),("median_Ef_at_best_J","median Ef at best J"),("rank_loss_starts_at","rank loss starts at")])
    lines += [
        "## Diffusion-limit verification 状态", "",
        "当前数据只支持固定离散参数下的 ε-uniform accuracy/robustness 结论，不能作为真正的 diffusion-limit verification。P1/P3 的制造 source 含 1/ε；仓库中也没有一组同时保存 kinetic solution、明确 diffusion-limit PDE solution、边界层处理和 ε→0 差异的数据。因此本次没有生成 diffusion-limit table。若论文标题或核心贡献明确强调 asymptotic-preserving，此表仍是高优先级缺失实验。", "",
    ]

    # Two compact main-text candidates from the complete table_5 data.
    colors={1.0:"#3B6FB6",1.0e-3:"#E07A5F",1.0e-6:"#4C956C"}
    fig,axes=plt.subplots(2,2,figsize=(8.2,6.2),sharex=True,constrained_layout=True)
    for col,problem in enumerate(("P1","P3")):
        for epsilon in (1.0,1.0e-3,1.0e-6):
            rows=[r for r in t5 if r["problem"]==problem and float(r["epsilon"])==epsilon]
            js=np.array((32,64,128,256))
            for row,key,label in ((0,"Ef",r"$E_f$"),(1,"Erho",r"$E_\rho$")):
                values=np.array([[float(r[key]) for r in rows if int(r["J"])==j] for j in js])
                center=np.median(values,axis=1); low=values.min(axis=1); high=values.max(axis=1)
                axes[row,col].loglog(js,center,"o-",color=colors[epsilon],label=fr"$\varepsilon={epsilon:.0e}$")
                axes[row,col].fill_between(js,low,high,color=colors[epsilon],alpha=.15)
                axes[row,col].set_ylabel(label); axes[row,col].grid(True,which="both",alpha=.22)
        axes[0,col].set_title(problem); axes[1,col].set_xlabel("J")
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.savefig(FIGURES/"feature_errors_vs_J.png",dpi=320); plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(8.2,3.2),constrained_layout=True)
    for col,problem in enumerate(("P1","P3")):
        axis2=axes[col].twinx()
        for epsilon in (1.0,1.0e-3,1.0e-6):
            rows=[r for r in t5 if r["problem"]==problem and float(r["epsilon"])==epsilon]
            js=np.array((32,64,128,256))
            ranks=np.array([np.median([int(r["r_eff"])/int(r["Ncoef"]) for r in rows if int(r["J"])==j]) for j in js])
            kappas=np.array([np.median([float(r["kappa_eff"]) for r in rows if int(r["J"])==j]) for j in js])
            axes[col].semilogx(js,ranks,"o-",color=colors[epsilon],label=fr"$\varepsilon={epsilon:.0e}$")
            axis2.loglog(js,kappas,"--",color=colors[epsilon],alpha=.65)
        axes[col].set(xlabel="J",ylabel=r"median $r_{eff}/N_{coef}$",title=problem); axes[col].set_ylim(0,1.05); axes[col].grid(True,which="both",alpha=.22); axis2.set_ylabel(r"median $\kappa_{eff}$")
    axes[0].legend(frameon=False,fontsize=8)
    fig.savefig(FIGURES/"feature_rank_condition_vs_J.png",dpi=320); plt.close(fig)

    (OUT / "publication_tables.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
