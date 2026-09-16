# E1–E5 执行与交付索引

> 2026-09-10最新状态：修复队列19/19项完成；P5三阈值同版本重算、MM测试加密均完成。统一入口为 [E1–E6当前汇总](experiments_current_summary_2026_09_10.md)。以下早期“运行中/基线缺失”叙述为历史，科学验收剩余项以新汇总为准。

最新填写汇总：见 [E1–E5 已填写表格](/mnt/store2/sheny-y23/OE-APRFM/docs/e1_e5_filled_tables_2026_09_09.md)。E5 已完成24/24组，P5相邻截断两组已完成；下方较早的运行中状态仅为历史记录。参考、求积、计时及P5同版本基线缺口按新表列示，不视为全部验收通过。

本轮新计算 E3/E5 使用对称 U(-1,1) 初始化。E1 固定空间加密与 E2/E4 旧档案不改变其原始初始化。所有预设结果均保留，不按 OE/MM 胜负筛选。

## E1：已完成

`results/minimum_gain/p1_J16_seed11_refined128.json` 保存独立内外求积加密、32/32 数值秩及无截断最小增益；`quadrature_check.csv`、`beta_epsilon.pdf` 为交付表图。这只是固定空间及离散求积诊断，不是连续稳定常数认证。

## E2：复用数据与图表索引

下表相对 `results/requirement_2026_08_20/`；不重跑已有训练。

| 内容 | 数值来源 | 已有图 |
|---|---|---|
| P1/P3 跨 epsilon 精度 | `tables_frozen/table_2_accuracy_knudsen.csv` | `figures_separate/fig01_1d_manufactured_error_vs_knudsen.pdf`、`fig01_2d_manufactured_error_vs_knudsen.pdf` |
| P1 parity 特征预算 | `tables_frozen/table_S4_parity_feature_budget.csv` | `figures_separate/fig03_parity_feature_budget_kinetic_error_vs_ncoef.pdf` |
| projection/rescaling 消融 | 冻结消融档案；图的来源由 `scripts/plot_publication_separate.py` 追踪 | `figures_separate/fig03_projection_rescaling_formulation.pdf` |
| 固定行数特征扫描 | `tables_frozen/table_3_feature_resolution_full.csv`、`feature_resolution_seedwise.csv` | `figures_separate/fig02_1d_manufactured_error_vs_features.pdf`、`fig02_2d_manufactured_error_vs_features.pdf` |
| P1 固定特征配点扫描 | `tables_frozen/table_3_collocation_refinement.csv` 的 p1 行；逐 seed 为 `results/collocation_sufficiency/p1_*.json` | 横轴必须为实际 Nrow/Ncoef；不混入旧 P3 两分量 |
| 精度—成本 | `tables_frozen/table_S3_accuracy_cost_sweep.csv`、`table_S2b_efficiency_seedwise.csv` | `figures_separate/fig04_1d_manufactured_accuracy_cost.pdf`、`fig04_2d_manufactured_accuracy_cost.pdf` |

P1 固定特征扫描为 2944 行；P3 应筛选四分量、13808 行。制造源项依赖 epsilon，不能替代 E3。上述为复用索引，逐项最终配置/计时审计仍需完成，不将存在文件等同于全部验收。

## E3：18 组已完成

固定单分区、每场 64 特征、128 系数、2944 行、rcond=1e-12。三 seed 扫描六个 epsilon；独立 64/128 阶评价求积，流直接计算 `<v j>`。epsilon=1、0.1 的两级输运参考均迭代收敛，六组 f/rho 误差通过参考差异小于方法误差 10% 的分辨率检查。

结果目录：`results/e3_slab/`。交付 `summary.csv`、`diffusion_limit.pdf`、`profiles.pdf`；逐 seed JSON 保存输运参考和求积检查。

## E4：保留冻结 P5，不重新调参

物理域为 (0,1)^2；sigma_s=1+y，sigma_a=0.1，Q=0；左右入流 1+0.2y，下侧 1，上侧 1.2。最终实际配置：四分量，分区 1×1×2，每分量每分区 J=128，scale=0.25，1024 系数、58096 行、角平均16点、rcond=1e-6、block_weights=[10,1,1,1]。不得以配置文件默认值覆盖这些运行参数。

逐 seed 误差表：`results/raw/p5_boundary_formal/p5_frozen_three_seed_level_C.csv`。

| epsilon | median Ef | median Erho |
|---|---:|---:|
| 1 | 0.0423426 | 0.00476428 |
| 0.001 | 0.0362395 | 0.00460222 |

已有 `results/requirement_2026_08_20/figures_separate/fig07_p5_*.pdf` 为 epsilon=0.001、seed 11 的系数场、level C 参考密度、数值密度和绝对密度误差；参考/数值共用色标，见同目录 MANIFEST.md。入流残差与参考加密逐项汇总尚待最终审计。

## E5：执行中，未验收

`scripts/run_e5_mixed.py` 执行 2 方法×3 seed×4 角预算。全局 epsilon=1，sigma_s=1/kappa，1152 系数，角平均64点，rcond=1e-6。空间内部126点（128点剔除两端），OE 正角半格点与 MM 的镜像正负点匹配；每侧128入流点。两法均行归一化、列均衡、gelsd 相对截断。不同残差形式的实际行数如实报告，不能称行数相同。

`scripts/report_e5_mixed.py` 从保存的完整 f 重算 Gauss 加权 Ef、Erho 和物理流误差 EF，避免旧 MM 报告的梯形角积分口径。主方法成本为 feature+assembly+solve，不含独立评价及参考生成。`results/e5_mixed/physical_metrics.csv`、`summary.csv`、`angular_budget.pdf` 可随作业完成刷新，未满24组及未通过角平均加密检查前均属阶段结果。
