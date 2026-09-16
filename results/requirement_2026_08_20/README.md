# 按《OE-APRFM 数值实验图表需求报告》整理的结果

生成日期：2026-08-20。该目录只收录数学定义兼容的数据；旧 manufactured P5 数据已明确排除。

## 完成状态

- Table 2/3/4/5：已由现有 P1/P3 原始 JSON 重建，逐 seed 数据保留在 CSV。
- Figure 1/2：已重建；仓库中未找到附件所称的 ε≈1e-16 原始 JSON，因此没有臆造极限尺度点。
- 新 P5：物理定义已冻结为 ε-independent Gaussian source、vacuum inflow、heterogeneous disk/channel medium。ε=1 A/B reference 已完成，但 A/B discrepancy 为 Ef=7.1119e-2、Eρ=9.5688e-3，尚未达到高精度认证。
- P5 ε=1e-3/1e-6：当前无 diffusion preconditioner 的 parity-GMRES 停滞，未写入正式误差表。
- Accuracy-cost frontier：现有数据不是多预算 matched-accuracy sweep，故不生成误导性的 Pareto 图。
- 旧 P5、旧 P5 的 MM-APRFM/OE-APNN 与图像均不得用于新 P5 横向比较。

## 尚需计算

1. 为 P5 reference solver 加 diffusion synthetic acceleration/preconditioner 后，完成 ε=1e-3/1e-6 的 A/B refinement。
2. 基于认证 fine reference 重跑 OE-APRFM、MM-APRFM、OE-APNN 三个 seeds。
3. 三方法各做至少三个预算点和三次 timing repeats，才能生成 Table 6 / Figure 4 / runtime breakdown。
