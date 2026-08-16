# 实验数据口径

| 项目 | 统一口径 |
|---|---|
| OE-APRFM 计时 | `total_seconds=feature+assembly+solve+evaluation`；不含参考解生成和绘图。跨种子计时只采用同一代码版本、同一批次生成的数据。旧 P3 seed 11 的 665.08 s 中 assembly 为 649.28 s；新批次 seed 7/17 为 139.73–140.33 s，其中 assembly 为 127.80–128.79 s，故旧单次结果不再与新批次混合统计计时。 |
| MM-APRFM 随机性 | 旧 vendored 特征模块使用固定模块级 seed，导致不同任务特征相同；旧三种子统计无效。运行器已显式设置 `seedX/seedXV` 并记录 `feature_matrix_sha256`，使用 seeds 7/11/17 重跑。 |
| OE-APNN P5 | 归档结果为中心方孔制造解；OE-APRFM、MM-APRFM 的 P5 为异质介质问题。OE-APNN P5 不进入同问题横向误差或成本比较。 |
| 未收敛 SI | 同时报告 `max_iterations`、实际 `iterations`、`tolerance`、最终迭代差或相对残差、停止准则和 `converged=false`。1D 准则为 $\|u^{k}-u^{k-1}\|_2<tol$；2D Krylov 准则为预条件相对 GMRES 残差小于 `tol`。 |
| P2 五种子表 | 内部随机特征稳健性实验；seeds 11/23/37/53/71，报告中位数及范围。 |
| P2 三种子表 | 外部方法公平比较；seeds 7/11/17，报告均值及样本标准差。不得与五种子中位数拼接为同一统计量。 |
| ε 扫描残差 | $R_\varepsilon^{1/2}=\|A_s c-b_s\|_2/\sqrt{N_{row}}$，$A_s,b_s$ 为实际求解所用的行归一化（及指定加权）系统。 |
