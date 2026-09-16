# P5 冻结设计：二维平滑非制造输运问题

P5 的唯一角色是检验 OE-APRFM 在无解析解时，能否处理二维平滑变系数介质与
全域内源，并在 kinetic/diffusive 两种尺度保持相近精度。它不承担尖界面、局部源、
复杂几何、自适应采样、方法横向比较或收敛性消融的论证任务。

## 数学问题

- 空间域：`D=(0,1)^2`；角变量：`v in S^1`。
- 散射：`sigma_s(x1,x2)=1+x2`，范围 `[1,2]`。
- 吸收：`sigma_a=0`。
- 内源：`Q=1+0.5 sin(pi x1) sin(pi x2)`，与 epsilon 无关，范围 `[1,1.5]`。
- 边界：所有 `v·n<0` 的入流均为零。
- Knudsen 数：仅 `1` 和 `1e-3`。
- 角表示：完整四分量 `(j1,r1,j2,r2)`。

## 冻结 OE-APRFM 协议

- 两个 epsilon 共用 2×2×1 分区、每分量每空间 patch 64 个特征、scale 1、
  32×32×16 均匀配点、相同边界权重和 `rcond=1e-12`。
- 总系数数 `Ncoef=4×4×64=1024`；seeds 为 11、23、37，报告中位数。
- 不使用界面/源重要性采样、局部加密或逐 epsilon 调参。
- 保存 `Ef`、`Erho`、`Ncoef`、`Nrow`、`Nang`、求解时间、最小 f/rho、
  density correlation 和归一化最小二乘残差。

## 参考解与准入

每个 epsilon 用独立 OE-SN 确定性求解器生成至少两个细网格（优先 A/B/C），
记录相邻层的 `delta_f`、`delta_rho`、网格、角点数、残差、迭代数和时间。
只有当最终加密差分别不超过 OE 误差约 10% 时，才量化对应的 `Ef` 或 `Erho`。

正式三种子运行前，两种 epsilon 的 pilot 都应满足 `Erho<5e-2`，目标为 1%–3%；
`Ef` 目标不超过 5%，且 density 应为正、无显著负振荡。若 kinetic reference 未通过
加密判据，主表中的 `Ef` 写为 `--`，不得用未认证值替代。

主文只增加两个 P5 数值行，并为 `epsilon=1e-3` 生成一张四联图：scattering、
reference density、OE density 和 absolute density error。后两者必须共用色限。
