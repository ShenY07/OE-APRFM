# 统一计时与稳定性校准协议

## 1. 统一计时

所有方法报告

\[
T_{total}=T_{setup}+T_{assembly}+T_{solve}+T_{evaluation}.
\]

计时前在同一进程内执行一次完整预热；预热不写入统计。正式计时独立执行 3 次，报告中位数及范围，同时保留三次原始分量。

统一排除参考解生成、绘图、数据保存、首次 JIT/库加载以及与方法无关的文件读取。传统离散方法的 `setup` 包括网格、角向求积和算子初始化，`assembly` 包括离散算子/预条件器构造，`solve` 为迭代过程，`evaluation` 为统一误差网格上的插值及误差计算。

## 2. 校准顺序

1. 每行归一化后，对 boundary、macro、even、odd 四个残差块分别乘以 $N_b^{-1/2}$。
2. 求解前进行矩阵列二范数归一化；输出归一化前后条件数。
3. P1、P3 使用 seed 11 校准：
   - `scale ∈ {0.5,1,2}`；
   - `rcond ∈ {1e-8,1e-10,1e-12,1e-14}`；
   - `Nrow/Ncoef ∈ {4,6,8,10}`。
4. 固定特征数：P1 使用 `J=64`，P3 使用 `J=128`；分区均为单区域。
5. 每个候选同时检查 $\varepsilon=1,10^{-3},10^{-6}$，不得按单个 $\varepsilon$ 选择参数。
6. 参数选择不使用参考解误差，依据独立验证残差、最坏数值秩比例、系数范数和条件数进行。
7. 冻结参数后才执行完整 ε 扫描、J 收敛和三种子统计。

## 3. 输出字段

每个原始结果至少保存：`training_residual_rms`、`validation_residual_rms`、`condition_number_raw`、`condition_number_scaled`、`rank`、`num_columns`、`coefficient_norm`、`oversampling_ratio`、四个计时分量、预热次数和三次计时原始值。

## 4. 最终实验顺序

1. 统一计时与输出格式；
2. 稳定性校准；
3. 冻结参数；
4. P1–P3 ε 扫描；
5. P1/P3 J 收敛；
6. OE/MM/Direct-RFM 条件数比较；
7. 与传统方法的误差—时间 Pareto 曲线；
8. ε 一致稳定性及拟最优误差结果。
