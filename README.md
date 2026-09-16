# OE-APRFM numerical experiments

本仓库只保留 JSC 稿件所需的五个互补实验及其公共求解组件。实验定义位于
`src/configuration/`：

| 配置 | 问题 | 作用 | 参考解 |
|---|---|---|---|
| `p1_manufactured_1d.py` | P1 一维制造解 | 全尺度 ε 一致精度与条件数 | 解析解 |
| `p2_heterogeneous_1d.py` | P2 一维非均匀纯散射 | 界面与边界层 | parity-SI+DSA 数值参考解；内部加密校验 |
| `p3_manufactured_2d.py` | P3 二维光滑制造解 | 相空间与宏观精度 | 解析解 |
| `p4_circular_hole_2d.py` | P4 圆孔区域 | 曲线内边界 | 修正后的解析解 |
| `p5_heterogeneous_2d.py` | P5 平滑分层介质与内源 | 二维非制造应用测试 | 独立 OE-SN 加密数值参考解 |

公共协议冻结在 `experiment_defaults.py`，包括三个随机种子、独立评估网格、
行缩放、SVD 截断、误差输出和 ε 序列。P5 固定使用 `sigma_s=1+x2`、零吸收、
平滑正内源、真空入流以及同一套 2×2×1/64-feature/32×32×16 配置；只测试
ε=1 和 1e-3。其误差必须通过独立 OE-SN 参考解的加密判据后才能进入结果表。

## 运行与报告原则

- 随机特征方法使用同一分区、配置预算和停止准则；正式结果运行 seeds 11、23、37。
- 误差只能在独立评估网格上计算，报告中位数及 `[min,max]`。
- P2 parity-SI+DSA 参考解记录正式离散、速度求积和迭代容差；内部网格加密仅作参考解精度检查，不列为独立实验。
- 条件数使用与求解相同 SVD 截断下的行缩放矩阵；时间拆分为特征、组装和求解。
- CPU/GPU 结果分表；只有同一硬件上的结果可称为 speedup。

运行测试：

```bash
PYTHONPATH=src pytest -q
```

核心算法仍分别位于 `src/constraints/`、`src/modules/`、`src/solver/` 和
`src/numerical/`。旧的重复实验、失效参考数据和带输出 Notebook 已移除，最终
数值结果应重新由上述冻结配置生成。

## 对比基模

| 方法 | 分解形式 | 代码位置 | 运行入口 |
|---|---|---|---|
| OE-OERFM | odd-even | `src/` | `scripts/run_p*_oe_aprfm.py` |
| MM-OERFM | micro-macro | `baselines/mm_oerfm/` | `scripts/run_mm_oerfm_baseline.py` |
| RFM | original transport equation | `baselines/mm_oerfm/original/` | `scripts/run_rfm_1d.py` |
| SI-DSA | transport sweep + DSA | `src/numerical/parity_reference.py` | `scripts/generate_reference.py` |
| OE-SI-DSA | odd-even iteration + DSA | `baselines/oe_si_dsa/` | 待完成运行器 |

MM-OERFM 保留 `/home/sheny-y23/rte` 的原始代码与六个实验 notebook，并通过
独立 `PYTHONPATH` 运行，避免覆盖当前 OE-OERFM 模块。统一对比协议记录在
`baselines/mm_oerfm/baseline.json`。
