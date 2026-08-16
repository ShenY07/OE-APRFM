#!/usr/bin/env python3
"""Append the current external comparisons and data inventory to the main report."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / "results/tables/all_experiment_results.md"
text = target.read_text()
marker = "## 10. 外部基准模型统一比较"
base = text.split(marker, 1)[0]

baseline = (ROOT / "results/tables/baseline_model_comparison.md").read_text()
baseline = baseline.split("\n", 1)[1].strip()
three_seed = (ROOT / "results/tables/three_seed_method_comparison.md").read_text()
three_seed = three_seed.split("\n", 1)[1].strip()
costs = (ROOT / "results/tables/baseline_cost_parameters.md").read_text()
costs = costs.split("\n", 1)[1].strip()

inventory = """| 数据集 | 当前有效数量 | 状态 | 位置 |
|---|---:|---|---|
| OE-APRFM 三种子补充 | 20 组 | 完成 | `results/baselines/oe_aprfm_3seed` |
| MM-APRFM 汇总 | 10 个问题-尺度组合 | 完成 | `results/tables/mm_aprfm_summary.csv` |
| OE-APNN | 30 组（5 问题 × 2 尺度 × 3 seeds） | 完成 | `results/baselines/oe_apnn_imported` |
| OE-SI-DSA | 10 组 | 完成；未收敛项保留 | `results/baselines/oe_si_dsa` |
| RFM 代表实验 | 2 组（P1） | 完成 | `results/baselines/rfm` |
| SI-DSA 代表实验 | 2 组（P1） | 完成；低尺度未收敛 | `results/baselines/si_dsa` |
| P2 parity-SI 参考解 | 2 组 | 完成 | `results/references` |
| OE-APNN 独立解图 | 60 张 | 完成 | `results/figures/oe_apnn` |
| 时间与规模对比图 | 4 张 | 完成 | `results/figures/method_comparison` |

## 13. 当前数据边界

| 项目 | 当前记录 |
|---|---|
| OE-APNN P5 | 使用归档协议的中心方孔制造解版本 |
| 其他方法 P5 | 使用当前异质介质版本 |
| P5 跨方法比较 | 两种 P5 不直接横向比较 |
| OE-APNN 时间 | GPU 训练时间 |
| 随机特征方法时间 | 特征生成、组装、求解及评估总时间 |
| SI-DSA/OE-SI-DSA 时间 | 迭代求解时间 |
| 未收敛结果 | 保留误差、迭代上限及收敛状态，不标记为成功收敛 |
"""

header = base.replace("# OERFM 实验结果汇总", "# 全部数值实验结果汇总", 1)
if "数据更新时间：" not in header[:120]:
    header = header.replace("# 全部数值实验结果汇总\n", "# 全部数值实验结果汇总\n\n数据更新时间：2026-08-15。\n", 1)

combined = (header.rstrip() + "\n\n" + marker + "\n\n" + baseline +
            "\n\n## 11. 三种子方法比较\n\n" + three_seed +
            "\n\n## 12. 当前结果数据清单\n\n" + inventory +
            "\n## 14. 时间成本与参数量级\n\n" + costs)
target.write_text(combined.rstrip() + "\n")
print(target)
