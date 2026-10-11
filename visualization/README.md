# 可视化

不需要重新训练即可画出仓库内保存的结果：

```bash
python visualization/plot_results.py results/comparisons/oe_aprfm/fields/p3_oe_aprfm_eps_1e-03_seed_11.npz --output results/comparisons/figures/p3_density.png
python visualization/plot_results.py models/oe_apnn/baseline/p3/eps_1e-03/steps_20000/seed_11/history.npz --output results/comparisons/figures/p3_training.png
```

支持一维/二维密度场及 OE-APNN 训练损失，输出 PNG、PDF 等 Matplotlib 格式。二维孔洞按原数据 mask 隐去。

完整论文图表的生成与审查入口仍在 [scripts](../scripts/)，例如 [report_apnn_retrain.py](../scripts/report_apnn_retrain.py)。它们可能依赖本地完整运行或可选 paper/；这里的两个示例只依赖已入库数据。
