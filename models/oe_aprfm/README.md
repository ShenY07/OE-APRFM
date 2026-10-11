# OE-APRFM 模型状态

已收集的 P3/P4 稳态实验保存了预测场和指标，未保存重建所需的随机特征状态与求解系数，因此这里没有可重载的稳态模型。

预测场在 [results/comparisons/oe_aprfm/fields](../../results/comparisons/oe_aprfm/fields/)，训练/求解入口见 [training](../../training/)。E6 运行器会保存系数和轨迹，运行后位于 results/e6_periodic/，但本次未将其历史运行纳入模型集。
