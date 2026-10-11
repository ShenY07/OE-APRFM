# 对比结果

- [oe_apnn/summary.csv](oe_apnn/summary.csv)：聚合误差和训练成本；[seedwise.csv](oe_apnn/seedwise.csv)：逐种子指标。对应权重在 [models](../../models/)。
- [oe_aprfm/summary.csv](oe_aprfm/summary.csv)：P3/P4 的特征规模和表示方式对比；[runs/](oe_aprfm/runs/) 保存全部 30 组逐次运行 JSON。
- [oe_aprfm/fields/](oe_aprfm/fields/)：P3/P4 四分量表示、J=128、ε=10⁻³、seed=11 的两份完整预测场，用于直接绘图；其余大体积场仍保留在本地原始结果目录。
- [figures/](figures/)：可直接查看的示例图，生成代码在 [visualization](../../visualization/)。
- [角域对比](angular_domains/)：缩减与展开角域的 12 组记录和汇总。

不同实验使用不同种子、计算设备和预算，不能直接把耗时解释为公平速度比。OE-APNN 的 P5 使用精确解方孔配置，与生产 P5 异质边界驱动配置不同，不能仅凭 P5 名称合并指标。原始配置与来源均保留，不重新标注为一次新实验。
