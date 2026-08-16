# Deterministic baselines

本目录保留与五个核心实验仍有关的源迭代、奇偶迭代、DSA 和微宏基线实现。
物理问题必须从 `configuration/p1_*.py` 至 `configuration/p5_*.py` 读取，不能
复用旧 ex 编号的缓存数据。

P2 正式参考解使用 parity-SI+DSA，离散分辨率、速度求积、迭代容差及内部网格
加密结果记录在 `docs/numerical_experiments.md`。内部加密只用于确认参考解误差
低于 OERFM 误差，不作为独立实验或正文结果表。参考解生成时间不计入方法耗时。
