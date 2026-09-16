# 一维多分区 PoU 配置修复

本轮只处理组装/评估一致性，不调整损失权重，不改变输运方程。

`continuous_1d_odd_even.py` 使用 psi_b/dpsi_b 组装内部方程，但
`RandomFeatureSpaceXV` 原先使用 psi_a；边界模块和解构造器调用后者。
单分区归一化后两者一致，多分区交叠处不同。

在空间与正速度各两分区的交叠点，修复前奇部算子行 24/24 个元素与实际评估特征的
自动微分结果不一致，最大绝对差约 3.528。修复将 function_space.py 的
RandomFeatureSpaceXV._psi 统一为内部算子已使用的 psi_b，不改变其他特征类。
边界和解构造器通过共享特征类自动使用相同 PoU。

新增 test/test_1d_pou_consistency.py：比较宏观、微观、奇部矩阵全部行与实际评估特征
及其自动微分，覆盖交叠区域和正负速度；另检查单分区归一化等于原始单块特征。

验证命令：

```
PYTHONPATH=src JAX_PLATFORMS=cpu python3 -m pytest -q test/test_1d_pou_consistency.py test/test_experiment_configs.py
```

旧一维多分区结果需要修复后重新计算再使用；本轮未覆盖冻结表或重跑历史结果。
含时 NumPy 单分区原型没有调用这个类，因此本修复不会自动改善现有含时误差，
也不能据此宣称达到 1e-3。后续多分区含时扫描必须复用一致的 PoU 及其导数。

更正此前随机种子说明：模块默认 seedXV=42，但 run_p1_oe_aprfm.py 在构建前会设置为
当前实验 seed，不能仅凭默认值断言正式运行忽略种子。

后续已完成 P2 六组重算并更新冻结表，见 [结果与表格更新](pou_fixed_table_update_2026_09_08.md)。
