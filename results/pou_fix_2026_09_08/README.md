# 一维 PoU 修复后的结果更新

固定旧 P2 参数：partitions=2×4、J=64、scale=1、rcond=1e-6、
内部 32×32、边界参数 64、两个 ε=1/1e-3、seeds=11/23/37。
只修复一维组装/评估 PoU 一致性，没有调权重或参数。

- `p2/`：新结果及日志。
- `references/`：从旧 consistency/p2 的 NPZ 恢复的同一参考场。
  原独立参考 NPZ 已缺失，旧结果保留了 x、velocity、reference_f、reference_rho。
  节点为 [-1,1] 的 512 个均匀点；梯形权重验证复现参考密度和旧 Ef 到浮点精度。
  这不是新生成或重新认证的参考解。
- `tables_before/`、`figures_before/`：更新前备份。
- `unaffected_single_patch_sources.json`：已检查不受本次多分区修复影响的表格来源。
- 完成后的 `comparison.json` 和 `manifest.json` 记录新旧误差、更新文件及源码摘要。

表格与图 5 统一通过 `scripts/p2_corrected_results.py` 加载完整六组新数据，
缺失/重复/不匹配时拒绝生成，不自动混用旧值。
图 5 使用 NPZ 内按梯形求积得到的 rho，不再使用简单算术角平均。
历史 consistency 与 epsilon_scan 记录保留作归档，不作为当前冻结 P2 表/图来源。
