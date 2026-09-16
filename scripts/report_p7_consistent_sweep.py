"""Validate and summarize a complete original-convention P7 sweep."""
import json
from pathlib import Path
import math

root=Path(__file__).resolve().parents[1]
rows=json.loads((root/'results/p7_transient_pilot/consistent_pou_sweep.json').read_text())
expected={(p,e) for p in ((1,1,1),(1,2,2),(2,2,2)) for e in (1,.01,.001)}
assert len(rows)==9 and {(tuple(r['partitions']),r['epsilon']) for r in rows}==expected
lines=['# P7 一致 PoU 含时实现：固定参数跨 ε 验证','','本轮沿用原方法的特征/PoU约定，不引入新的宏观空间或前沿基。',
'固定每组参数运行 ε=1、.01、.001，seed=11；以下是单种子诊断，不是三种子认证。','',
'## 实现与验证','',
'- 新增 ConsistentFeatures：t/x/v 均按分区中心及半宽归一化，正速度域分区、PoU 对负速度镜像。',
'- 使用原 psi_b 的函数与导数公式；归一化和乘积导数作用在与解评估完全相同的特征上。',
'- 随机参数为 [0,scale)；r/j 共享特征池，NumPy 随机序列不等于 Flax 逐位复现。',
'- 系数仍为常数，整个时间区间一次线性求解；没有改动 PDE 或增加新方程。',
'- 固定单位行 L2 归一化、行求和、重复宏观行，初边值权重均为1；rcond=1e-12。',
'- 每组固定内部1024时空点×16正角度，每侧边界1024点、初值2048点；与之前小预算不同，跨组/跨ε相同。',
'- 新增断点续跑；可选 P7_BLAS_THREADS，默认仍1。部分扫描用4线程，不比较运行速度。',
'- 导数有限差分、奇偶性、共享特征池、单分区与上一轮对齐特征逐项相等检查通过。',
'- 原P7制造解、局部导数和参考扩散极限测试通过。','',
'## t=.1 结果','',
'Er、Ej 分别为 r=(f(v)+f(−v))/2、j=(f(v)−f(−v))/(2ε) 相对参考解的角求积加权 L2 误差。',
'它们都是相对误差；Ej 对应缩放奇部 j，而不是 εj 的绝对误差。参考 j 为面通量平均到单元中心后的值。','',
'| t×x×v分区 | J/场总数 | ε | Ef | Eρ | Er | Ej | 独立初值RMS | 独立宏观RMS |',
'|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 assert r['repeat_macro'] and not r['block_average']
 assert all(math.isfinite(s[k]) for s in r['snapshots'] for k in ('E_f','E_rho'))
 s=r['snapshots'][-1];h=r['held_out']
 lines.append(f"| {'×'.join(map(str,r['partitions']))} | {r['J']} | {r['epsilon']:g} | {s['E_f']:.6g} | {s['E_rho']:.6g} | {s['E_r']:.6g} | {s['E_j']:.6g} | {h['initial_rms']:.5g} | {h['macro_rms']:.5g} |")
passed=[r for r in rows if all(s[k]<=1e-3 for s in r['snapshots'] for k in ('E_f','E_rho'))]
lines+=['',f'两个正时间快照 Ef/Eρ 均≤1e-3 的运行数：{len(passed)}/9。',
'完整数据还保留 t=.05、独立微观/奇部/边界残差和矩阵秩，不能仅看最终密度误差。','',
'参考仍为256空间单元、16正角度、800步。ε=1的参考尚不足以认证1e-3；这里只能用于筛查明显大误差。',
'独立残差为有限随机点检查，不是全域严格界。Er/Ej 相对同一离散参考解计算，尚未分别做参考加密认证。','',
'本轮实现一致性修复与参数筛查不能自动证明含时 ε-一致收敛；若未达标，不将结果写入冻结成功算例表。',
'原有 P2 修复后的表格不受本轮修改影响。','',
'复现：`python3 -m pytest -q test/test_p7_consistent_features.py`；',
'`P7_BLAS_THREADS=4 python3 scripts/run_p7_consistent_sweep.py`；',
'`python3 scripts/report_p7_consistent_sweep.py`。']
lines+=['','## 当前结论','',
 '2×2×2 的固定配置在三个 ε 上的最终密度误差为 0.37567、0.009268、0.012705，',
 '独立初值 RMS 为 0.09581、0.09113、0.09565。相比本轮单分区，解误差有所降低，',
 '但宏观/奇部独立残差没有同时下降。因此不能宣称全面改善或含时精度已解决。',
 '后续可检查局部分辨率与初始时间层采样；本轮没有证明继续增大参数必然达到1e-3，',
 '也没有反证原方法的所有参数配置。']
(root/'docs/p7_consistent_pou_sweep_2026_09_08.md').write_text('\n'.join(lines)+'\n')
print('Complete sweep verified;',len(passed),'runs meet the 1e-3 screen.')
