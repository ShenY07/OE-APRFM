# SI 迭代诊断

| 问题 | ε | 方法 | 状态 | 迭代计数 | 上限 | 容差 | 终止量 | 停止准则 |
|---|---:|---|---|---:|---:|---:|---:|---|
| P1 | 1e+00 | si_dsa | 收敛 | 39 | 500 | 1.0e-09 | 8.7662e-10 | 相邻迭代解的 L2 差 < tol |
| P1 | 1e-03 | si_dsa | 未收敛 | 500 | 500 | 1.0e-09 | 1.2681e-02 | 相邻迭代解的 L2 差 < tol |
| P1 | 1e+00 | oe_si_dsa | 收敛 | 39 | 500 | 1.0e-09 | 8.7662e-10 | 相邻迭代解的 L2 差 < tol |
| P1 | 1e-03 | oe_si_dsa | 未收敛 | 500 | 500 | 1.0e-09 | 1.2681e-02 | 相邻迭代解的 L2 差 < tol |
| P2 | 1e+00 | oe_si_dsa | 收敛 | 16 | 500 | 1.0e-09 | 2.6432e-10 | 相邻迭代解的 L2 差 < tol |
| P2 | 1e-03 | oe_si_dsa | 未收敛 | 500 | 500 | 1.0e-09 | 3.7239e-06 | 相邻迭代解的 L2 差 < tol |
| P3 | 1e+00 | oe_si_dsa_krylov | 收敛 | 212 | 500 restart cycles | 1.0e-09 | 9.2255e-10 | 预条件相对 GMRES 残差 < tol |
| P3 | 1e-03 | oe_si_dsa_krylov | 收敛 | 249 | 500 restart cycles | 1.0e-09 | 9.6312e-10 | 预条件相对 GMRES 残差 < tol |
| P4 | 1e+00 | oe_si_dsa_krylov | 收敛 | 203 | 500 restart cycles | 1.0e-09 | 9.7393e-10 | 预条件相对 GMRES 残差 < tol |
| P4 | 1e-03 | oe_si_dsa_krylov | 收敛 | 167 | 500 restart cycles | 1.0e-09 | 9.7986e-10 | 预条件相对 GMRES 残差 < tol |
| P5 | 1e+00 | oe_si_dsa_krylov | 收敛 | 253 | 500 restart cycles | 1.0e-09 | 9.7667e-10 | 预条件相对 GMRES 残差 < tol |
| P5 | 1e-03 | oe_si_dsa_krylov | 收敛 | 8028 | 500 restart cycles | 1.0e-09 | 1.0000e-09 | 预条件相对 GMRES 残差 < tol |
