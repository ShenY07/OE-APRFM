"""Report convergence limits and terminal residuals for all SI baselines."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
paths = sorted((root / "results/baselines/si_dsa").glob("*.json")) + sorted((root / "results/baselines/oe_si_dsa").glob("*.json"))
lines = ["# SI 迭代诊断", "", "| 问题 | ε | 方法 | 状态 | 迭代计数 | 上限 | 容差 | 终止量 | 停止准则 |", "|---|---:|---|---|---:|---:|---:|---:|---|"]
for path in paths:
    r = json.loads(path.read_text())
    krylov = "krylov" in r["method"]
    criterion = "预条件相对 GMRES 残差 < tol" if krylov else "相邻迭代解的 L2 差 < tol"
    terminal = r.get("relative_residual") if krylov else r.get("final_difference")
    limit = f"{r['max_iterations']} restart cycles" if krylov else str(r["max_iterations"])
    lines.append(f"| {r['problem'].upper()} | {r['epsilon']:.0e} | {r['method']} | {'收敛' if r['converged'] else '未收敛'} | {r['iterations']} | {limit} | {r['tolerance']:.1e} | {terminal:.4e} | {criterion} |")
(root / "results/tables/iterative_solver_diagnostics.md").write_text("\n".join(lines) + "\n")
