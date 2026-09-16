#!/usr/bin/env python3
"""Write the three-seed OE-APNN accuracy/time table."""

import json
from pathlib import Path
import numpy as np

roots = [Path("results/baselines/oe_apnn"), Path("results/baselines/oe_apnn_imported")]
rows = []
for problem in range(1, 6):
    for label, epsilon in (("eps_1e0", 1.0), ("eps_1e-3", 1e-3)):
        by_seed = {}
        for root in roots:
            for path in (root/f"P{problem}"/label).glob("seed_*/metrics.json"):
                record=json.loads(path.read_text())
                if record.get("evaluation_status") == "complete": by_seed[int(record["seed"])]=record
        records = list(by_seed.values())
        if not records or any(r.get("E_f") is None for r in records):
            continue
        row = [f"P{problem}", epsilon]
        for key in ("E_f", "E_rho", "train_time_s"):
            values = np.asarray([r[key] for r in records], float)
            row.extend((values.mean(), values.std(ddof=1)))
        rows.append(row)
output = Path("results/tables/oe_apnn_three_seed.md")
lines = [
    "# OE-APNN 三种子结果", "",
    "| 问题 | ε | E_f 均值 | E_f 标准差 | E_ρ 均值 | E_ρ 标准差 | 训练时间均值/s | 训练时间标准差/s |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append("| " + " | ".join([row[0], f"{row[1]:.0e}"] + [f"{v:.3e}" for v in row[2:]]) + " |")
lines.extend(["", "P5 使用压缩包训练协议中的中心方孔制造解版本。"])
output.write_text("\n".join(lines)+"\n")
print(f"wrote {len(rows)} rows to {output}")
