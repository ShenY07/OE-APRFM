"""Summarize completed matched MM-APRFM seed sweeps."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FIELDS = (
    "relative_l2_f",
    "relative_l2_rho",
    "condition_number",
    "rank",
    "num_rows",
    "num_columns",
    "oversampling_ratio",
    "feature_seconds",
    "assembly_seconds",
    "solve_seconds",
    "total_seconds",
)


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path("results/baselines/mm_aprfm"))
parser.add_argument("--output", type=Path, default=Path("results/tables/mm_aprfm_summary.csv"))
args = parser.parse_args()
records = [json.loads(path.read_text()) for path in args.root.glob("*.json")]
rows = []
for problem in ("p1", "p2", "p3", "p4", "p5"):
    for epsilon in (1.0, 1.0e-3):
        group = [r for r in records if r["problem"] == problem and r["epsilon"] == epsilon]
        if not group:
            continue
        row = {"problem": problem, "epsilon": epsilon, "seeds": len(group)}
        for field in FIELDS:
            values = np.asarray([r[field] for r in group], dtype=float)
            row[f"{field}_median"] = float(np.median(values))
            row[f"{field}_min"] = float(values.min())
            row[f"{field}_max"] = float(values.max())
        rows.append(row)

args.output.parent.mkdir(parents=True, exist_ok=True)
columns = list(rows[0]) if rows else ["problem", "epsilon", "seeds"]
with args.output.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {len(rows)} rows to {args.output}")
