#!/usr/bin/env python3
"""Summarize available OE-SI-DSA results without dropping failed runs."""

import csv
import json
from pathlib import Path

root = Path("results/baselines/oe_si_dsa")
rows = [json.loads(path.read_text()) for path in sorted(root.glob("*.json"))]
columns = (
    "problem", "epsilon", "method", "grid", "tolerance", "max_iterations",
    "iterations", "converged", "relative_l2_f", "relative_l2_rho",
    "final_difference", "relative_residual", "runtime_seconds",
)
output = Path("results/tables/oe_si_dsa_summary.csv")
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        row["grid"] = "x".join(map(str, row["grid"]))
        writer.writerow(row)
print(f"wrote {len(rows)} rows to {output}")
