"""Aggregate completed seed records into manuscript-ready CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "relative_l2_f",
    "relative_l2_rho",
    "condition_number",
    "feature_seconds",
    "assembly_seconds",
    "solve_seconds",
    "total_seconds",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/p1_summary.csv"))
    args = parser.parse_args()
    groups: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for path in sorted(args.input_dir.glob("*.json")):
        record = json.loads(path.read_text())
        groups[(record["problem"], record["method"], float(record["epsilon"]))].append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["problem", "method", "epsilon", "seeds", "count"]
    fields += [f"{metric}_{stat}" for metric in METRICS for stat in ("median", "min", "max")]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, records in sorted(groups.items(), key=lambda item: item[0][2], reverse=True):
            row = dict(zip(("problem", "method", "epsilon"), key))
            row["seeds"] = ";".join(str(record["seed"]) for record in sorted(records, key=lambda x: x["seed"]))
            row["count"] = len(records)
            for metric in METRICS:
                available = [record[metric] for record in records if metric in record]
                if not available:
                    row[f"{metric}_median"] = ""
                    row[f"{metric}_min"] = ""
                    row[f"{metric}_max"] = ""
                    continue
                values = np.asarray(available, dtype=float)
                row[f"{metric}_median"] = float(np.median(values))
                row[f"{metric}_min"] = float(np.min(values))
                row[f"{metric}_max"] = float(np.max(values))
            writer.writerow(row)
    print(f"wrote {len(groups)} groups to {args.output}")


if __name__ == "__main__":
    main()
