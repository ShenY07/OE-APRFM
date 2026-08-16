"""Create reproducible tuning rankings from per-run JSON records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "problem", "epsilon", "seed", "relative_l2_f", "relative_l2_rho",
    "condition_number", "rank", "partitions", "strides",
    "features_per_patch", "scale", "rcond", "block_weights",
    "total_seconds", "file",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/tuning"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/tuning_candidates.csv"))
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob("**/*.json")):
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "relative_l2_f" not in record:
            continue
        row = {key: record.get(key, "") for key in FIELDS}
        row["file"] = str(path)
        for key in ("partitions", "strides", "block_weights"):
            if row[key] != "":
                row[key] = json.dumps(row[key], sort_keys=True, separators=(",", ":"))
        rows.append(row)
    rows.sort(key=lambda row: (row["problem"], float(row["epsilon"]), float(row["relative_l2_f"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} candidates to {args.output}")


if __name__ == "__main__":
    main()
