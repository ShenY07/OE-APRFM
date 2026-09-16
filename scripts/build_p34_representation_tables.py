"""Build the formal P3/P4 four-component and representation-comparison tables."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "p34_representation_comparison"
OUT = ROOT / "results" / "tables"
SEEDS = (11, 23, 37)


def sci(value: float) -> str:
    return f"{float(value):.8e}"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    records = [json.loads(path.read_text()) for path in sorted(RAW.glob("*.json"))]
    records = [record for record in records if record["seed"] in SEEDS]
    if len(records) != 24:
        raise RuntimeError(f"expected 24 formal records, found {len(records)}")

    table1_path = OUT / "table_1.csv"
    if table1_path.exists():
        with table1_path.open(newline="") as handle:
            config_rows = list(csv.DictReader(handle))
        replacements = {
            "P3": ("P3 smooth manufactured 2D (four component)", "128", "15856", "30.9688"),
            "P4": ("P4 perforated domain (four component)", "128", "14736", "28.7812"),
        }
        for row in config_rows:
            prefix = row["problem"][:2]
            if prefix in replacements:
                row["problem"], row["J"], row["Nrow"], row["ratio"] = replacements[prefix]
        write_csv(table1_path, config_rows)

    # Replace the P3/P4 portion of the canonical 2-D raw-error table with the
    # four-component, first-quadrant formulation.  P5 is deliberately retained.
    table4_path = OUT / "table_4.csv"
    retained = []
    if table4_path.exists():
        with table4_path.open(newline="") as handle:
            retained = [row for row in csv.DictReader(handle) if row["problem"] not in ("P3", "P4")]
    four = [record for record in records if record["angular_representation"] == "four_component"]
    main_rows = [
        {
            "problem": record["problem"].upper(),
            "epsilon": f"{record['epsilon']:.0e}",
            "seed": record["seed"],
            "Ef": sci(record["relative_l2_f"]),
            "Erho": sci(record["relative_l2_rho"]),
            "rank": f"{record['rank']}/{record['num_columns']}",
        }
        for record in sorted(four, key=lambda r: (r["problem"], -r["epsilon"], r["seed"]))
    ]
    write_csv(table4_path, main_rows + retained)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        groups[(record["problem"], record["epsilon"], record["angular_representation"])].append(record)

    comparison = []
    for (problem, epsilon, representation), group in sorted(groups.items()):
        if sorted(record["seed"] for record in group) != list(SEEDS):
            raise RuntimeError(f"incomplete seed set for {(problem, epsilon, representation)}")
        med = lambda key: float(np.median([record[key] for record in group]))
        comparison.append(
            {
                "problem": problem.upper(),
                "epsilon": f"{epsilon:.0e}",
                "representation": representation,
                "sampled_angle_domain": "[0,pi/2]" if representation == "four_component" else "[0,pi]",
                "components": 4 if representation == "four_component" else 2,
                "features_per_component": int(med("features_per_patch")),
                "Ncoef": int(med("num_columns")),
                "Nrow": int(med("num_rows")),
                "Ef_median": sci(med("relative_l2_f")),
                "Erho_median": sci(med("relative_l2_rho")),
                "condition_median": sci(med("condition_number")),
                "assembly_seconds_median": f"{med('assembly_seconds'):.4f}",
                "total_seconds_median": f"{med('total_seconds'):.4f}",
                "rank_median": int(med("rank")),
                "seeds": "11;23;37",
            }
        )
    write_csv(OUT / "table_11_p34_two_vs_four_component.csv", comparison)

    labels = {"legacy_two_component": "二分量", "four_component": "四分量"}
    lines = [
        "# P3/P4 二分量与四分量对比",
        "",
        "固定口径：16×16×16 配点、单空间分区、32 阶角积分、rcond=1e-12、总未知数 Ncoef=512；数值为 seeds 11/23/37 的中位数。四分量仅在第一象限 [0,π/2] 采样，二分量沿用 [0,π]。",
        "",
        "| 问题 | ε | 表示 | 采样角域 | 分量数×每分量 J | Nrow | Ef | Eρ | κ(A) | 组装(s) | 总时间(s) |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['problem']} | {row['epsilon']} | {labels[row['representation']]} | "
            f"{row['sampled_angle_domain']} | {row['components']}×{row['features_per_component']} | "
            f"{row['Nrow']} | {row['Ef_median']} | {row['Erho_median']} | "
            f"{row['condition_median']} | {row['assembly_seconds_median']} | {row['total_seconds_median']} |"
        )
    lines += [
        "",
        "说明：等总未知数比较隔离了角表示本身的影响。四分量把四个象限值编码为 (j1,r1,j2,r2)，因此基角度采样域减半；P3、ε=1e-3 的 Ef 中位数略高，其余组合的 Ef 及全部组合的条件数均改善。",
        "",
    ]
    (OUT / "table_11_p34_two_vs_four_component.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
