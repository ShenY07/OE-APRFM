"""Write a compact manuscript-facing OE-APRFM result table."""

import csv
from pathlib import Path


def scientific(value):
    return f"{float(value):.4e}" if value else "--"


rows=list(csv.DictReader(open("results/tables/tuned_validation_summary.csv")))
lines=["# OE-APRFM main error table","", "Five-seed median with `[minimum, maximum]` dispersion.","", "| Problem | epsilon | seeds | median E_f [min,max] | median E_rho [min,max] | median condition number | median time (s) |", "|---|---:|---:|---:|---:|---:|---:|"]
for row in sorted(rows,key=lambda r:(r["problem"],-float(r["epsilon"]))):
    ef=f"{scientific(row['relative_l2_f_median'])} [{scientific(row['relative_l2_f_min'])}, {scientific(row['relative_l2_f_max'])}]"
    er="--" if not row["relative_l2_rho_median"] else f"{scientific(row['relative_l2_rho_median'])} [{scientific(row['relative_l2_rho_min'])}, {scientific(row['relative_l2_rho_max'])}]"
    lines.append(f"| {row['problem'].upper()} | {scientific(row['epsilon'])} | {row['count']} | {ef} | {er} | {scientific(row['condition_number_median'])} | {float(row['total_seconds_median']):.2f} |")
lines.extend(["", "All entries use the selected OE-RFM parameters in `selected_oerfm_parameters.csv`."])
Path("results/tables/oerfm_main_table.md").write_text("\n".join(lines)+"\n")
