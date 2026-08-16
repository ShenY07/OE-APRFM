"""Summarize the controlled P3 spatial-domain-decomposition experiment."""

import csv, json
from pathlib import Path

root=Path("results/quadrant/domain_decomposition")
baseline=Path("results/quadrant/final")
records=[]
for eps in (1.0,1e-3):
    p=baseline/f"p3_oe_aprfm_eps_{eps:.0e}_seed_11_quadrant_f256.json"
    records.append(json.loads(p.read_text()))
records += [json.loads(p.read_text()) for p in root.glob("*.json")]
records.sort(key=lambda r:(-float(r["epsilon"]),int(r["partitions"][0])*int(r["partitions"][1])))
fields=("epsilon","partitions","strides","features_per_patch","rank","condition_number","relative_l2_f","relative_l2_rho","total_seconds")
out=Path("results/tables/p3_domain_decomposition.csv"); out.parent.mkdir(parents=True,exist_ok=True)
with out.open("w",newline="") as h:
    w=csv.DictWriter(h,fieldnames=("epsilon","spatial_patches","total_features")+fields[1:]); w.writeheader()
    for r in records:
        row={k:r[k] for k in fields}; patches=int(r["partitions"][0])*int(r["partitions"][1]); row["spatial_patches"]=patches; row["total_features"]=patches*int(r["features_per_patch"]); row["partitions"]="x".join(map(str,r["partitions"])); row["strides"]=json.dumps(r["strides"],sort_keys=True); w.writerow(row)
print(out)
