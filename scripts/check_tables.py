"""Additional cell-level checks for retained manuscript tables and update status."""

import csv, json, re, hashlib, sys
from pathlib import Path
from scripts.repo_paths import resolve_path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
ARC = ROOT / ".release-backup/20261009/original-worktree"
OUT = ROOT / "results/checks/tables"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ARC / "scripts"))
from p2_corrected_results import corrected_records
from e5_timing_data import timing_summary

text = (ROOT / "paper/tables.tex").read_text()
tables = {
    re.search(r"\\label\{([^}]+)\}", b)[1]: b
    for b in re.findall(r"\\begin\{table\}.*?\\end\{table\}", text, re.S)
}


def rows(label):
    b = (
        tables["tab:" + label]
        .split(r"\midrule", 1)[1]
        .split(r"\bottomrule")[0]
    )
    b = re.sub(r"\\addlinespace(?:\[[^\]]*\])?", "", b)
    return [
        [c.strip() for c in row.split("&")]
        for row in re.split(r"\\\\", b)
        if row.strip() and r"\multicolumn" not in row
    ]


def csvs(p):
    with p.open() as f:
        return list(csv.DictReader(f))


checks = []
errors = []


def check(label, row, col, value):
    cell = rows(label)[row][col]
    m = re.search(r"([0-9.]+)\\times10\^\{?(-?\d+)\}?", cell)
    if m:
        mant, exp = m[1], int(m[2])
    else:
        mant = re.search(r"-?\d+(?:\.\d+)?", cell)[0]
        exp = 0
    actual = float(mant) * 10**exp
    dec = len(mant.split(".")[1]) if "." in mant else 0
    tol = 0.501 * 10 ** (exp - dec)
    ok = abs(float(value) - actual) <= tol
    r = dict(
        table=label,
        row=row + 1,
        column=col + 1,
        printed=cell,
        source_value=float(value),
        matches_rounding=ok,
    )
    checks.append(r)
    if not ok:
        errors.append(r)


base = ARC / "results"
# Minimum gain, collocation, angular manufactured fields, and angular OE/MM comparison.
raw = json.loads(
    (base / "minimum_gain/p1_J16_seed11_refined128.json").read_text()
)
for i, e in enumerate((1.0, 0.1, 0.01, 0.001, 1e-6, 0.0)):
    r = next(
        r
        for r in raw["rows"]
        if r["epsilon"] == e
        and r["nx"] == 128
        and r["nv"] == 128
        and r["nq"] == 128
    )
    check("minimum-gain", i, 1, r["beta"])
frozen = base / "requirement_2026_08_20/tables_frozen"
for i, r in enumerate(csvs(frozen / "table_3_collocation_refinement.csv")):
    for c, k in [
        (1, "oversampling_ratio"),
        (2, "relative_l2_f"),
        (3, "relative_l2_rho"),
    ]:
        check("collocation", i, c, r[k])
for i, r in enumerate(csvs(frozen / "table_6_angular_manufactured.csv")):
    for c, k in enumerate(
        ("relative_l2_f", "relative_l2_rho", "relative_l2_r", "relative_l2_j"),
        1,
    ):
        check("angular-manufactured", i, c, r[k])
a = csvs(base / "p6_mm_comparison/summary.csv")
for i, e in enumerate((1.0, 0.001, 1e-6)):
    for method, col in [("OE-APRFM", 1), ("MM-APRFM", 2)]:
        r = next(
            r for r in a if float(r["epsilon"]) == e and r["method"] == method
        )
        check("angular-mm", i, col, r["E_f"])
        check("angular-mm", i, col + 2, r["flux_relative_l2"])
# P2 must use the post-PoU-fix records, not the obsolete frozen CSV.
p2 = corrected_records()
for i, e in enumerate((1.0, 0.001)):
    for c, k in [(2, "relative_l2_f"), (3, "relative_l2_rho")]:
        check(
            "extended", i, c, median(r[k] for p, r in p2 if r["epsilon"] == e)
        )
# Transient table: all four error columns, rather than only the plotted kinetic errors.
a = csvs(base / "e6_mm_comparison/summary.csv")
for i, (e, method) in enumerate(
    ((1.0, "MM"), (1.0, "OE"), (0.001, "MM"), (0.001, "OE"))
):
    r = next(
        r for r in a if float(r["epsilon"]) == e and r["method"] == method
    )
    for c, k in enumerate(("Ef_CT", "Ef_BE", "Erho_BE", "Eq_BE"), 2):
        check("transient-comparison", i, c, r[k])
a = csvs(base / "e6_periodic/time_refinement.csv")
for label, ref, keys in [
    ("time-refinement", "ct", ("Ef", "Erho", "Eq_relative")),
    ("be-refinement", "be", ("Ef", "Erho", "Eq_relative", "Ediff_BE")),
]:
    for i, (e, dt) in enumerate(
        (e, dt) for e in (1.0, 0.001) for dt in (0.004, 0.002, 0.001)
    ):
        r = next(
            r
            for r in a
            if float(r["epsilon"]) == e
            and float(r["dt"]) == dt
            and r["reference"] == ref
        )
        for c, k in enumerate(keys, 2):
            check(label, i, c, r[k])
a = csvs(base / "e6_periodic/assembly_feature_checks.csv")
for i, (kind, e, t) in enumerate(
    (kind, e, t)
    for kind in ("feature_check", "quadrature_check")
    for e in (1.0, 0.001)
    for t in (0.02, 0.1, 0.2)
):
    r = next(
        r
        for r in a
        if r["run"].startswith(kind + f"/s11_eps{e:.0e}")
        and float(r["t"]) == t
    )
    for c, k in enumerate(("f_change", "rho_change", "q_change"), 4):
        check("transient-field-checks", i, c, r[k])
# P5 independent deterministic-reference solver records and refinement differences.
for i, (e, level) in enumerate(
    (e, level) for e in (1.0, 0.001) for level in ("A", "B", "C")
):
    p = (
        base
        / "references_p5_boundary"
        / f"p5_parity_ref_eps_{e:.0e}_level_{level}.json"
    )
    r = json.loads(p.read_text())
    check("p5-reference", i, 3, r["grid"][2])
    check("p5-reference", i, 4, r["iterations"])
    check("p5-reference", i, 5, r["relative_residual"])
    if level != "A":
        pair = "BA" if level == "B" else "CB"
        r = json.loads(
            (
                p.parent / f"p5_parity_ref_eps_{e:.0e}_refinement_{pair}.json"
            ).read_text()
        )
        for c, k in [(6, "f"), (7, "rho")]:
            check("p5-reference", i, c, r[f"relative_difference_{k}_{pair}"])
for i, r in enumerate(
    sorted(
        timing_summary(),
        key=lambda r: (0 if r["method"] == "mm" else 1, r["angular_budget"]),
    )
):
    for c, k in enumerate(("median", "minimum", "maximum"), 2):
        check("mixed-timing", i, c, r[k])
# Verify current completed APNN cost groups, regardless of later baseline additions.
a = csvs(ROOT / "results/runs/neural_study/summary.csv")
for problem, label in [
    ("p1", "supp-efficiency"),
    ("p3", "supp-efficiency-2d"),
]:
    for i, steps in enumerate((500, 2000, 5000), 7):
        r = next(
            r
            for r in a
            if r["suite"] == "cost"
            and r["problem"] == problem
            and int(r["steps"]) == steps
        )
        for c, k in [
            (4, "E_f_median"),
            (5, "E_rho_median"),
            (6, "train_time_s_median"),
        ]:
            check(label, i, c, r[k])
# Cross-check every error, dimension, and timing in the full cost tables.
from efficiency_refresh_data import efficiency_records

records = efficiency_records()
new2d = [
    json.loads((resolve_path(item["record"])).read_text())
    for item in json.loads(
        (ROOT / "results/runs/feature_study/status.json").read_text()
    )
]
for problem, label in (
    ("p1", "supp-efficiency"),
    ("p3", "supp-efficiency-2d"),
):
    sn = (
        ["32x16", "64x32", "128x64", "256x128"]
        if problem == "p1"
        else ["12x12x16", "16x16x16", "24x24x16", "32x32x16"]
    )
    mm = [32, 64, 128] if problem == "p1" else [64, 128, 256]
    plan = [(i, "OE-$S_N$-Krylov", budget) for i, budget in enumerate(sn)]
    plan += [(i + 4, "MM-APRFM", f"J={j}") for i, j in enumerate(mm)]
    plan += [
        (i + 10, "OE-APRFM", f"J={j}") for i, j in enumerate((32, 64, 128))
    ]
    for i, method, budget in plan:
        rr = [
            r
            for r in records
            if r["problem"] == problem
            and r["method"] == method
            and r["budget"] == budget
        ]
        values = [
            median(float(r[k]) for r in rr)
            for k in (
                "model_size",
                "residual_points",
                "E_f",
                "E_rho",
                "solve_time",
            )
        ]
        if problem == "p3" and method == "OE-APRFM":
            rr = [
                r
                for r in new2d
                if r["problem"] == "p3"
                and r["angular_representation"] == "four_component"
                and r["epsilon"] == 0.001
                and r["features_per_patch"] == int(budget[2:])
            ]
            values = [
                median(r[k] for r in rr)
                for k in (
                    "num_columns",
                    "num_rows",
                    "relative_l2_f",
                    "relative_l2_rho",
                )
            ]
            values += [
                median(
                    sum(
                        r[k]
                        for k in (
                            "feature_seconds",
                            "assembly_seconds",
                            "solve_seconds",
                        )
                    )
                    for r in rr
                )
            ]
        for c, value in enumerate(values, 2):
            check(label, i, c, value)
        selected = (
            problem == "p1"
            and budget == ("32x16" if method == "OE-$S_N$-Krylov" else "J=128")
        ) or (
            problem == "p3"
            and budget
            == (
                "24x24x16"
                if method == "OE-$S_N$-Krylov"
                else "J=256" if method == "MM-APRFM" else "J=128"
            )
        )
        if selected:
            ri = (0 if problem == "p1" else 4) + {
                "OE-$S_N$-Krylov": 0,
                "MM-APRFM": 1,
                "OE-APRFM": 3,
            }[method]
            for c, value in zip(
                (2, 3, 4, 5), (values[0], values[2], values[3], values[4])
            ):
                check("efficiency-summary", ri, c, value)
for i, (problem, eps, rep) in enumerate(
    (p, e, r)
    for p in ("p3", "p4")
    for e in (1.0, 0.001)
    for r in ("legacy_two_component", "four_component")
):
    rr = [
        r
        for r in new2d
        if r["problem"] == problem
        and r["epsilon"] == eps
        and r["angular_representation"] == rep
        and r["num_columns"] == 512
    ]
    values = [
        median(r[k] for r in rr)
        for k in (
            "num_rows",
            "relative_l2_f",
            "relative_l2_rho",
            "condition_number",
        )
    ]
    values += [
        median(
            sum(
                r[k]
                for k in (
                    "feature_seconds",
                    "assembly_seconds",
                    "solve_seconds",
                )
            )
            for r in rr
        )
    ]
    for c, value in enumerate(values, 2):
        check("p34-components", i, c, value)

# New same-objective angular-domain control, independent of old table indices.
angular = json.loads(
    (ROOT / "results/comparisons/angular_domains/summary.json").read_text()
)["summary"]
for i, row in enumerate(angular):
    for column, key in enumerate(
        (
            "angular_locations",
            "num_rows",
            "relative_l2_f",
            "relative_l2_rho",
            "assembly_seconds",
            "solve_seconds",
        ),
        2,
    ):
        check("angular-domain-reduction", i, column, row[key])

# Every indexed source must still exist and retain its recorded hash.
index = json.loads(
    (ROOT / "results/checks/figures/table_source_index.json").read_text()
)
source_issues = []
for r in index:
    for s in r["sources"]:
        p = resolve_path(s["path"])
        if (
            not p.exists()
            or hashlib.sha256(p.read_bytes()).hexdigest() != s["sha256"]
        ):
            source_issues.append(s["path"])
result = dict(
    table_count=len(tables),
    additional_checked_cells=len(checks),
    numeric_mismatches=errors,
    source_hash_issues=source_issues,
    checks=checks,
)
(OUT / "cell_checks.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))
assert not errors
