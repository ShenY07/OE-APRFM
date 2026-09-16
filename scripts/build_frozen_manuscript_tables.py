#!/usr/bin/env python3
"""Build booktabs manuscript tables using the frozen reporting conventions."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from feature_resolution_data import feature_summary, export as export_features
from p2_corrected_results import corrected_records

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "requirement_2026_08_20" / "tables_frozen"


def load(pattern):
    return [json.loads(path.read_text()) for path in sorted(ROOT.glob(pattern))]


def sci(value):
    if value in (None, ""):
        return r"---"
    return f"{float(value):.2e}".replace("e+00", "").replace("e-0", "e-").replace("e+0", "e")


def latex_sci(value):
    if value in (None, ""):
        return r"---"
    mantissa, exponent = f"{float(value):.2e}".split("e")
    exponent = int(exponent)
    return f"${mantissa}\\times10^{{{exponent}}}$" if exponent else f"${mantissa}$"


def eps_tex(value):
    exponent = int(round(np.log10(float(value))))
    return "$1$" if exponent == 0 else f"$10^{{{exponent}}}$"


def write_csv(name, rows):
    if not rows: return
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def environment(number, caption, label, columns, header, body):
    lines = [f"% Table {number}: {caption}", r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{columns}}}", r"\toprule", header + r"\\", r"\midrule"]
    rules = {r"\midrule", r"\addlinespace"}
    lines.extend(row if row in rules else row + r"\\" for row in body)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (OUT / f"table_{number}.tex").write_text("\n".join(lines))
    return lines


def median_groups(rows, keys, metrics):
    grouped = defaultdict(list)
    for row in rows: grouped[tuple(row[k] for k in keys)].append(row)
    output=[]
    for group, members in sorted(grouped.items()):
        item=dict(zip(keys, group)); item["seeds"]=len(members)
        for metric in metrics: item[metric]=float(np.median([m[metric] for m in members]))
        output.append(item)
    return output


def main():
    p2=[r for _,r in corrected_records()]
    OUT.mkdir(parents=True, exist_ok=True); all_tex=[]
    t1 = [
        ("1D manufactured", r"$1\times1$", 64, 128, 2944, 8, r"$10^{-12}$"),
        ("1D heterogeneous", r"$2\times4$", 64, 1024, p2[0]['num_rows'], 8, r"$10^{-6}$"),
        ("2D manufactured (four component)", r"$1\times1\times1$", 128, 512, 15856, 32, r"$10^{-12}$"),
        ("Perforated domain (four component)", r"$1\times1\times1$", 128, 512, 14736, 32, r"$10^{-12}$"),
        ("P5 smooth heterogeneous", r"$1\times1\times2$", 128, 1024, 58096, 16, r"$10^{-6}$"),
    ]
    write_csv("table_1_numerical_configurations.csv", [dict(problem=a,partition=b,J=c,Ncoef=d,Nrow=e,angular_order=f,rcond=g) for a,b,c,d,e,f,g in t1])
    all_tex += environment(1, "Principal numerical configurations. P3--P5 use the corrected four-component $(j_1,r_1,j_2,r_2)$ representation. For P5, $J=128$ is used per component and angular patch; both Knudsen regimes use the same frozen approximation space.", "tab:configurations", "lrrrrrr", r"Problem & Partition & $J$ & $N_{\rm coef}$ & $N_{\rm row}$ & $N_{\rm ang}$ & \texttt{rcond}", [f"{a} & {b} & {c} & {d} & {e} & {f} & {g}" for a,b,c,d,e,f,g in t1])

    p1_scan=[r for r in load("results/epsilon_scan/*.json") if r["problem"] == "p1"]
    p3_four=[r for r in load("results/p34_representation_comparison/*.json")
             if r["problem"] == "p3" and r.get("angular_representation") == "four_component"]
    scan=p1_scan+p3_four
    write_csv("table_S2a_knudsen_seedwise.csv", [{k:r.get(k,"") for k in ("problem","epsilon","seed","relative_l2_f","relative_l2_rho","empirical_stability_ratio")} for r in scan])
    t2=median_groups(scan,("problem","epsilon"),("relative_l2_f","relative_l2_rho","empirical_stability_ratio"))
    t2.sort(key=lambda row: (("p1", "p3").index(row["problem"]), -row["epsilon"]))
    write_csv("table_2_accuracy_knudsen.csv",t2); body=[]; previous=None
    for row in t2:
        problem={"p1":"1D manufactured","p3":"2D manufactured"}[row["problem"]]
        if previous and previous != row["problem"]: body.append(r"\midrule")
        shown=problem if previous != row["problem"] else ""
        body.append(f"{shown} & {eps_tex(row['epsilon'])} & {latex_sci(row['relative_l2_f'])} & {latex_sci(row['relative_l2_rho'])} & ${row['empirical_stability_ratio']:.3f}$")
        previous=row["problem"]
    all_tex += environment(2, "Knudsen-regime accuracy and empirical stability. Values are three-seed medians. The P3 entries are the updated four-component first-quadrant results; no new four-component $\\varepsilon=10^{-6}$ result is available, so none is inferred from the legacy representation.", "tab:knudsen-accuracy", "lcccc", r"Problem & $\varepsilon$ & $E_f$ & $E_\rho$ & $C_{\rm emp}$", body)

    coll=load("results/collocation_sufficiency/*.json")
    t3=median_groups(coll,("problem","oversampling_ratio"),("relative_l2_f","relative_l2_rho")); write_csv("table_3_collocation_refinement.csv",t3)
    body=[]; previous=None
    for row in t3:
        problem={"p1":"1D manufactured","p3":"2D manufactured"}[row["problem"]]
        if previous and previous != row["problem"]: body.append(r"\midrule")
        body.append(f"{problem if previous != row['problem'] else ''} & ${row['oversampling_ratio']:.3f}$ & {latex_sci(row['relative_l2_f'])} & {latex_sci(row['relative_l2_rho'])}"); previous=row["problem"]
    all_tex += environment(3, "Legacy two-component collocation-refinement study at $J=128$, $N_{\\rm coef}=256$, and $\\varepsilon=10^{-3}$. These retained data measure collocation sufficiency and are not mixed with the new four-component principal P3/P4 results. Values are three-seed medians and $\\eta_{\\rm col}=N_{\\rm row}/N_{\\rm coef}$.", "tab:collocation", "lccc", r"Problem & $\eta_{\rm col}$ & $E_f$ & $E_\rho$", body)

    mechanism=load("results/ablation/*.json")
    mechanism += [r for r in load("results/baselines/rfm/p1_*.json") if r.get("seed")==11]
    mechanism += [r for r in load("results/baselines/mm_aprfm/p1_*.json") if r.get("seed")==11]
    names={"rfm":"Direct RFM","mm_aprfm":"MM-APRFM","oe_original":"OE w/o proj./rescale","full":"OE-APRFM","full_angular":"Unconstrained","parity":"Parity-constrained"}
    keyed=defaultdict(dict); nrows={}
    for r in mechanism:
        key=r.get("variant",r.get("method"))
        if key in names: keyed[key][r["epsilon"]]=r["relative_l2_f"]; nrows[key]=r.get("num_rows", "---")
    body=[r"\multicolumn{5}{l}{\emph{A. Transport formulation}}", r"\addlinespace"]
    for key in ("rfm","mm_aprfm","oe_original","full"):
        body.append(f"{names[key]} & {nrows.get(key,'---')} & " + " & ".join(latex_sci(keyed[key].get(e)) for e in (1.,1e-3,1e-6)))
    body += [r"\addlinespace",r"\multicolumn{5}{l}{\emph{B. Angular approximation space}}",r"\addlinespace"]
    for key in ("full_angular","full"):
        label=names[key] if key!="full" else "Parity-constrained"
        body.append(f"{label} & {nrows.get(key,'---')} & " + " & ".join(latex_sci(keyed[key].get(e)) for e in (1.,1e-3,1e-6)))
    all_tex += environment(4, "Structural comparison for seed 11. All configurations use $N_{\\rm coef}=128$. Residual dimensions differ in Block A, so it is not a fixed-cost benchmark. In Block B, the projection-rescaled OE equations are fixed and only the angular approximation space is changed; both configurations have identical coefficient and residual dimensions.", "tab:mechanism", "lrrrr", r"Configuration & $N_{\rm row}$ & $E_f(1)$ & $E_f(10^{-3})$ & $E_f(10^{-6})$", body)

    efficiency_raw=[]
    for path in sorted((ROOT/"results/requirement_2026_08_20/efficiency_raw").glob("oe_*/*.json")):
        r=json.loads(path.read_text())
        # The original P3 efficiency files use the legacy two-component
        # representation.  Keep the P1 files, but never mix legacy P3 data
        # into the final four-component accuracy--cost table.
        if r["problem"] == "p3":
            continue
        efficiency_raw.append(dict(problem=r["problem"],epsilon=r["epsilon"],method="OE-APRFM",seed=r["seed"],budget=f"J={r['features_per_patch']}",model_size=r["num_columns"],residual_points=r["num_rows"],E_f=r["relative_l2_f"],E_rho=r["relative_l2_rho"],solve_time=r["feature_seconds"]+r["assembly_seconds"]+r["solve_seconds"],eval_time=r["evaluation_seconds"]))
    for path in sorted((ROOT/"results/requirement_2026_08_20/efficiency_raw_four_component_normalized").glob("oe_p3_*/*.json")):
        r=json.loads(path.read_text())
        if r.get("angular_representation") != "four_component":
            raise ValueError(f"non-four-component P3 efficiency record: {path}")
        efficiency_raw.append(dict(problem=r["problem"],epsilon=r["epsilon"],method="OE-APRFM",seed=r["seed"],budget=f"J={r['features_per_patch']}",model_size=r["num_columns"],residual_points=r["num_rows"],E_f=r["relative_l2_f"],E_rho=r["relative_l2_rho"],solve_time=r["feature_seconds"]+r["assembly_seconds"]+r["solve_seconds"],eval_time=r["evaluation_seconds"]))
    for path in sorted((ROOT/"results/requirement_2026_08_20/efficiency_raw").glob("mm_*/*.json")):
        r=json.loads(path.read_text()); feature=r.get("features_per_field_patch",r.get("rho_features_per_patch")); efficiency_raw.append(dict(problem=r["problem"],epsilon=r["epsilon"],method="MM-APRFM",seed=r["seed"],budget=f"J={feature}",model_size=r["num_columns"],residual_points=r["num_rows"],E_f=r["relative_l2_f"],E_rho=r["relative_l2_rho"],solve_time=r["feature_seconds"]+r["assembly_seconds"]+r["solve_seconds"],eval_time=r["evaluation_seconds"]))
    for path in sorted((ROOT/"results/requirement_2026_08_20/efficiency_raw").glob("apnn_*/metrics.json")):
        r=json.loads(path.read_text()); efficiency_raw.append(dict(problem=r["problem"].lower(),epsilon=r["epsilon"],method="OE-APNN",seed=r["seed"],budget=f"{r['steps']} steps",model_size=r["parameter_count"],residual_points=r["n_int_per_step"]+r["n_bdy_per_step"],E_f=r["E_f"],E_rho=r["E_rho"],solve_time=r["train_time_s"],eval_time=r["evaluation_seconds"]))
    for path in sorted((ROOT/"results/requirement_2026_08_20/oe_dsa_si_sweep").glob("p3_oe_dsa_si_N*_A16_eps_*.json")):
        r=json.loads(path.read_text())
        if r.get("converged"):
            nx,ny,na=r["grid"]; nang=4*na
            efficiency_raw.append(dict(problem="p3",epsilon=r["epsilon"],method=r"OE-$S_N$-Krylov",seed="deterministic",budget=f"{nx}x{ny}x{nang}",model_size=nx*ny*nang,residual_points=nx*ny*nang,E_f=r["relative_l2_f"],E_rho=r["relative_l2_rho"],solve_time=r["runtime_seconds"],eval_time=0.0))
    for path in sorted((ROOT/"results/requirement_2026_08_20/oe_sn_krylov_verified_gauss").glob("p1_oe_sn_krylov_N*_eps_*.json")):
        r=json.loads(path.read_text())
        if r.get("converged"):
            nx,nang=r["grid"]
            efficiency_raw.append(dict(problem="p1",epsilon=r["epsilon"],method=r"OE-$S_N$-Krylov",seed="deterministic",budget=f"{nx}x{nang}",model_size=nx*nang,residual_points=nx*nang,E_f=r["relative_l2_f"],E_rho=r["relative_l2_rho"],solve_time=r["runtime_seconds"],eval_time=0.0))
    method_order={r"OE-$S_N$-Krylov":0,"MM-APRFM":1,"OE-APNN":2,"OE-APRFM":3}
    write_csv("table_S2b_efficiency_seedwise.csv",efficiency_raw)
    groups=defaultdict(list)
    for row in efficiency_raw: groups[(row["problem"],row["method"],row["budget"],row["model_size"],row["residual_points"])].append(row)
    efficiency=[]
    for key,members in groups.items():
        problem,method,budget,model_size,residual_points=key
        efficiency.append(dict(problem=problem,method=method,budget=budget,model_size=model_size,residual_points=residual_points,seeds=len(members),E_f=float(np.median([m["E_f"] for m in members])),E_rho=float(np.median([m["E_rho"] for m in members])),solve_time=float(np.median([m["solve_time"] for m in members])),eval_time=float(np.median([m["eval_time"] for m in members]))))
    def budget_order(row):
        if row["method"] == "OE-APNN":
            return int(row["budget"].split()[0])
        return int(row["model_size"])
    efficiency.sort(key=lambda r: (("p1","p3").index(r["problem"]),method_order[r["method"]],budget_order(r)))
    write_csv("table_S3_accuracy_cost_sweep.csv",efficiency)
    body=[]; previous_problem=None; previous_method=None
    for row in efficiency:
        if previous_problem and previous_problem != row["problem"]: body.append(r"\midrule")
        problem={"p1":"1D manufactured","p3":"2D manufactured"}[row["problem"]] if previous_problem != row["problem"] else ""
        method=row["method"] if previous_problem != row["problem"] or previous_method != row["method"] else ""
        body.append(f"{problem} & {method} & {row['budget']} & {row['model_size']} & {row['residual_points']} & {latex_sci(row['E_f'])} & {latex_sci(row['E_rho'])} & ${row['solve_time']:.2f}$")
        previous_problem,previous_method=row["problem"],row["method"]
    supplement_tex=[]
    ref_dir=ROOT/"results/references_p5_boundary"
    ref_rows=[]
    for epsilon in (1.0, 1.0e-3):
        for level in ("A","B","C"):
            meta_path=ref_dir/f"p5_parity_ref_eps_{epsilon:.0e}_level_{level}.json"
            if meta_path.exists(): ref_rows.append(json.loads(meta_path.read_text()))
    comparisons={}
    for epsilon in (1.0, 1.0e-3):
        for pair in ("BA","CB"):
            path=ref_dir/f"p5_parity_ref_eps_{epsilon:.0e}_refinement_{pair}.json"
            if path.exists(): comparisons[(epsilon,pair)]=json.loads(path.read_text())
    s1_body=[]
    previous_epsilon=None
    for r in ref_rows:
        nx,ny,nang=r["grid"]
        pair={"B":"BA","C":"CB"}.get(r["level"])
        comparison=comparisons.get((r["epsilon"],pair),{}) if pair else {}
        ef=comparison.get(f"relative_difference_f_{pair}") if pair else None
        er=comparison.get(f"relative_difference_rho_{pair}") if pair else None
        shown_epsilon=eps_tex(r["epsilon"]) if r["epsilon"] != previous_epsilon else ""
        if previous_epsilon is not None and r["epsilon"] != previous_epsilon:
            s1_body.append(r"\midrule")
        s1_body.append(f"{shown_epsilon} & {r['level']} & ${nx}\\times{ny}$ & {nang} & {r['iterations']} & {latex_sci(r['relative_residual'])} & ${r['runtime_seconds']:.2f}$ & {latex_sci(ef)} & {latex_sci(er)}")
        previous_epsilon=r["epsilon"]
    supplement_tex += environment("S1", "Certified deterministic-reference refinement for the smooth boundary-driven P5 problem. Both kinetic and density B/C discrepancies are below 10\\% of the corresponding median OE-APRFM errors, so both $E_f$ and $E_\\rho$ are reportable.", "tab:p5-reference", "clrrrrrrr", r"$\varepsilon$ & Level & Spatial grid & $N_{\rm ang}$ & Iterations & Rel. residual & CPU (s) & $\delta_f$ & $\delta_\rho$", s1_body)
    supplement_tex += environment("S3", "Accuracy--cost sweep at $\\varepsilon=10^{-3}$. Randomized-method values are medians over seeds 11, 23, and 37; the deterministic OE--$S_N$--Krylov entries are single reproducible solves with Gauss--Legendre quadrature, a common $10^{-12}$ tolerance, and true residuals below $10^{-12}$. Since the 1D linear manufactured solution is represented exactly by diamond difference, its $10^{-11}$--$10^{-10}$ errors are an algebraic roundoff floor rather than a discretization-convergence curve. The 2D OE-APRFM entries use the final normalized four-component representation. Solve time excludes independent test-grid evaluation. Budget definitions are method-specific. All runs use CPU float64 in the same environment.", "tab:supp-efficiency", "llrrrrrr", r"Problem & Method & Budget & Model size & Residual budget & $E_f$ & $E_\rho$ & $T_{\rm solve}$ (s)", body)

    parity_raw=load("results/requirement_2026_08_20/parity_budget_raw/*.json")
    parity_groups=defaultdict(list)
    for row in parity_raw:
        parity_groups[(row["variant"],row["features_per_patch"],row["num_columns"],row["num_rows"],row["oversampling_ratio"])].append(row)
    parity=[]
    representation_names={"full_angular":"Unconstrained","full":"Parity-constrained"}
    for key,members in parity_groups.items():
        variant,J,ncoef,nrow,eta=key
        ef=np.asarray([row["relative_l2_f"] for row in members])
        erho=np.asarray([row["relative_l2_rho"] for row in members])
        parity.append(dict(representation=representation_names[variant],J=J,Ncoef=ncoef,Nrow=nrow,
                           eta_col=eta,seeds=len(members),median_E_f=float(np.median(ef)),
                           min_E_f=float(np.min(ef)),max_E_f=float(np.max(ef)),
                           median_E_rho=float(np.median(erho)),min_E_rho=float(np.min(erho)),
                           max_E_rho=float(np.max(erho))))
    parity.sort(key=lambda row: (("Unconstrained","Parity-constrained").index(row["representation"]),row["Ncoef"]))
    write_csv("table_S4_parity_feature_budget.csv",parity)
    s4_body=[]; previous=None
    for row in parity:
        shown=row["representation"] if row["representation"] != previous else ""
        s4_body.append(f"{shown} & {row['J']} & {row['Ncoef']} & {row['Nrow']} & ${row['eta_col']:.3f}$ & {latex_sci(row['median_E_f'])} & {latex_sci(row['median_E_rho'])}")
        previous=row["representation"]
    supplement_tex += environment("S4", "Parity feature-budget sweep at $\\varepsilon=10^{-3}$. Values are medians over seeds 11, 23, and 37. At every budget, the unconstrained and parity-constrained spaces use identical coefficient and residual dimensions; $\\eta_{\\rm col}=N_{\\rm row}/N_{\\rm coef}$.", "tab:parity-budget", "llrrrrr", r"Angular space & $J$ & $N_{\rm coef}$ & $N_{\rm row}$ & $\eta_{\rm col}$ & $E_f$ & $E_\rho$", s4_body)

    p4=[r for r in load("results/p34_representation_comparison/*.json")
        if r.get("problem") == "p4" and r.get("seed") in (11,23,37)
        and r.get("angular_representation") == "four_component"]
    extended=median_groups(p2,("problem","epsilon"),("relative_l2_f","relative_l2_rho"))
    extended += median_groups(p4,("problem","epsilon"),("relative_l2_f","relative_l2_rho"))
    p5_summary_path=ROOT/"results/requirement_2026_08_20/../raw/p5_boundary_formal/p5_frozen_three_seed_summary_level_C.json"
    if p5_summary_path.exists():
        p5_summary=json.loads(p5_summary_path.read_text())["medians"]
        for epsilon,key in ((1.0,"1e+00"),(1.0e-3,"1e-03")):
            extended.append(dict(problem="p5",epsilon=epsilon,seeds=3,
                                 relative_l2_f=p5_summary[key]["relative_l2_f"],
                                 relative_l2_rho=p5_summary[key]["relative_l2_rho"]))
    extended.sort(key=lambda row: (("p2", "p4", "p5").index(row["problem"]), -row["epsilon"]))
    write_csv("table_5_extended_examples.csv",extended)
    body=[]; previous=None
    for row in extended:
        problem={"p2":"1D heterogeneous","p4":"Perforated domain","p5":"P5 smooth boundary-driven heterogeneous"}[row["problem"]]
        if previous and previous != row["problem"]: body.append(r"\midrule")
        body.append(f"{problem if previous != row['problem'] else ''} & {eps_tex(row['epsilon'])} & {latex_sci(row['relative_l2_f'])} & {latex_sci(row['relative_l2_rho'])}"); previous=row["problem"]
    all_tex += environment(5, "Extended numerical examples. Values are medians over seeds 11, 23, and 37. P2 uses consistent continuous partition-of-unity functions in assembly and evaluation. P4 and P5 use the corrected four-component formulation. P5 errors are evaluated against the certified level-C deterministic reference.", "tab:extended", "lccc", r"Problem & $\varepsilon$ & $E_f$ & $E_\rho$", body)

    p6=load("results/requirement_2026_08_20/p6_raw/*.json")
    p6_summary=median_groups(p6,("epsilon",),("relative_l2_f","relative_l2_rho","relative_l2_r","relative_l2_j"))
    p6_summary.sort(key=lambda row:-row["epsilon"])
    write_csv("table_6_angular_manufactured.csv",p6_summary)
    p6_body=[f"{eps_tex(row['epsilon'])} & {latex_sci(row['relative_l2_f'])} & {latex_sci(row['relative_l2_rho'])} & {latex_sci(row['relative_l2_r'])} & {latex_sci(row['relative_l2_j'])}" for row in p6_summary]
    all_tex += environment(6, "Angular-dependent manufactured test with a fixed approximation space ($N_{\\rm coef}=128$, $N_{\\rm row}=2944$, and $N_{\\rm ang}=8$). Values are medians over seeds 11, 23, and 37.", "tab:angular-manufactured", "ccccc", r"$\varepsilon$ & $E_f$ & $E_\rho$ & $E_r$ & $E_j$", p6_body)
    efficiency_body=[]; previous_problem=None; previous_method=None
    for row in efficiency:
        if previous_problem and previous_problem != row["problem"]: efficiency_body.append(r"\midrule")
        problem={"p1":"1D manufactured","p3":"2D manufactured"}[row["problem"]] if previous_problem != row["problem"] else ""
        method=row["method"] if previous_problem != row["problem"] or previous_method != row["method"] else ""
        efficiency_body.append(f"{problem} & {method} & {row['budget']} & {row['model_size']} & {latex_sci(row['E_f'])} & {latex_sci(row['E_rho'])} & ${row['solve_time']:.2f}$")
        previous_problem,previous_method=row["problem"],row["method"]
    all_tex += environment(7, "Accuracy--cost comparison at $\\varepsilon=10^{-3}$. The 2D OE-APRFM sweep uses the final normalized four-component representation at matched total coefficient budgets. Randomized values are three-seed medians. Deterministic OE--$S_N$--Krylov values use Gauss--Legendre quadrature, one solver, a common $10^{-12}$ tolerance, and true relative residuals below $10^{-12}$. The 1D linear manufactured solution is represented exactly by diamond difference, so its $10^{-11}$--$10^{-10}$ errors reflect the algebraic roundoff floor and are not interpreted as a grid-convergence curve.", "tab:accuracy-cost", "lllrrrr", r"Problem & Method & Budget & Size & $E_f$ & $E_\rho$ & $T_{\rm solve}$ (s)", efficiency_body)

    p34_comparison=load("results/p34_representation_comparison/*.json")
    comparison_groups=defaultdict(list)
    for row in p34_comparison:
        comparison_groups[(row["problem"],row["epsilon"],row["angular_representation"])].append(row)
    comparison=[]
    representation_names={"legacy_two_component":"Two component","four_component":"Four component"}
    angle_domains={"legacy_two_component":r"$[0,\pi]$","four_component":r"$[0,\pi/2]$"}
    for (problem,epsilon,representation),members in sorted(comparison_groups.items()):
        comparison.append(dict(
            problem=problem,epsilon=epsilon,representation=representation_names[representation],
            angle_domain=angle_domains[representation],components=4 if representation=="four_component" else 2,
            J=int(np.median([m["features_per_patch"] for m in members])),
            Ncoef=int(np.median([m["num_columns"] for m in members])),
            Nrow=int(np.median([m["num_rows"] for m in members])),
            E_f=float(np.median([m["relative_l2_f"] for m in members])),
            E_rho=float(np.median([m["relative_l2_rho"] for m in members])),
            condition=float(np.median([m["condition_number"] for m in members])),
            comp_time=float(np.median([
                m["feature_seconds"] + m["assembly_seconds"] + m["solve_seconds"]
                for m in members
            ])),seeds=len(members)))
    comparison.sort(key=lambda row: (("p3","p4").index(row["problem"]),-row["epsilon"],
                                     ("Two component","Four component").index(row["representation"])))
    write_csv("table_8_p34_two_vs_four_component.csv",comparison)
    comparison_body=[]; previous_problem=None; previous_epsilon=None
    for row in comparison:
        if previous_problem and previous_problem != row["problem"]: comparison_body.append(r"\midrule")
        problem={"p3":"2D manufactured","p4":"Perforated domain"}[row["problem"]] if previous_problem != row["problem"] else ""
        shown_epsilon=eps_tex(row["epsilon"]) if previous_problem != row["problem"] or previous_epsilon != row["epsilon"] else ""
        comparison_body.append(
            f"{problem} & {shown_epsilon} & {row['representation']} & {row['angle_domain']} & "
            f"${row['components']}\\times{row['J']}$ & {row['Nrow']} & {latex_sci(row['E_f'])} & "
            f"{latex_sci(row['E_rho'])} & {latex_sci(row['condition'])} & ${row['comp_time']:.2f}$")
        previous_problem,previous_epsilon=row["problem"],row["epsilon"]
    all_tex += environment(8, "P3/P4 two-component versus four-component angular representations at equal total coefficient budget $N_{\\rm coef}=512$. Values are medians over seeds 11, 23, and 37 with $16\\times16\\times16$ collocation and 32-point angular quadrature. The four-component formulation samples only $[0,\\pi/2]$; the legacy two-component formulation samples $[0,\\pi]$. Computation time is $T_{\\rm comp}=T_{\\rm feature}+T_{\\rm assembly}+T_{\\rm linear}$ and excludes test-grid evaluation.", "tab:p34-components", "lllrcrrrrr", r"Problem & $\varepsilon$ & Representation & Sampled angle & Components$\times J$ & $N_{\rm row}$ & $E_f$ & $E_\rho$ & $\kappa(A)$ & $T_{\rm comp}$ (s)", comparison_body)
    export_features(OUT)
    export_features(ROOT / "results/requirement_2026_08_20/tables")
    feature_body = []
    for row in feature_summary():
        name = "1D manufactured" if row["problem"] == "p1" else "2D manufactured (four component)"
        feature_body.append(f"{name} & {row['features_per_patch']} & {row['num_columns']} & {row['num_rows']} & {latex_sci(row['relative_l2_f'])} & {latex_sci(row['relative_l2_rho'])} & ${row['rank_fraction']:.6f}$")
    supplement_tex += environment("S5", "Random-feature resolution at $\\varepsilon=10^{-3}$ using the same runs as Table 7. Values are medians over seeds 11, 23, and 37. Residual dimensions are fixed within each problem; all runs use $\\texttt{rcond}=10^{-12}$. P3 uses the final normalized four-component representation. No legacy two-component P3 results are included. The rank fraction is the median of seedwise effective-rank ratios.", "tab:feature-resolution", "lrrrrrr", r"Problem & $J$ & $N_{\rm coef}$ & $N_{\rm row}$ & $E_f$ & $E_\rho$ & $r_{\rm eff}/N_{\rm coef}$", feature_body)
    # The consolidated file intentionally includes both main and supplementary
    # tables; the standalone supplement file is retained for manuscript use.
    (OUT/"all_available_tables.tex").write_text("\n".join(all_tex + supplement_tex))
    (OUT/"supplement_tables.tex").write_text("\n".join(supplement_tex))
    (OUT/"README.md").write_text("# Frozen manuscript tables\n\n`all_available_tables.tex` contains thirteen available tables: eight main tables plus supplementary Tables S1, S3, the derived S3a accuracy-target comparison, S4, and S5. The separate `supplement_tables.tex` file is retained for manuscript assembly. The 2D OE-APRFM accuracy--cost sweep uses the final normalized four-component formulation. Legacy two-component efficiency files remain archived but are excluded. P5 errors are certified against level-C deterministic references.\n")


    with (OUT/"README.md").open("a") as handle:
        handle.write("\nThe S3a companion is derived from the existing S3 aggregate CSV without new measurements. Regenerate it with `python3 scripts/build_s3_target_accuracy.py`. P7 transient pilot results remain separate in `results/p7_transient_pilot/three_seed_J128/REPORT.md`.\n")
        handle.write("\nP2 uses the six corrected PoU runs in `results/pou_fix_2026_09_08/p2`; table 1 uses their actual row count. The P2 publication panels share this source. Old consistency/epsilon-scan P2 records are retained for comparison only.\n")


if __name__ == "__main__":
    main()
    from build_s3_target_accuracy import update as update_target_accuracy
    update_target_accuracy()
