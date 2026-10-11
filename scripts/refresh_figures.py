"""Reproducible standalone E1--E6 panels; no new solves or reference substitution."""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oe_current_panels_mpl")
import csv, json, hashlib, zipfile
from pathlib import Path
from scripts.paper_paths import paper_file
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / ".release-backup/20261009/original-worktree"
import sys

sys.path.insert(0, str(ROOT / "scripts"))
OUT = REPO / "results/figure/new"
AUDIT = REPO / "results/figure_refresh_20261010"
AUDIT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "pdf.compression": 9,
    }
)
manifest = []
layout_manifest = []
sources = set()


def read(path):
    sources.add(path)
    with (ROOT / path).open() as f:
        return list(csv.DictReader(f))


def val(rows, key):
    return np.array([float(r[key]) for r in rows])


def axes(xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(3.6, 3.6), layout="constrained")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.tick_params(direction="out")
    ax.grid(which="major", alpha=0.16, linewidth=0.5)
    return fig, ax


def save(fig, ax, name, note):
    # Validate rendered labels, independently of ASCII data keys/file names.
    labels = [
        ax.get_xlabel(),
        ax.get_ylabel(),
        *ax.get_legend_handles_labels()[1],
    ]
    assert not ax.get_title(), "Standalone panels must not have titles"
    for label in labels:
        for word in ("rho", "epsilon", "beta", "Delta"):
            assert word not in label.replace("\\varepsilon", "").replace(
                "\\" + word, ""
            ), f"Unformatted mathematical symbol: {label}"
    old_name = name
    # File names must not depend on case alone to distinguish f from flux F.
    replacements = {
        "E2_parity_budget_f": "E2_parity_budget_distribution_error",
        "E2_parity_budget_rho": "E2_parity_budget_density_error",
        "E3_slab_E_diff": "E3_slab_diffusion_density_error",
        "E3_slab_E_q_diff": "E3_slab_diffusion_scaled_current_error",
        "relative_l2_f": "distribution_relative_L2_error",
        "relative_l2_rho": "density_relative_L2_error",
        "residual_half": "normalized_training_residual_RMS",
        "budget_dependence_Ef_": "budget_dependence_distribution_error_",
        "budget_dependence_Erho_": "budget_dependence_density_error_",
        "budget_dependence_EF_": "budget_dependence_physical_flux_error_",
        "fixed_collocation_feature_error": "fixed_collocation_distribution_error_vs_features",
        "_transport_error_three_times": "_distribution_error_three_times",
    }
    for before, after in replacements.items():
        name = name.replace(before, after)
    if old_name != name:
        for ext in ("pdf", "png"):
            old = OUT / f"{old_name}.{ext}"
            new = OUT / f"{name}.{ext}"
            if old.exists() and not new.exists():
                old.rename(new)
    # Common square canvas keeps independently exported panels consistent.
    fig.set_size_inches(3.6, 3.6)
    # Distinct dashes and marker shapes retain information in monochrome.
    styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]
    for i, line in enumerate(ax.lines):
        line.set_linestyle(styles[i % 4])
        if len(line.get_xdata()) <= 10:
            line.set_marker(markers[i % 4])
            line.set_markerfacecolor("white")
        else:
            line.set_markevery(max(1, len(line.get_xdata()) // 10))
    handles, labels = ax.get_legend_handles_labels()
    legend_columns = (
        3
        if name.startswith("E6_") and len(handles) == 3
        else (
            2
            if name.startswith("E6_") and len(handles) == 4
            else 1 if name.startswith("E2_parity") else min(2, len(handles))
        )
    )
    if len(handles) > 1:
        ax.legend(
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=legend_columns,
            handlelength=1.8,
            columnspacing=0.8,
            labelspacing=0.3,
        )
    fig.canvas.draw()
    assert all(not a.get_title() for a in fig.axes)
    legend = ax.get_legend()
    if legend is not None:
        bounds = legend.get_window_extent(fig.canvas.get_renderer())
        assert bounds.x0 >= 0 and bounds.y0 >= 0
        assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
    layout_manifest.append(
        dict(
            name=name,
            size_inches=fig.get_size_inches().tolist(),
            legend_entries=len(handles),
            legend_columns=legend_columns,
            legend_rows=(
                (len(handles) + legend_columns - 1) // legend_columns
                if legend_columns
                else 0
            ),
        )
    )
    for ext in ("pdf", "png"):
        target = paper_file(OUT / f"{name}.{ext}")
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=300)
    # Record exact plotted arrays, including original seed-derived envelopes.
    entries = []
    for line in ax.lines:
        entries.append(
            dict(
                label=line.get_label(),
                x=np.asarray(line.get_xdata()).tolist(),
                y=np.asarray(line.get_ydata()).tolist(),
            )
        )
    plt.close(fig)
    manifest.append(dict(name=name, note=note, curves=entries))


def curve(ax, rows, x, y, label, color, lo=None, hi=None):
    rows = sorted(rows, key=lambda r: float(r[x]))
    xx = val(rows, x)
    yy = val(rows, y)
    ax.plot(xx, yy, "o-", label=label, color=color)
    if lo and hi:
        ax.fill_between(
            xx,
            val(rows, lo),
            val(rows, hi),
            color=color,
            alpha=0.14,
            linewidth=0,
        )


e1 = read("results/minimum_gain/quadrature_check.csv")
fig, ax = axes(r"$\varepsilon$", r"$\beta_{\varepsilon,m}$")
curve(
    ax,
    [r for r in e1 if float(r["epsilon"]) > 0],
    "epsilon",
    "beta",
    "OE",
    BLUE,
)
ax.set_xscale("log")
ax.axhline(
    float(e1[-1]["beta"]), color=ORANGE, ls="--", label=r"$\varepsilon=0$"
)
save(
    fig,
    ax,
    "E1_fixed_space_minimum_gain",
    "32-dimensional diagnostic with H1 x H1 and unweighted inflow L2 norms; refined quadrature; no truncation. Not verification of Assumption 1 for main experiments.",
)
parity_path = REPO / "results/review_audit_20261010/parity/audit.json"
sources.add(str(parity_path))
parity = json.loads(parity_path.read_text())
assert len(parity) == 24
for metric in ("f", "rho"):
    fig, ax = axes(
        r"$N_{\mathrm{coef}}$", {"f": r"$E_f$", "rho": r"$E_{\rho}$"}[metric]
    )
    for variant, label, color in [
        ("full", "Projected", BLUE),
        ("full_angular", "Half-range", ORANGE),
    ]:
        grouped = []
        for j in (16, 32, 64, 128):
            rr = [
                r
                for r in parity
                if r["variant"] == variant and r["features_per_patch"] == j
            ]
            assert sorted(r["seed"] for r in rr) == [11, 23, 37]
            key = (
                ("folded_Ef" if metric == "f" else "folded_Erho")
                if variant == "full_angular"
                else "relative_l2_" + metric
            )
            values = [r[key] for r in rr]
            grouped.append(
                dict(
                    n=2 * j,
                    e=np.median(values),
                    lo=min(values),
                    hi=max(values),
                )
            )
        curve(ax, grouped, "n", "e", label, color, "lo", "hi")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([32, 64, 128, 256], labels=["32", "64", "128", "256"])
    save(
        fig,
        ax,
        f"E2_parity_budget_{metric}",
        "Corrected half-range reconstruction; both solutions have even r and odd j. Counts matched within J, not across J. Three-seed median/range.",
    )
from feature_resolution_data import feature_records

for problem in ("p1", "p3"):
    fig, ax = axes(r"$N_{\mathrm{coef}}$", r"$E_f$")
    rr = [r for r in feature_records() if r["problem"] == problem]
    if problem == "p3":
        rr = []
        for p in sorted(
            (REPO / "results/table_refresh_20261010/raw").glob(
                "p3_four_component_*eps1e-03*/*.json"
            )
        ):
            r = json.loads(p.read_text())
            r["source"] = str(p)
            rr.append(r)
        assert len(rr) == 9
    for r in rr:
        sources.add(r["source"])
    grouped = []
    for n in sorted({r["num_columns"] for r in rr}):
        a = [r["relative_l2_f"] for r in rr if r["num_columns"] == n]
        grouped.append(dict(n=n, e=np.median(a), lo=min(a), hi=max(a)))
    curve(ax, grouped, "n", "e", "OE", BLUE, "lo", "hi")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    save(
        fig,
        ax,
        f"E2_{problem}_fixed_collocation_feature_error",
        "Audited fixed-collocation records; P3 four-component implementation.",
    )
e3 = read("results/e3_slab/summary.csv")
for key, label in [
    ("E_diff", r"$E_{\rho,\mathrm{diff}}$"),
    ("E_q_diff", r"$E_{q,\mathrm{diff}}$"),
    ("fick_defect", r"$D_{\mathrm{Fick}}$"),
]:
    fig, ax = axes(r"$\varepsilon$", label)
    curve(ax, e3, "epsilon", key, "OE", BLUE, key + "_min", key + "_max")
    ax.set_xscale("log")
    ax.set_yscale("log")
    save(
        fig,
        ax,
        "E3_slab_" + key,
        "Fixed data and fixed discretization, three-seed median/range. Evidence limited to this slab configuration.",
    )
p5 = []
for p in sorted(
    (ROOT / "results/comparison_repairs_2026_09_10").glob("p5_*/*.json")
):
    sources.add(str(p.relative_to(ROOT)))
    p5.append(json.loads(p.read_text()))
for key, label in [
    ("relative_l2_f", r"$E_f$"),
    ("relative_l2_rho", r"$E_\rho$"),
    ("rank", r"$r_{\mathrm{eff}}$"),
    ("coefficient_norm", r"$\|\theta\|_2$"),
    ("residual_half", "LS residual RMS"),
]:
    fig, ax = axes(r"SVD cutoff $\tau$", label)
    curve(ax, p5, "rcond", key, "OE", BLUE)
    ax.set_xscale("log")
    if key != "rank":
        ax.set_yscale("log")
    save(
        fig,
        ax,
        "E4_P5_seed11_same_version_" + key,
        "epsilon=1e-3; same-version cutoff sensitivity, not three-seed main results. Residual is complete normalized training residual, not independent physical residual; reference adequacy remains to be checked.",
    )
e5 = read("results/e5_mixed/summary.csv")
for key, label in [("Ef", r"$E_f$"), ("Erho", r"$E_\rho$"), ("EF", r"$E_F$")]:
    fig, ax = axes(r"$N_{\mathrm{row}}$", label)
    for method, color in [("oe", BLUE), ("mm", ORANGE)]:
        curve(
            ax,
            [r for r in e5 if r["method"] == method],
            "Nrow",
            key,
            method.upper(),
            color,
            key + "_min",
            key + "_max",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    save(
        fig,
        ax,
        "E5_budget_dependence_" + key + "_old_reference_pending_reassessment",
        "All eight presets retained. Original reference; new references not yet postprocessed. Actual rows include repeated equations and changing block weights, not independent information. Positive budgets OE/MM each 8,16,32,64 in order.",
    )
from e5_timing_data import timing_records, timing_summary

for record in timing_records():
    sources.add(record["source"])
timing = [
    dict(
        method=r["method"],
        n=r["angular_budget"],
        t=r["median"],
        lo=r["minimum"],
        hi=r["maximum"],
    )
    for r in timing_summary()
]
fig, ax = axes("Positive angular nodes", "Recorded wall time (s)")
for method, color in [("oe", BLUE), ("mm", ORANGE)]:
    curve(
        ax,
        [r for r in timing if r["method"] == method],
        "n",
        "t",
        method.upper(),
        color,
        "lo",
        "hi",
    )
ax.set_xticks([32, 64])
save(
    fig,
    ax,
    "E5_seed11_repeated_wall_time_not_speedup",
    "Seed11 only, three repetitions, feature+assembly+solve. Serial queue does not guarantee isolated host. Not paired with three-seed median errors.",
)
e6 = read("results/e6_mm_comparison/seedwise.csv")
for eps in (1.0, 0.001):
    for kind in ("ct", "be"):
        fig, ax = axes(r"$t$", r"$E_f^{\mathrm{" + kind.upper() + "}}$")
        for method, color in [("OE", BLUE), ("MM", ORANGE)]:
            grouped = []
            for t in (0.02, 0.1, 0.2):
                a = [
                    float(r["Ef"])
                    for r in e6
                    if r["method"] == method
                    and float(r["epsilon"]) == eps
                    and r["reference"] == kind
                    and float(r["t"]) == t
                ]
                assert len(a) == 3
                grouped.append(dict(t=t, e=np.median(a), lo=min(a), hi=max(a)))
            curve(ax, grouped, "t", "e", method, color, "lo", "hi")
        ax.set_yscale("log")
        ax.set_xticks([0.02, 0.1, 0.2])
        save(
            fig,
            ax,
            f"E6_eps{eps:.0e}_{kind}_transport_error_three_times",
            "Three actually evaluated output times only; seed median/range. CT includes time error; BE compares same time step. CT curves may overlap.",
        )
acc = read("results/e6_periodic/accuracy_summary.csv")
fig, ax = axes(r"$\varepsilon$", r"$E_{\rho,\mathrm{diff}}^{\mathrm{BE}}$")
for t, color in [(0.02, BLUE), (0.1, ORANGE), (0.2, GREEN)]:
    curve(
        ax,
        [r for r in acc if float(r["t"]) == t and r["reference"] == "be"],
        "epsilon",
        "Ediff_BE",
        f"$t={t:g}$",
        color,
        "Ediff_BE_min",
        "Ediff_BE_max",
    )
ax.set_xscale("log")
ax.set_yscale("log")
save(
    fig,
    ax,
    "E6_discrete_diffusion_fixed_dt",
    "Difference from BE diffusion, not RF error or CT total error.",
)
tr = read("results/e6_periodic/time_refinement.csv")
fig, ax = axes(r"$\Delta t$", r"$E_f^{\mathrm{CT}}$")
for eps, color in [(1.0, BLUE), (0.001, ORANGE)]:
    curve(
        ax,
        [
            r
            for r in tr
            if float(r["epsilon"]) == eps and r["reference"] == "ct"
        ],
        "dt",
        "Ef",
        r"$\varepsilon=" + str(eps) + "$",
        color,
    )
ax.set_xscale("log", base=2)
ax.set_yscale("log")
save(
    fig,
    ax,
    "E6_time_refinement_seed11_CT",
    "T=0.2 fixed, 50/100/200 BE steps; seed11 only.",
)
for eps in (1.0, 0.001):
    path = next(
        (ROOT / "results/e6_periodic/factorized_check").glob(
            f"s11_eps{eps:.0e}_dt2e-03_J128_q64/fields.npz"
        )
    )
    sources.add(str(path.relative_to(ROOT)))
    d = np.load(path)
    x = d["x"]
    for key, label in [("rho", r"$\rho$"), ("q", r"$q$")]:
        fig, ax = axes(r"$x$", label)
        for step, color in [
            (0, "#666666"),
            (10, BLUE),
            (50, ORANGE),
            (100, GREEN),
        ]:
            ax.plot(
                x, d[f"{key}_{step}"], color=color, label=f"$t={step*.002:g}$"
            )
        save(
            fig,
            ax,
            f"E6_eps{eps:.0e}_seed11_{key}_evolution",
            "OE seed11 actual saved fields; q(0)=0 retained. Time colors consistent across field panels; not a multi-method comparison.",
        )
    fig, ax = axes(r"$t$", "Cosine amplitude")
    refpath = (
        ROOT / f"results/e6_periodic/references/eps{eps:.0e}_dt2e-03_n256.npz"
    )
    sources.add(str(refpath.relative_to(ROOT)))
    ref = np.load(refpath)
    _, w = np.polynomial.legendre.leggauss(256)
    w = w / 2
    ts = [0, 0.02, 0.1, 0.2]
    am = [
        2
        * np.mean((d[f"rho_{int(round(t/.002))}"] - 1) * np.cos(2 * np.pi * x))
        for t in ts
    ]
    ax.plot(ts, am, "o-", color=BLUE, label="OE")
    for kind, color, style in [("ct", ORANGE, "--"), ("be", GREEN, ":")]:
        aa = [0.2] + [
            0.2 * float(np.real(ref[f"{kind}_{t:.2f}"][:256] @ w))
            for t in ts[1:]
        ]
        ax.plot(ts, aa, style, color=color, label=kind.upper(), marker="x")
    save(
        fig,
        ax,
        f"E6_eps{eps:.0e}_seed11_cosine_amplitude_CT_BE",
        "Saved output times only; native reference angular moments. OE and BE may overlap; amplitude exposes background-masked time error.",
    )

# The two existing APNN filenames are retained. Each file has one axis with both
# kinetic and density errors, avoiding embedded subplots and oversized titles.
apnn_path = REPO / "results/oe_apnn_parity_20261010/summary.csv"
sources.add(str(apnn_path))
with apnn_path.open() as stream:
    apnn = list(csv.DictReader(stream))
main_out = OUT
for problem in ("p1", "p3"):
    OUT = REPO / "results/oe_apnn_parity_20261010/figures"
    fig, ax = axes("Adam steps", "Relative error")
    rr = [
        r
        for r in apnn
        if r["suite"] == "cost"
        and r["problem"] == problem
        and float(r["epsilon"]) == 0.001
    ]
    assert len(rr) == 3
    for key, label, color in [
        ("E_f", r"$E_f$", BLUE),
        ("E_rho", r"$E_\rho$", ORANGE),
    ]:
        curve(
            ax,
            rr,
            "steps",
            key + "_median",
            label,
            color,
            key + "_min",
            key + "_max",
        )
    ax.set_yscale("log")
    ax.set_xticks([500, 2000, 5000])
    fig.set_size_inches(3.6, 3.6)
    save(
        fig,
        ax,
        problem + "_cost_accuracy",
        "Hard-parity APNN; existing filename retained, kinetic and density errors on one axis. No numerical changes.",
    )
OUT = main_out
(AUDIT / "layout.json").write_text(
    json.dumps(layout_manifest, indent=2) + "\n"
)
(AUDIT / "plot_data.json").write_text(json.dumps(manifest, indent=2) + "\n")
resolved_sources = []
for p in sorted(sources):
    p = Path(p)
    p = p if p.is_absolute() else ROOT / p
    assert p.exists(), p
    resolved_sources.append(
        dict(
            path=str(p.relative_to(REPO)),
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        )
    )
(AUDIT / "figure_sources.json").write_text(
    json.dumps(resolved_sources, indent=2) + "\n"
)
print(json.dumps(dict(panels=len(manifest), output=str(OUT))))
