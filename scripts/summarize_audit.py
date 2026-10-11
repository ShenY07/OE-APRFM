"""Build review-audit CSVs, diagnostic figure, and a plain-text report."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/checks/parity"
ARCHIVE = ROOT / ".release-backup/20261009/original-worktree/results"


def load(path):
    return json.loads(path.read_text())


def write_csv(name, rows):
    with (OUT / name).open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    records = load(OUT / "parity/audit.json")
    assert len(records) == 24
    summary = []
    for J in (16, 32, 64, 128):
        raw = [
            r
            for r in records
            if r["features_per_patch"] == J and r["variant"] == "full_angular"
        ]
        projected = [
            r
            for r in records
            if r["features_per_patch"] == J and r["variant"] == "full"
        ]
        assert sorted(r["seed"] for r in raw) == [11, 23, 37]
        assert sorted(r["seed"] for r in projected) == [11, 23, 37]
        row = dict(J=J, Ncoef=2 * J, Nrow=raw[0]["num_rows"])
        for label, group, key in [
            ("legacy", raw, "relative_l2_"),
            ("folded", raw, "folded_E"),
            ("projected", projected, "relative_l2_"),
        ]:
            for field in ("f", "rho"):
                vals = [r[key + field] for r in group]
                for stat, fn in [
                    ("median", np.median),
                    ("min", np.min),
                    ("max", np.max),
                ]:
                    row[f"{label}_E{field}_{stat}"] = float(fn(vals))
        summary.append(row)
    write_csv("parity_summary.csv", summary)
    mechanisms = load(OUT / "parity_mechanism/audit.json")
    assert len(mechanisms) == 6
    mrows = []
    for eps in (1.0, 0.001, 0.000001):
        a = next(
            r
            for r in mechanisms
            if r["epsilon"] == eps and r["variant"] == "full_angular"
        )
        b = next(
            r
            for r in mechanisms
            if r["epsilon"] == eps and r["variant"] == "full"
        )
        mrows.append(
            dict(
                epsilon=eps,
                seed=11,
                J=64,
                legacy_Ef=a["relative_l2_f"],
                folded_Ef=a["folded_Ef"],
                projected_Ef=b["relative_l2_f"],
            )
        )
    write_csv("mechanism_summary.csv", mrows)
    p3 = load(OUT / "p3/audit.json")
    assert len(p3) == 6
    refreshed = {}
    for p in (ARCHIVE / "efficiency_refresh_2026_10_06/raw").glob(
        "*/*/audit.json"
    ):
        r = load(p)[0]
        if (
            r.get("problem") == "p3"
            and r.get("method") == "oe_aprfm"
            and r.get("features_per_patch") == 128
        ):
            refreshed[r["seed"]] = (p, r)
    prows = []
    for r in p3:
        historical = (
            ARCHIVE
            / "p34_representation_comparison"
            / f"p3_oe_aprfm_eps_{r['epsilon']:.0e}_seed_{r['seed']}_four_component.json"
        )
        old = load(historical)
        row = dict(
            epsilon=r["epsilon"],
            seed=r["seed"],
            new_Nrow=r["num_rows"],
            new_Ef=r["relative_l2_f"],
            archived_15856_Ef=old["relative_l2_f"],
            archived_15856_source=str(historical.relative_to(ROOT)),
            archived_13808_Ef=None,
            archived_13808_source=None,
            delta_13808=None,
        )
        if r["epsilon"] == 0.001:
            p, ref = refreshed[r["seed"]]
            row.update(
                archived_13808_Ef=ref["relative_l2_f"],
                archived_13808_source=str(p.relative_to(ROOT)),
                delta_13808=r["relative_l2_f"] - ref["relative_l2_f"],
            )
        prows.append(row)
    write_csv("p3_comparison.csv", prows)
    frozen = list(
        csv.DictReader(
            (
                ARCHIVE
                / "requirement_2026_08_20/tables_frozen/table_S4_parity_feature_budget.csv"
            ).open()
        )
    )
    table_deltas = []
    for row in frozen:
        variant = (
            "full_angular"
            if row["representation"] == "Unconstrained"
            else "full"
        )
        vals = [
            r["archive_Ef"]
            for r in records
            if r["features_per_patch"] == int(row["J"])
            and r["variant"] == variant
        ]
        table_deltas.append(
            abs(float(row["median_E_f"]) - float(np.median(vals)))
        )
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, field in zip(axs, ("f", "rho")):
        for label, title, color in [
            ("legacy", "Unprojected: legacy reconstruction", "#b23a48"),
            (
                "folded",
                "Unprojected: consistent half-range reconstruction",
                "#2274a5",
            ),
            ("projected", "Parity-projected", "#3d8b55"),
        ]:
            x = [r["Ncoef"] for r in summary]
            y = [r[f"{label}_E{field}_median"] for r in summary]
            ax.loglog(x, y, "o-", label=title, color=color)
            ax.fill_between(
                x,
                [r[f"{label}_E{field}_min"] for r in summary],
                [r[f"{label}_E{field}_max"] for r in summary],
                color=color,
                alpha=0.1,
            )
        ax.set_xlabel("Number of coefficients")
        ax.set_xticks([32, 64, 128, 256], labels=["32", "64", "128", "256"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylabel(f"Relative L2 error ({field})")
        ax.grid(True, which="both", alpha=0.2)
    axs[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(
        "P1 parity audit: epsilon=1e-3; median and range over seeds 11, 23, 37"
    )
    fig.tight_layout()
    fig.savefig(OUT / "parity_reconstruction_audit.png", dpi=180)
    fig.savefig(OUT / "parity_reconstruction_audit.pdf")
    plt.close(fig)
    angles = load(OUT / "angular_checks.json")
    fields = load(OUT / "archive_field_checks.json")
    e6 = load(OUT / "e6/audit.json")
    refs = list(
        csv.DictReader(
            (OUT / "e6/results/e6_periodic/reference_comparison.csv").open()
        )
    )
    metrics = dict(
        parity_archive_max_absolute_error_difference=max(
            abs(r["relative_l2_f"] - r["archive_Ef"]) for r in records
        ),
        parity_frozen_table_max_delta=max(table_deltas),
        archived_field_count=len(fields),
        archived_field_max_error_delta=max(r["delta"] for r in fields),
        folded_boundary_max_mismatch=max(
            r.get("folded_right_trace_mismatch", 0.0) for r in records
        ),
        p3_13808_max_error_delta=max(
            abs(r["delta_13808"])
            for r in prows
            if r["delta_13808"] is not None
        ),
        e6_archive_max_field_delta=max(
            max(r["archive_max_abs_by_field"].values()) for r in e6
        ),
        angular=angles,
    )
    (OUT / "verification_summary.json").write_text(
        json.dumps(metrics, indent=2)
    )
    lines = [
        "OE-APRFM review audit — 2026-10-10",
        "Base commit: "
        + subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "Completed: 24 parity-budget solves + 6 mechanism solves + 6 P3 solves + 2 E6 trajectories.",
        "Existing core regression suite: 22 passed (optional OE-APNN tests excluded); see tests.log.",
        "Scope: current source plus retained local archives. Manuscript full text was unavailable. Historical data were not overwritten.",
        "",
        "1. Parity ablation: substantive reconstruction confound confirmed.",
        "The folded numbers reuse exactly the unprojected fit coefficients; only full-angle evaluation changes.",
        "This is a consistent HALF-RANGE control, not a fully unconstrained full-angle trial space.",
        "J | legacy unprojected Ef median | folded unprojected Ef median | projected Ef median",
    ]
    lines += [
        f"{r['J']} | {r['legacy_Ef_median']:.10e} | {r['folded_Ef_median']:.10e} | {r['projected_Ef_median']:.10e}"
        for r in summary
    ]
    lines += [
        "At J=16 the folded control median is lower than the projected median. The original magnitude of advantage cannot be attributed solely to parity embedding.",
        "Residual row counts vary with J; interpret within-budget comparisons, not the cross-budget curve as fixed-collocation convergence.",
        "See mechanism_summary.csv for the seed-11 epsilon sweep underlying the structural comparison.",
        "",
        "2. P3 provenance: current 13,808-row configuration verified. Follow-up boundary recovery is reported in p3_row_history/REPORT.txt.",
        "Current blocks: macro 2352; even sum/difference 2352+2352; odd sum/difference 2352+2352; physical inflow 2048.",
        "All three epsilon=1e-3 reruns reproduce the newer archived Ef exactly in this environment.",
        "Follow-up recovered all six 15,856-row P3 fields to relative differences below 6e-13 using independent paired boundary rows at physical angles with repeated inflow RHS values.",
        "The 2048-row difference comes from 4096 paired boundary rows versus 2048 physical trace rows. The paired interface assumes first-quadrant angles and does not match physical reconstruction for general sampler angles. The original generating command/commit remains unavailable; the numerical assembly signature has been recovered.",
        "New 13,808-row results at epsilon=1 and 1e-3, all three seeds, are in p3_comparison.csv; do not silently replace old tables without naming the new protocol.",
        "",
        "3. E6 initialization: t=0 is analytic, without fitted coefficients. Both rerun trajectories match archived saved fields exactly.",
        "Independent Fourier references were regenerated with angular orders 128 and 256. Final-time errors:",
        "epsilon | reference | Ef | reference refinement Ef",
    ]
    lines += [
        f"{r['epsilon']} | {r['reference']} | {float(r['Ef']):.10e} | {float(r['reference_delta_f']):.10e}"
        for r in refs
        if float(r["t"]) == 0.2
    ]
    lines += [
        "This does not revalidate all 20 study trajectories or all independent test-grid checks. No evidence from these two reruns of an incorrect BE advance.",
        "",
        "4. Angular partition (1,1,2): current implementation passes independent evaluated-field/operator and boundary/reconstruction checks.",
        json.dumps(angles),
        "These checks establish consistency at the tested points for current code, not historical P5 table provenance or global accuracy.",
        "",
        "5. Periodic objective: separate r/j jumps, each block quadrature sums to one, followed by per-row norm scaling. This differs from an unscaled f-jump penalty.",
        "6. Figure/equation cross-reference cannot be checked without manuscript source or PDF.",
        "",
        "Arithmetic/provenance metrics:",
        json.dumps(metrics, indent=2),
        "",
        "Reproduction: see commands.txt, source_manifest.json, scripts/, and per-run JSON/NPZ files. Timings were not isolated performance measurements.",
    ]
    (OUT / "REPORT.txt").write_text("\n".join(lines) + "\n")
    source_dir = OUT / "scripts"
    source_dir.mkdir(exist_ok=True)
    for p in (ROOT / "scripts").glob("*review*.py"):
        (source_dir / p.name).write_bytes(p.read_bytes())
    manifest = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            *sorted((ROOT / "src").rglob("*.py")),
            *sorted((ROOT / "experiments").glob("*.py")),
            *sorted((ROOT / "scripts").glob("*.py")),
        ]
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    print((OUT / "REPORT.txt").read_text())


if __name__ == "__main__":
    main()
